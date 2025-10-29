#include "InviscidScheme.cuh"
#include "Mesh.h"
#include "Parameter.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Reconstruction.cuh"
#include "RiemannSolver.cuh"
#include "Parallel.h"

namespace cfd {
template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};
  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                            // fc
                     + n_computation_per_block * (n_var + 3 + 1)) * sizeof(real); // pv[n_var]+metric[3]+jacobian

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const int tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const int block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = static_cast<int>((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];
  memset(&fc[tid * n_var], 0, n_var * sizeof(real));

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 0; l < 3; ++l) {
    metric[i_shared * 3 + l] = zone->metric(idx[0], idx[1], idx[2], direction * 3 + l);
    //    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < 3; ++l) {
        metric[ig_shared * 3 + l] = zone->metric(g_idx[0], g_idx[1], g_idx[2], direction * 3 + l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < 3; ++l) {
        metric[ig_shared * 3 + l] = zone->metric(g_idx[0], g_idx[1], g_idx[2], direction * 3 + l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      riemannSolver_laxFriedrich<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      riemannSolver_hllc<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, const int idx_shared, DParameter *param) {
  auto n_var = param->n_var;
  switch (param->reconstruction) {
    case 2:
      MUSCL_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    case 3:
      NND2_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    default:
      first_order_reconstruct(pv, pv_l, pv_r, idx_shared, n_var);
  }
  if constexpr (mix_model != MixtureModel::Air) {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    const auto n_spec = param->n_spec;
    real mw_inv_l{0.0}, mw_inv_r{0.0};
    for (int l = 0; l < n_spec; ++l) {
      mw_inv_l += pv_l[5 + l] * param->imw[l];
      mw_inv_r += pv_r[5 + l] * param->imw[l];
    }
    const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
    const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

    real hl[MAX_SPEC_NUMBER], hr[MAX_SPEC_NUMBER], cpl_i[MAX_SPEC_NUMBER], cpr_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
    compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
    real cpl{0}, cpr{0}, cvl{0}, cvr{0};
    for (auto l = 0; l < n_spec; ++l) {
      cpl += cpl_i[l] * pv_l[l + 5];
      cpr += cpr_i[l] * pv_r[l + 5];
      cvl += pv_l[l + 5] * (cpl_i[l] - param->gas_const[l]);
      cvr += pv_r[l + 5] * (cpr_i[l] - param->gas_const[l]);
      el += hl[l] * pv_l[l + 5];
      er += hr[l] * pv_r[l + 5];
    }
    pv_l[n_var] = pv_l[0] * el - pv_l[4]; //total energy
    pv_r[n_var] = pv_r[0] * er - pv_r[4];

    pv_l[n_var + 1] = cpl / cvl; //specific heat ratio
    pv_r[n_var + 1] = cpr / cvr;
  } else {
    const real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    const real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    pv_l[n_var] = el * pv_l[0] + pv_l[4] / (gamma_air - 1);
    pv_r[n_var] = er * pv_r[0] + pv_r[4] / (gamma_air - 1);
  }
}

template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};

  // Compute the entropy fix delta
  dim3 thread_per_block{8, 8, 4};
  if (extent[2] == 1) {
    thread_per_block = {16, 16, 1};
  }
  dim3 block_per_grid{
    (extent[0] + 1) / thread_per_block.x + 1,
    (extent[1] + 1) / thread_per_block.y + 1,
    (extent[2] + 1) / thread_per_block.z + 1
  };
  compute_entropy_fix_delta<mix_model><<<block_per_grid, thread_per_block>>>(zone, param);

  constexpr int block_dim = 128;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                       // fc
                     + n_computation_per_block * (n_var + 1)) * sizeof(real) // pv[n_var]+jacobian
                    + n_computation_per_block * 3 * sizeof(real)             // metric[3]
                    + n_computation_per_block * sizeof(real);                // entropy fix delta

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void compute_entropy_fix_delta(DZone *zone, DParameter *param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  const auto &bv{zone->bv};
  const auto &metric{zone->metric};

  const real U = abs(
    bv(i, j, k, 1) * metric(i, j, k, 0) + bv(i, j, k, 2) * metric(i, j, k, 1) + bv(i, j, k, 3) * metric(i, j, k, 2));
  const real V = abs(
    bv(i, j, k, 1) * metric(i, j, k, 3) + bv(i, j, k, 2) * metric(i, j, k, 4) + bv(i, j, k, 3) * metric(i, j, k, 5));
  const real W = abs(
    bv(i, j, k, 1) * metric(i, j, k, 6) + bv(i, j, k, 2) * metric(i, j, k, 7) + bv(i, j, k, 3) * metric(i, j, k, 8));

  const real kx = norm3d(metric(i, j, k, 0), metric(i, j, k, 1), metric(i, j, k, 2));
  const real ky = norm3d(metric(i, j, k, 3), metric(i, j, k, 4), metric(i, j, k, 5));
  const real kz = norm3d(metric(i, j, k, 6), metric(i, j, k, 7), metric(i, j, k, 8));

  real acoustic_speed{0};
  if constexpr (mix_model != MixtureModel::Air) {
    acoustic_speed = zone->acoustic_speed(i, j, k);
  } else {
    acoustic_speed = sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0));
  }
  if (param->dim == 2) {
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + acoustic_speed * 0.5 * (kx + ky));
  } else {
    // 3D
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + W + acoustic_speed * (kx + ky + kz) / 3.0);
  }
}

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = static_cast<int>((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *entropy_fix_delta = &jac[n_point];
  real *fc = &entropy_fix_delta[n_point];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 0; l < 3; ++l) {
    metric[i_shared * 3 + l] = zone->metric(idx[0], idx[1], idx[2], direction * 3 + l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
  entropy_fix_delta[i_shared] = zone->entropy_fix_delta(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < 3; ++l) {
        metric[ig_shared * 3 + l] = zone->metric(g_idx[0], g_idx[1], g_idx[2], direction * 3 + l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    entropy_fix_delta[tid + ngg] = zone->entropy_fix_delta(idx[0] + labels[0], idx[1] + labels[1], idx[2] + labels[2]);
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < 3; ++l) {
        metric[ig_shared * 3 + l] = zone->metric(g_idx[0], g_idx[1], g_idx[2], direction * 3 + l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  riemannSolver_Roe<mix_model>(zone, pv, tid, param, fc, metric, jac, entropy_fix_delta);
  __syncthreads();


  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

// template instantiation
template void compute_convective_term_pv<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void compute_convective_term_pv<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void Roe_compute_inviscid_flux<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void Roe_compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);
}

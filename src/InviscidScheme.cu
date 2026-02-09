#include "InviscidScheme.cuh"
#include "Constants.h"
#include "Mesh.h"
#include "Parameter.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Reconstruction.cuh"
#include "RiemannSolver.cuh"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model> void compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter) {
  switch (parameter.get_int("inviscid_type")) {
    case 0: // Compute the term with primitive reconstruction methods. (MUSCL/NND/1stOrder + LF/AUSM+/HLLC)
      compute_convective_term_pv<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 3: // Compute the term with WENO-Z-5
      compute_convective_term_weno<mix_model>(block, zone, param, n_var);
      break;
    case 2:  // Roe scheme
    default: // Roe scheme
      Roe_compute_inviscid_flux<mix_model>(block, zone, param, n_var, parameter);
      break;
  }
}

template<MixtureModel mix_model> void compute_convective_term_pv(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter) {
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

template<MixtureModel mix_model> __global__ void compute_convective_term_pv_1D(DZone *zone, int direction,
  int max_extent, DParameter *param) {
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

template<MixtureModel mix_model> __device__ void reconstruction(real *pv, real *pv_l, real *pv_r, const int idx_shared,
  DParameter *param) {
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

template<MixtureModel mix_model> void Roe_compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter) {
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

template<MixtureModel mix_model> __global__ void compute_entropy_fix_delta(DZone *zone, DParameter *param) {
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

template<MixtureModel mix_model> __global__ void Roe_compute_inviscid_flux_1D(DZone *zone, int direction,
  int max_extent, DParameter *param) {
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

template<MixtureModel mix_model> __device__ void riemannSolver_ausmPlus(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared) {
  const auto metric_l = &metric[i_shared * 3], metric_r = &metric[(i_shared + 1) * 3];
  const auto jac_l = jac[i_shared], jac_r = jac[i_shared + 1];
  const real k1 = 0.5 * (jac_l * metric_l[0] + jac_r * metric_r[0]);
  const real k2 = 0.5 * (jac_l * metric_l[1] + jac_r * metric_r[1]);
  const real k3 = 0.5 * (jac_l * metric_l[2] + jac_r * metric_r[2]);
  const real grad_k_div_jac = sqrt(k1 * k1 + k2 * k2 + k3 * k3);

  const real ul = (k1 * pv_l[1] + k2 * pv_l[2] + k3 * pv_l[3]) / grad_k_div_jac;
  const real ur = (k1 * pv_r[1] + k2 * pv_r[2] + k3 * pv_r[3]) / grad_k_div_jac;

  const real pl = pv_l[4], pr = pv_r[4], rho_l = pv_l[0], rho_r = pv_r[0];
  real gam_l{gamma_air}, gam_r{gamma_air};
  const int n_var = param->n_var;
  auto n_reconstruct{n_var};
  if constexpr (mix_model != MixtureModel::Air) {
    gam_l = pv_l[n_reconstruct + 1];
    gam_r = pv_r[n_reconstruct + 1];
  }
  const real c = 0.5 * (sqrt(gam_l * pl / rho_l) + sqrt(gam_r * pr / rho_r));
  const real mach_l = ul / c, mach_r = ur / c;
  real mlp{0}, mrn{0}, plp{0}, prn{0}; // m for M, l/r for L/R, p/n for +/-. mlp=M_L^+
  constexpr static real alpha{3 / 16.0};

  if (abs(mach_l) > 1) {
    mlp = 0.5 * (mach_l + abs(mach_l));
    plp = mlp / mach_l;
  } else {
    const real ml_plus1_squared_div4 = (mach_l + 1) * (mach_l + 1) * 0.25;
    const real ml_squared_minus_1_squared = (mach_l * mach_l - 1) * (mach_l * mach_l - 1);
    mlp = ml_plus1_squared_div4 + 0.125 * ml_squared_minus_1_squared;
    plp = ml_plus1_squared_div4 * (2 - mach_l) + alpha * mach_l * ml_squared_minus_1_squared;
  }
  if (abs(mach_r) > 1) {
    mrn = 0.5 * (mach_r - abs(mach_r));
    prn = mrn / mach_r;
  } else {
    const real mr_minus1_squared_div4 = (mach_r - 1) * (mach_r - 1) * 0.25;
    const real mr_squared_minus_1_squared = (mach_r * mach_r - 1) * (mach_r * mach_r - 1);
    mrn = -mr_minus1_squared_div4 - 0.125 * mr_squared_minus_1_squared;
    prn = mr_minus1_squared_div4 * (2 + mach_r) - alpha * mach_r * mr_squared_minus_1_squared;
  }

  const real p_coeff = plp * pl + prn * pr;

  const real m_half = mlp + mrn;
  const real mach_pos = 0.5 * (m_half + abs(m_half));
  const real mach_neg = 0.5 * (m_half - abs(m_half));
  const real mass_flux_half = c * (rho_l * mach_pos + rho_r * mach_neg);
  const real coeff = mass_flux_half * grad_k_div_jac;

  const auto fci = &fc[tid * n_var];
  if (mass_flux_half >= 0) {
    fci[0] = coeff;
    fci[1] = coeff * pv_l[1] + p_coeff * k1;
    fci[2] = coeff * pv_l[2] + p_coeff * k2;
    fci[3] = coeff * pv_l[3] + p_coeff * k3;
    fci[4] = coeff * (pv_l[n_reconstruct] + pv_l[4]) / pv_l[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_l[l];
    }
  } else {
    fci[0] = coeff;
    fci[1] = coeff * pv_r[1] + p_coeff * k1;
    fci[2] = coeff * pv_r[2] + p_coeff * k2;
    fci[3] = coeff * pv_r[3] + p_coeff * k3;
    fci[4] = coeff * (pv_r[n_reconstruct] + pv_r[4]) / pv_r[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_r[l];
    }
  }
}

template<MixtureModel mix_model> __device__ void riemannSolver_hllc(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared) {
  const int n_var = param->n_var;
  int n_reconstruct{n_var};

  // Compute the Roe averaged variables.
  const real rl_c{std::sqrt(pv_l[0]) / (std::sqrt(pv_l[0]) + std::sqrt(pv_r[0]))},
      rr_c{std::sqrt(pv_r[0]) / (std::sqrt(pv_l[0]) + std::sqrt(pv_r[0]))};
  const real u_tilde{pv_l[1] * rl_c + pv_r[1] * rr_c};
  const real v_tilde{pv_l[2] * rl_c + pv_r[2] * rr_c};
  const real w_tilde{pv_l[3] * rl_c + pv_r[3] * rr_c};

  real gamma{gamma_air};
  real c_tilde;

  if constexpr (mix_model == MixtureModel::Air) {
    const real hl{(pv_l[n_reconstruct] + pv_l[4]) / pv_l[0]};
    const real hr{(pv_r[n_reconstruct] + pv_r[4]) / pv_r[0]};
    const real h_tilde{hl * rl_c + hr * rr_c};
    const real ek_tilde{0.5 * (u_tilde * u_tilde + v_tilde * v_tilde + w_tilde * w_tilde)};
    c_tilde = std::sqrt((gamma - 1) * (h_tilde - ek_tilde));
  } else {
    real svm[MAX_SPEC_NUMBER + 4] = {};
    for (int l = 0; l < param->n_var - 5; ++l) {
      svm[l] = rl_c * pv_l[l + 5] + rr_c * pv_r[l + 5];
    }

    real mw_inv{0};
    for (int l = 0; l < param->n_spec; ++l) {
      mw_inv += svm[l] * param->imw[l];
    }

    const real tl{pv_l[4] / pv_l[0]};
    const real tr{pv_r[4] / pv_r[0]};
    const real t{(rl_c * tl + rr_c * tr) / (R_u * mw_inv)};

    real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cv{0}, cp{0};
    for (int l = 0; l < param->n_spec; ++l) {
      cp += cp_i[l] * svm[l];
      cv += svm[l] * (cp_i[l] - param->gas_const[l]);
    }
    gamma = cp / cv;
    c_tilde = std::sqrt(gamma * R_u * mw_inv * t);
  }

  const real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  const real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  const real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);
  const real U_tilde_bar{(kx * u_tilde + ky * v_tilde + kz * w_tilde) / gradK};

  const auto fci = &fc[tid * n_var];
  const real jac_ave{0.5 * (jac[i_shared] + jac[i_shared + 1])};

  real gamma_l{gamma_air}, gamma_r{gamma_air};
  if constexpr (mix_model != MixtureModel::Air) {
    gamma_l = pv_l[n_reconstruct + 1];
    gamma_r = pv_r[n_reconstruct + 1];
  }
  const real Ul{kx * pv_l[1] + ky * pv_l[2] + kz * pv_l[3]};
  const real cl{std::sqrt(gamma_l * pv_l[4] / pv_l[0])};
  const real sl{min(Ul / gradK - cl, U_tilde_bar - c_tilde)};
  if (sl >= 0) {
    // The flow is supersonic from left to right, the flux is computed from the left value.
    const real rhoUk{pv_l[0] * Ul};
    fci[0] = jac_ave * rhoUk;
    fci[1] = jac_ave * (rhoUk * pv_l[1] + pv_l[4] * kx);
    fci[2] = jac_ave * (rhoUk * pv_l[2] + pv_l[4] * ky);
    fci[3] = jac_ave * (rhoUk * pv_l[3] + pv_l[4] * kz);
    fci[4] = jac_ave * ((pv_l[n_reconstruct] + pv_l[4]) * Ul);
    for (int l = 5; l < n_var; ++l) {
      fci[l] = jac_ave * rhoUk * pv_l[l];
    }
    return;
  }

  const real Ur{kx * pv_r[1] + ky * pv_r[2] + kz * pv_r[3]};
  const real cr{std::sqrt(gamma_r * pv_r[4] / pv_r[0])};
  const real sr{max(Ur / gradK + cr, U_tilde_bar + c_tilde)};
  if (sr < 0) {
    // The flow is supersonic from right to left, the flux is computed from the right value.
    const real rhoUk{pv_r[0] * Ur};
    fci[0] = jac_ave * rhoUk;
    fci[1] = jac_ave * (rhoUk * pv_r[1] + pv_r[4] * kx);
    fci[2] = jac_ave * (rhoUk * pv_r[2] + pv_r[4] * ky);
    fci[3] = jac_ave * (rhoUk * pv_r[3] + pv_r[4] * kz);
    fci[4] = jac_ave * ((pv_r[n_reconstruct] + pv_r[4]) * Ur);
    for (int l = 5; l < n_var; ++l) {
      fci[l] = jac_ave * rhoUk * pv_r[l];
    }
    return;
  }

  // Else, the current position is in star region; we need to identify the left and right star states.
  const real sm{
    ((pv_r[0] * Ur * (sr - Ur / gradK) - pv_l[0] * Ul * (sl - Ul / gradK)) / gradK + pv_l[4] - pv_r[4]) /
    (pv_r[0] * (sr - Ur / gradK) - pv_l[0] * (sl - Ul / gradK))
  };
  const real pm{pv_l[0] * (sl - Ul / gradK) * (sm - Ul / gradK) + pv_l[4]};
  if (sm >= 0) {
    // Left star region, F_{*L}
    const real pCoeff{1.0 / (sl - sm)};
    const real QlCoeff{jac_ave * pCoeff * sm * (sl * gradK - Ul) * pv_l[0]};
    fci[0] = QlCoeff;
    const real dP{(sl * pm - sm * pv_l[4]) * pCoeff * jac_ave};
    fci[1] = QlCoeff * pv_l[1] + dP * kx;
    fci[2] = QlCoeff * pv_l[2] + dP * ky;
    fci[3] = QlCoeff * pv_l[3] + dP * kz;
    fci[4] = QlCoeff * pv_l[n_reconstruct] / pv_l[0] + pCoeff * jac_ave * (sl * pm * sm * gradK - sm * pv_l[4] * Ul);
    for (int l = 5; l < n_var; ++l) {
      fci[l] = QlCoeff * pv_l[l];
    }
  } else {
    // Right star region, F_{*R}
    const real pCoeff{1.0 / (sr - sm)};
    const real QrCoeff{jac_ave * pCoeff * sm * (sr * gradK - Ur) * pv_r[0]};
    fci[0] = QrCoeff;
    const real dP{(sr * pm - sm * pv_r[4]) * pCoeff * jac_ave};
    fci[1] = QrCoeff * pv_r[1] + dP * kx;
    fci[2] = QrCoeff * pv_r[2] + dP * ky;
    fci[3] = QrCoeff * pv_r[3] + dP * kz;
    fci[4] = QrCoeff * pv_r[n_reconstruct] / pv_r[0] + pCoeff * jac_ave * (sr * pm * sm * gradK - sm * pv_r[4] * Ur);
    for (int l = 5; l < n_var; ++l) {
      fci[l] = QrCoeff * pv_r[l];
    }
  }
}

template<MixtureModel mixtureModel> __device__ void compute_half_sum_left_right_flux(const real *pv_l, const real *pv_r,
  DParameter *param, const real *jac, const real *metric, int i_shared, real *fc) {
  real JacKx = jac[i_shared] * metric[i_shared * 3];
  real JacKy = jac[i_shared] * metric[i_shared * 3 + 1];
  real JacKz = jac[i_shared] * metric[i_shared * 3 + 2];
  real Uk = pv_l[1] * JacKx + pv_l[2] * JacKy + pv_l[3] * JacKz;

  int n_reconstruct{param->n_var};
  real coeff = Uk * pv_l[0];
  fc[0] = 0.5 * coeff;
  fc[1] = 0.5 * (coeff * pv_l[1] + pv_l[4] * JacKx);
  fc[2] = 0.5 * (coeff * pv_l[2] + pv_l[4] * JacKy);
  fc[3] = 0.5 * (coeff * pv_l[3] + pv_l[4] * JacKz);
  fc[4] = 0.5 * Uk * (pv_l[4] + pv_l[n_reconstruct]);
  for (int l = 5; l < param->n_var; ++l) {
    fc[l] = 0.5 * coeff * pv_l[l];
  }

  JacKx = jac[i_shared + 1] * metric[(i_shared + 1) * 3];
  JacKy = jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1];
  JacKz = jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2];
  Uk = pv_r[1] * JacKx + pv_r[2] * JacKy + pv_r[3] * JacKz;

  coeff = Uk * pv_r[0];
  fc[0] += 0.5 * coeff;
  fc[1] += 0.5 * (coeff * pv_r[1] + pv_r[4] * JacKx);
  fc[2] += 0.5 * (coeff * pv_r[2] + pv_r[4] * JacKy);
  fc[3] += 0.5 * (coeff * pv_r[3] + pv_r[4] * JacKz);
  fc[4] += 0.5 * Uk * (pv_r[4] + pv_r[n_reconstruct]);
  for (int l = 5; l < param->n_var; ++l) {
    fc[l] += 0.5 * coeff * pv_r[l];
  }
}

template<MixtureModel mix_model> __device__ void riemannSolver_Roe(DZone *zone, real *pv, int tid, DParameter *param,
  real *fc, real *metric, const real *jac, const real *entropy_fix_delta) {
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  const int i_shared = tid - 1 + zone->ngg;
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);

  // The entropy fix delta may not need shared memory, which may be replaced by shuffle instructions.
  int n_reconstruct{param->n_var};

  // Compute the left and right convective fluxes, which uses the reconstructed primitive variables, rather than the roe averaged ones.
  const auto fci = &fc[tid * param->n_var];
  compute_half_sum_left_right_flux<mix_model>(pv_l, pv_r, param, jac, metric, i_shared, fci);

  // Compute the Roe averaged variables.
  const real dl = std::sqrt(pv_l[0]), dr = std::sqrt(pv_r[0]);
  const real inv_denominator = 1.0 / (dl + dr);
  const real u = (dl * pv_l[1] + dr * pv_r[1]) * inv_denominator;
  const real v = (dl * pv_l[2] + dr * pv_r[2]) * inv_denominator;
  const real w = (dl * pv_l[3] + dr * pv_r[3]) * inv_denominator;
  const real ek = 0.5 * (u * u + v * v + w * w);
  const real hl = (pv_l[n_reconstruct] + pv_l[4]) / pv_l[0];
  const real hr = (pv_r[n_reconstruct] + pv_r[4]) / pv_r[0];
  const real h = (dl * hl + dr * hr) * inv_denominator;

  real gamma{gamma_air};
  real c = std::sqrt((gamma - 1) * (h - ek));
  real mw{mw_air};
  real svm[MAX_SPEC_NUMBER + 4] = {};
  for (int l = 0; l < param->n_var - 5; ++l) {
    svm[l] = (dl * pv_l[l + 5] + dr * pv_r[l + 5]) * inv_denominator;
  }

  real h_i[MAX_SPEC_NUMBER];
  if constexpr (mix_model != MixtureModel::Air) {
    real mw_inv{0};
    for (int l = 0; l < param->n_spec; ++l) {
      mw_inv += svm[l] * param->imw[l];
    }

    const real tl{pv_l[4] / pv_l[0]};
    const real tr{pv_r[4] / pv_r[0]};
    const real t{(dl * tl + dr * tr) * inv_denominator / (R_u * mw_inv)};

    real cp_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cv{0}, cp{0};
    for (int l = 0; l < param->n_spec; ++l) {
      cp += cp_i[l] * svm[l];
      cv += svm[l] * (cp_i[l] - param->gas_const[l]);
    }
    gamma = cp / cv;
    c = std::sqrt(gamma * R_u * mw_inv * t);
    mw = 1.0 / mw_inv;
  }

  // Compute the characteristics
  real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);
  real Uk = kx * u + ky * v + kz * w;

  real characteristic[3]{Uk - gradK * c, Uk, Uk + gradK * c};
  // entropy fix
  const real entropy_fix_delta_ave{0.5 * (entropy_fix_delta[i_shared] + entropy_fix_delta[i_shared + 1])};
  for (auto &cc: characteristic) {
    cc = std::abs(cc);
    if (cc < entropy_fix_delta_ave) {
      cc = 0.5 * (cc * cc / entropy_fix_delta_ave + entropy_fix_delta_ave);
    }
  }

  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  Uk /= gradK;

  // compute dQ
  const real jac_ave{0.5 * (jac[i_shared] + jac[i_shared + 1])};
  real dq[5 + MAX_SPEC_NUMBER + 4] = {};
  dq[0] = jac_ave * (pv_r[0] - pv_l[0]);
  for (int l = 1; l < param->n_var; ++l) {
    dq[l] = jac_ave * (pv_r[0] * pv_r[l] - pv_l[0] * pv_l[l]);
  }
  dq[4] = jac_ave * (pv_r[n_reconstruct] - pv_l[n_reconstruct]);

  real c1 = (gamma - 1) * (ek * dq[0] - u * dq[1] - v * dq[2] - w * dq[3] + dq[4]) / (c * c);
  real c2 = (kx * dq[1] + ky * dq[2] + kz * dq[3] - Uk * dq[0]) / c;
  for (int l = 0; l < param->n_spec; ++l) {
    c1 += (mw * param->imw[l] - h_i[l] * (gamma - 1) / (c * c)) * dq[l + 5];
  }
  real c3 = dq[0] - c1;

  // compute L*dQ
  real LDq[5 + MAX_SPEC_NUMBER + 4] = {};
  LDq[0] = 0.5 * (c1 - c2);
  LDq[1] = kx * c3 - ((kz * v - ky * w) * dq[0] - kz * dq[2] + ky * dq[3]) / c;
  LDq[2] = ky * c3 - ((kx * w - kz * u) * dq[0] - kx * dq[3] + kz * dq[1]) / c;
  LDq[3] = kz * c3 - ((ky * u - kx * v) * dq[0] - ky * dq[1] + kx * dq[2]) / c;
  LDq[4] = 0.5 * (c1 + c2);
  for (int l = 0; l < param->n_scalar_transported; ++l) {
    LDq[l + 5] = dq[l + 5] - svm[l] * dq[0];
  }

  // To reduce memory usage, we use dq array to contain the b array to be computed
  const auto b = dq;
  b[0] = -characteristic[0] * LDq[0];
  for (int l = 1; l < param->n_var; ++l) {
    b[l] = -characteristic[1] * LDq[l];
  }
  b[4] = -characteristic[2] * LDq[4];

  const real c0 = kx * b[1] + ky * b[2] + kz * b[3];
  c1 = c0 + b[0] + b[4];
  c2 = c * (b[4] - b[0]);
  c3 = 0;
  for (int l = 0; l < param->n_spec; ++l)
    c3 += (h_i[l] - mw * param->imw[l] * c * c / (gamma - 1)) * b[l + 5];

  fci[0] += 0.5 * c1;
  fci[1] += 0.5 * (u * c1 + kx * c2 - c * (kz * b[2] - ky * b[3]));
  fci[2] += 0.5 * (v * c1 + ky * c2 - c * (kx * b[3] - kz * b[1]));
  fci[3] += 0.5 * (w * c1 + kz * c2 - c * (ky * b[1] - kx * b[2]));
  fci[4] += 0.5 *
  (h * c1 + Uk * c2 - c * c * c0 / (gamma - 1) + c * ((kz * v - ky * w) * b[1] + (kx * w - kz * u) * b[2] +
                                                      (ky * u - kx * v) * b[3]) + c3);
  for (int l = 0; l < param->n_var - 5; ++l)
    fci[5 + l] += 0.5 * (b[l + 5] + svm[l] * c1);
}

template<MixtureModel mix_model> __device__ void riemannSolver_laxFriedrich(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared) {
  printf("LF flux for mixture is not implemented yet. Please use AUSM+ or Roe instead.\n");
}

template<> __device__ void riemannSolver_laxFriedrich<MixtureModel::Air>(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared) {
  const int n_var = param->n_var;
  const int n_reconstruct{n_var};

  // The metrics are just the average of the two adjacent cells.
  const real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  const real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  const real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = sqrt(kx * kx + ky * ky + kz * kz);

  // compute the left and right contravariance velocity
  const real Ukl{pv_l[1] * kx + pv_l[2] * ky + pv_l[3] * kz};
  const real Ukr{pv_r[1] * kx + pv_r[2] * ky + pv_r[3] * kz};
  const real cl{sqrt(gamma_air * pv_l[4] / pv_l[0])};
  const real cr{sqrt(gamma_air * pv_r[4] / pv_r[0])};
  const real spectral_radius{max(abs(Ukl) + cl * gradK, abs(Ukr) + cr * gradK)};

  const auto fci = &fc[tid * n_var];
  const real half_jac_ave{0.5 * 0.5 * (jac[i_shared] + jac[i_shared + 1])};

  const real rhoUl{pv_l[0] * Ukl};
  const real rhoUr{pv_r[0] * Ukr};

  fci[0] = (rhoUl + rhoUr - spectral_radius * (pv_r[0] - pv_l[0])) * half_jac_ave;
  fci[1] = (rhoUl * pv_l[1] + rhoUr * pv_r[1] + kx * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[1] * pv_r[0] - pv_l[1] * pv_l[0])) * half_jac_ave;
  fci[2] = (rhoUl * pv_l[2] + rhoUr * pv_r[2] + ky * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[2] * pv_r[0] - pv_l[2] * pv_l[0])) * half_jac_ave;
  fci[3] = (rhoUl * pv_l[3] + rhoUr * pv_r[3] + kz * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[3] * pv_r[0] - pv_l[3] * pv_l[0])) * half_jac_ave;
  fci[4] = ((pv_l[n_reconstruct] + pv_l[4]) * Ukl + (pv_r[n_reconstruct] + pv_r[4]) * Ukr -
            spectral_radius * (pv_r[n_reconstruct] - pv_l[n_reconstruct])) * half_jac_ave;
}

// template instantiation
template void compute_inviscid_flux<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);
}

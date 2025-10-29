#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Constants.h"
#include "Thermo.cuh"
#include "FiniteRateChem.cuh"

namespace cfd {
struct DZone;

template<MixtureModel mixture_model>
__global__ void compute_DQ_0(DZone *zone, const DParameter *param, real diag_factor = 0) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt_local = zone->dt_local(i, j, k);
  auto &dq = zone->dq;
  for (int l = 0; l < param->n_var; ++l) {
    dq(i, j, k, l) *= dt_local;
  }

  const auto &inviscid_spectral_radius = zone->inv_spectr_rad(i, j, k);
  real diag = 1 + diag_factor * dt_local + dt_local * (inviscid_spectral_radius[0] + inviscid_spectral_radius[1] +
                                                       inviscid_spectral_radius[2]);
  if (param->viscous_scheme > 0)
    diag += 2 * dt_local *
        (zone->visc_spectr_rad(i, j, k)[0] + zone->visc_spectr_rad(i, j, k)[1] + zone->visc_spectr_rad(i, j, k)[2]);
  const int n_spec{param->n_spec};
  const int n_reac{param->n_reac};
  if (n_reac > 0) {
    // Use point implicit method to treat the chemical source
    for (int l = 0; l < 5; ++l) {
      dq(i, j, k, l) /= diag;
    }
    // Point implicit
    switch (param->chemSrcMethod) {
      case 1: // EPI
        EPI_for_dq0(zone, diag, i, j, k, n_spec);
        break;
      case 2: // DA
        for (int l = 0; l < n_spec; ++l) {
          zone->dq(i, j, k, 5 + l) /= diag - dt_local * zone->chem_src_jac(i, j, k, l);
        }
        break;
      case 0: // explicit treat
      default:
        for (int l = 0; l < n_spec; ++l) {
          dq(i, j, k, l + 5) /= diag;
        }
        break;
    }
  } else {
    for (int l = 0; l < 5 + n_spec; ++l) {
      dq(i, j, k, l) /= diag;
    }
  }
}

template<MixtureModel mixture_model>
__device__ void
compute_jacobian_times_dq(const DParameter *param, DZone *zone, int i, int j, int k, int dir, real pm_spectral_radius,
  real *convJacTimesDq) {
  const auto &m = zone->metric;
  const auto xi_x{m(i, j, k, dir * 3)}, xi_y{m(i, j, k, dir * 3 + 1)}, xi_z{m(i, j, k, dir * 3 + 2)};

  const auto &pv = zone->bv;
  const real u = pv(i, j, k, 1), v = pv(i, j, k, 2), w = pv(i, j, k, 3);
  const real U = xi_x * u + xi_y * v + xi_z * w;
  const real lmd1 = U + pm_spectral_radius;
  const real e = 0.5 * (u * u + v * v + w * w);
  real gamma{gamma_air};
  real b3{0}, b4{0}, h{0};
  auto &dq = zone->dq;
  const auto &sv = zone->sv;

  if constexpr (mixture_model == MixtureModel::Air) {
    h = gamma / (gamma - 1) * pv(i, j, k, 4) / pv(i, j, k, 0) + e;
  } else {
    const auto &R = param->gas_const;
    real enthalpy[MAX_SPEC_NUMBER];
    const real t{pv(i, j, k, 5)};
    compute_enthalpy(t, enthalpy, param);
    gamma = zone->gamma(i, j, k);
    for (int l = 0; l < param->n_spec; ++l) {
      b3 += R[l] * t * dq(i, j, k, 5 + l);
      b4 += enthalpy[l] * dq(i, j, k, 5 + l);
      h += sv(i, j, k, l) * enthalpy[l];
    }
    b3 *= gamma;
    b4 *= gamma - 1;
    h += e;
  }
  const double b1 = xi_x * dq(i, j, k, 1) + xi_y * dq(i, j, k, 2) + xi_z * dq(i, j, k, 3) - U * dq(i, j, k, 0);
  const double b2 = (gamma - 1) * (e * dq(i, j, k, 0) - u * dq(i, j, k, 1) - v * dq(i, j, k, 2) - w * dq(i, j, k, 3) +
                                   dq(i, j, k, 4));

  convJacTimesDq[0] = b1 + lmd1 * dq(i, j, k, 0);
  convJacTimesDq[1] = u * b1 + xi_x * b2 + lmd1 * dq(i, j, k, 1) + xi_x * (b3 - b4);
  convJacTimesDq[2] = v * b1 + xi_y * b2 + lmd1 * dq(i, j, k, 2) + xi_y * (b3 - b4);
  convJacTimesDq[3] = w * b1 + xi_z * b2 + lmd1 * dq(i, j, k, 3) + xi_z * (b3 - b4);
  convJacTimesDq[4] = h * b1 + U * b2 + lmd1 * dq(i, j, k, 4) + U * (b3 - b4);

  for (int l = 0; l < param->n_scalar_transported; ++l) {
    convJacTimesDq[5 + l] = lmd1 * dq(i, j, k, 5 + l) + sv(i, j, k, l) * b1;
  }
}

template<MixtureModel mixture_model>
__global__ void DPLUR_inner_iteration(const DParameter *param, DZone *zone, real diag_factor = 0) {
  // This can be split into 3 kernels, such that the shared memory can be used.
  // E.g., i=2 needs ii=1 and ii=3, while i=4 needs ii=3 and ii=5, thus the ii=3 is recomputed.
  // If we use a kernel in i direction, with each thread computing an ii, for ii=-1~blockDim,
  // then all threads in the block except threadID=0 and blockDim can use the just computed convJacTimesDq.
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  constexpr int n_var_max = 5 + MAX_SPEC_NUMBER + 4; // 5+n_spec+n_turb(n_turb<=2)
  real convJacTimesDq[n_var_max], dq_total[n_var_max] = {};

  const int n_var{param->n_var};
  const auto &inviscid_spectral_radius = zone->inv_spectr_rad;
  const auto &viscous_spectral_radius = zone->visc_spectr_rad;
  int ii{i - 1}, jj{j - 1}, kk{k - 1};
  if (i > 0) {
    compute_jacobian_times_dq<mixture_model>(param, zone, ii, j, k, 0, inviscid_spectral_radius(ii, j, k)[0]
                                                                       + 2 * viscous_spectral_radius(ii, j, k)[0],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] += 0.5 * convJacTimesDq[l];
    }
  }
  if (j > 0) {
    compute_jacobian_times_dq<mixture_model>(param, zone, i, jj, k, 1, inviscid_spectral_radius(i, jj, k)[1]
                                                                       + 2 * viscous_spectral_radius(i, jj, k)[1],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] += 0.5 * convJacTimesDq[l];
    }
  }
  if (k > 0) {
    compute_jacobian_times_dq<mixture_model>(param, zone, i, j, kk, 2, inviscid_spectral_radius(i, j, kk)[2]
                                                                       + 2 * viscous_spectral_radius(i, j, kk)[2],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] += 0.5 * convJacTimesDq[l];
    }
  }

  if (i != extent[0] - 1) {
    ii = i + 1;
    compute_jacobian_times_dq<mixture_model>(param, zone, ii, j, k, 0, -inviscid_spectral_radius(ii, j, k)[0]
                                                                       - 2 * viscous_spectral_radius(ii, j, k)[0],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] -= 0.5 * convJacTimesDq[l];
    }
  }
  if (j != extent[1] - 1) {
    jj = j + 1;
    compute_jacobian_times_dq<mixture_model>(param, zone, i, jj, k, 1, -inviscid_spectral_radius(i, jj, k)[1]
                                                                       - 2 * viscous_spectral_radius(i, jj, k)[1],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] -= 0.5 * convJacTimesDq[l];
    }
  }
  if (k != extent[2] - 1) {
    kk = k + 1;
    compute_jacobian_times_dq<mixture_model>(param, zone, i, j, kk, 2, -inviscid_spectral_radius(i, j, kk)[2]
                                                                       - 2 * viscous_spectral_radius(i, j, kk)[2],
                                             convJacTimesDq);
    for (int l = 0; l < n_var; ++l) {
      dq_total[l] -= 0.5 * convJacTimesDq[l];
    }
  }

  const real dt_local = zone->dt_local(i, j, k);
  const auto &spec_rad = inviscid_spectral_radius(i, j, k);
  real diag = 1 + diag_factor * dt_local + dt_local * (spec_rad[0] + spec_rad[1] + spec_rad[2])
              + 2 * dt_local * (zone->visc_spectr_rad(i, j, k)[0] + zone->visc_spectr_rad(i, j, k)[1] +
                                zone->visc_spectr_rad(i, j, k)[2]);
  auto &dqk = zone->dqk;
  const auto &dq0 = zone->dq0;
  const int n_spec{param->n_spec}, n_reac{param->n_reac};
  if (n_reac > 0) {
    // Use point implicit method to dispose chemical source
    #pragma unroll
    for (int l = 0; l < 5; ++l) {
      dqk(i, j, k, l) = dq0(i, j, k, l) + dt_local * dq_total[l] / diag;
    }
    // Point implicit
    switch (param->chemSrcMethod) {
      case 1: // EPI
        EPI_for_dqk(zone, diag, i, j, k, dq_total, n_spec);
        break;
      case 2: // DA
        for (int l = 0; l < n_spec; ++l) {
          dqk(i, j, k, 5 + l) =
              dq0(i, j, k, 5 + l) + dt_local * dq_total[5 + l] / (diag - dt_local * zone->chem_src_jac(i, j, k, l));
        }
        break;
      case 0: // explicit treat
      default:
        for (int l = 0; l < n_spec; ++l) {
          dqk(i, j, k, 5 + l) = dq0(i, j, k, 5 + l) + dt_local * dq_total[5 + l] / diag;
        }
        break;
    }
  } else {
    for (int l = 0; l < 5 + n_spec; ++l) {
      dqk(i, j, k, l) = dq0(i, j, k, l) + dt_local * dq_total[l] / diag;
    }
  }
}

__global__ void convert_dq_back_to_dqDt(DZone *zone, const DParameter *param);

struct DBoundCond;

void set_wall_dq_to_0(const Block &block, const DParameter *param, DZone *zone, const DBoundCond &bound_cond,
  bool ngg_extended);

template<MixtureModel mixture_model>
void DPLUR(const Block &block, const DParameter *param, DZone *d_ptr, DZone *h_ptr, const Parameter &parameter,
  DBoundCond &bound_cond, real diag_factor = 0) {
  const int extent[3]{block.mx, block.my, block.mz};
  const int dim{extent[2] == 1 ? 2 : 3};
  dim3 tpb{8, 8, 4};
  if (dim == 2) {
    tpb = {16, 16, 1};
  }
  const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};

  // DQ(0)=dt*DQ/(1+dt*DRho+dt*dS/dQ)
  compute_DQ_0<mixture_model><<<bpg, tpb>>>(d_ptr, param, diag_factor);
  // Take care of all such treatments where n_var is used to decide the memory size,
  // for when the flamelet model is used, the data structure should be modified to make the useful data contiguous.
  const auto mem_sz = h_ptr->dq.size() * parameter.get_int("n_var") * sizeof(real);
  cudaMemcpy(h_ptr->dq0.data(), h_ptr->dq.data(), mem_sz, cudaMemcpyDeviceToDevice);

  bool ngg_extended{false};
  if (parameter.get_int("viscous_order") == 2) {
    ngg_extended = true;
  }

  for (int iter = 0; iter < parameter.get_int("DPLUR_inner_step"); ++iter) {
    set_wall_dq_to_0(block, param, d_ptr, bound_cond, ngg_extended);

    DPLUR_inner_iteration<mixture_model><<<bpg, tpb>>>(param, d_ptr, diag_factor);
    // Theoretically, there should be a data communication here to exchange dq among processes.
    cudaMemcpy(h_ptr->dq.data(), h_ptr->dqk.data(), mem_sz, cudaMemcpyDeviceToDevice);
  }

  convert_dq_back_to_dqDt<<<bpg, tpb>>>(d_ptr, param);
}
}

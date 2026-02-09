// Contains some operations on different variables.
// E.g., total energy is computed from V and h; conservative variables are computed from basic variables
#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Transport.cuh"
#include "Constants.h"

namespace cfd {
__device__ void compute_temperature_and_pressure(int i, int j, int k, const DParameter *param, DZone *zone,
  real total_energy);

template<MixtureModel mixture_model> __device__ __forceinline__ void compute_total_energy(int i, int j, int k,
  DZone *zone, const DParameter *param) {
  auto &bv = zone->bv;

  const real V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  real total_energy = 0.5 * V2;
  if constexpr (mixture_model != MixtureModel::Air) {
    real enthalpy[MAX_SPEC_NUMBER];
    compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
    // Add species enthalpy together up to kinetic energy to get total enthalpy
    for (auto l = 0; l < param->n_spec; l++) {
      // h = \Sum_{i=1}^{n_spec} h_i * Y_i
      total_energy += enthalpy[l] * zone->sv(i, j, k, l);
    }
    total_energy *= bv(i, j, k, 0); // \rho * h
    total_energy -= bv(i, j, k, 4); // (\rho e =\rho h - p)
  } else {
    total_energy *= bv(i, j, k, 0); // \rho * h
    total_energy += bv(i, j, k, 4) / (gamma_air - 1);
  }
  zone->cv(i, j, k, 4) = total_energy;
}

template<MixtureModel mix_model> __global__ void compute_cv_from_bv(DZone *zone, DParameter *param) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const auto &bv = zone->bv;
  auto &cv = zone->cv;
  const real rho = bv(i, j, k, 0);
  const real u = bv(i, j, k, 1);
  const real v = bv(i, j, k, 2);
  const real w = bv(i, j, k, 3);

  cv(i, j, k, 0) = rho;
  cv(i, j, k, 1) = rho * u;
  cv(i, j, k, 2) = rho * v;
  cv(i, j, k, 3) = rho * w;
  const auto &sv = zone->sv;
  const int n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) {
    cv(i, j, k, 5 + l) = rho * sv(i, j, k, l);
  }

  compute_total_energy<mix_model>(i, j, k, zone, param);
}

template<MixtureModel mix_model> __device__ __forceinline__ void compute_cv_from_bv_1_point(DZone *zone,
  const DParameter *param, int i, int j, int k) {
  const auto &bv = zone->bv;
  auto &cv = zone->cv;
  const real rho = bv(i, j, k, 0);

  cv(i, j, k, 0) = rho;
  cv(i, j, k, 1) = rho * bv(i, j, k, 1);
  cv(i, j, k, 2) = rho * bv(i, j, k, 2);
  cv(i, j, k, 3) = rho * bv(i, j, k, 3);
  // It seems we don't need an if here, if there are no other scalars, n_scalar=0; else, n_scalar=n_spec+n_turb
  const auto &sv = zone->sv;
  const int n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) {
    cv(i, j, k, 5 + l) = rho * sv(i, j, k, l);
  }

  compute_total_energy<mix_model>(i, j, k, zone, param);
}

template<MixtureModel mix_model> __global__ void update_physical_properties(DZone * __restrict__ zone,
  DParameter * __restrict__ param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz}, ngg{zone->ngg};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real temperature{zone->bv(i, j, k, 5)};
  const auto V = norm3d(zone->bv(i, j, k, 1), zone->bv(i, j, k, 2), zone->bv(i, j, k, 3));
  if constexpr (mix_model != MixtureModel::Air) {
    const int n_spec{param->n_spec};
    auto &yk = zone->sv;
    real imw{0};
    real temp{0}; // First used as cp_tot
    real cp[MAX_SPEC_NUMBER];
    compute_cp(temperature, cp, param);
    for (auto l = 0; l < n_spec; ++l) {
      const auto y = yk(i, j, k, l);
      imw = fma(y, param->imw[l], imw);
      temp = fma(y, cp[l], temp);
    }
    const real R = R_u * imw;            // R = R_u / mw
    temp = temp / (temp - R);            // temp is specific heat ratio (gamma) now, gamma = cp / cv = cp / (cp - R)
    zone->gamma(i, j, k) = temp;         // gamma = cp / cv
    temp = sqrt(temp * R * temperature); // temp is acoustic speed now
    zone->acoustic_speed(i, j, k) = temp;
    zone->mach(i, j, k) = V / temp;
    compute_transport_property(i, j, k, temperature, 1 / imw, cp, param, zone);
  } else {
    constexpr real c_temp{gamma_air * R_u / mw_air};
    zone->mul(i, j, k) = Sutherland(temperature);
    zone->mach(i, j, k) = V / std::sqrt(c_temp * temperature);
  }
}

template<MixtureModel mixture_model> __device__ __forceinline__ real compute_total_energy_1_point(int i, int j, int k,
  DZone *zone, DParameter *param) {
  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const real V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  real total_energy = 0.5 * V2;
  if constexpr (mixture_model != MixtureModel::Air) {
    real enthalpy[MAX_SPEC_NUMBER];
    compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
    // Add species enthalpy together up to kinetic energy to get total enthalpy
    for (auto l = 0; l < param->n_spec; l++) {
      // h = \Sum_{i=1}^{n_spec} h_i * Y_i
      total_energy += enthalpy[l] * sv(i, j, k, l);
    }
    total_energy *= bv(i, j, k, 0); // \rho * h
    total_energy -= bv(i, j, k, 4); // (\rho e =\rho h - p)
  } else {
    total_energy *= bv(i, j, k, 0); // \rho * u_i * u_i * 0.5
    total_energy += bv(i, j, k, 4) / (gamma_air - 1);
  }
  return total_energy;
}

template<MixtureModel mix_model> __global__ void update_cv_and_bv(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;

  const real dt_div_jac = zone->dt_local(i, j, k) / zone->jac(i, j, k);
  for (int l = 0; l < param->n_var; ++l) {
    cv(i, j, k, l) += dt_div_jac * zone->dq(i, j, k, l);
  }
  if (extent[2] == 1) {
    cv(i, j, k, 3) = 0;
  }

  auto &bv = zone->bv;

  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;

  auto &sv = zone->sv;
  real sum_part_den{0};
  for (int l = 0; l < param->n_spec; ++l) {
    if (cv(i, j, k, 5 + l) < 0) {
      // If the species mass fraction is negative, we set it to zero.
      cv(i, j, k, 5 + l) = 0;
    } else
      sum_part_den += cv(i, j, k, l + 5);
  }
  sum_part_den = 1.0 / sum_part_den;
  const auto denComDivDenReal = cv(i, j, k, 0) * sum_part_den;
  for (int l = 0; l < param->n_spec; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * sum_part_den;
    cv(i, j, k, 5 + l) *= denComDivDenReal;
  }
  // For multiple species or RANS methods, there will be scalars to be computed
  for (int l = param->n_spec; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
  }

  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
  } else {
    // Air
    const real V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
    //V^2
    bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * V2);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }
}

__global__ void compute_shock_sensor(DZone *zone, const DParameter *param);

__global__ void eliminate_k_gradient(DZone *zone, const DParameter *param);
}

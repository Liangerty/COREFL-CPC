#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "Field.h"
#include "Constants.h"
#include "TurbMethod.hpp"
#include "FieldOperation.cuh"

namespace cfd {
__global__ void store_last_step(DZone *zone);

template<MixtureModel mixture>
__global__ void local_time_step(DZone *zone, DParameter *param);

__global__ void local_time_step_without_reaction(DZone *zone, DParameter *param);

__global__ void compute_square_of_dbv(DZone *zone);

real global_time_step(const Mesh &mesh, const Parameter &parameter, const std::vector<Field> &field);

__global__ void min_of_arr(real *arr, int size);

__global__ void update_physical_time(DParameter *param, real t);

template<MixtureModel mixture>
__global__ void limit_flow(DZone *zone, DParameter *param);
}

template<MixtureModel mixture>
__global__ void cfd::local_time_step(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &metric{zone->metric};
  const auto &bv = zone->bv;
  const int dim{zone->mz == 1 ? 2 : 3};

  const real grad_xi =   norm3d(metric(i, j, k, 0), metric(i, j, k, 1), metric(i, j, k, 2));
  const real grad_eta =  norm3d(metric(i, j, k, 3), metric(i, j, k, 4), metric(i, j, k, 5));
  const real grad_zeta = norm3d(metric(i, j, k, 6), metric(i, j, k, 7), metric(i, j, k, 8));

  const real u{bv(i, j, k, 1)}, v{bv(i, j, k, 2)}, w{bv(i, j, k, 3)};
  const real U = u * metric(i, j, k, 0) + v * metric(i, j, k, 1) + w *metric(i, j, k, 2);
  const real V = u * metric(i, j, k, 3) + v * metric(i, j, k, 4) + w * metric(i, j, k, 5);
  const real W = u * metric(i, j, k, 6) + v * metric(i, j, k, 7) + w * metric(i, j, k, 8);

  real acoustic_speed{0};
  if constexpr (mixture == MixtureModel::Air) {
    acoustic_speed = std::sqrt(gamma_air * R_u * bv(i, j, k, 5) / mw_air);
  } else {
    acoustic_speed = zone->acoustic_speed(i, j, k);
  }
  auto &inviscid_spectral_radius = zone->inv_spectr_rad(i, j, k);
  inviscid_spectral_radius[0] = std::abs(U) + acoustic_speed * grad_xi;
  inviscid_spectral_radius[1] = std::abs(V) + acoustic_speed * grad_eta;
  real max_spectral_radius = max(inviscid_spectral_radius[0], inviscid_spectral_radius[1]);
  inviscid_spectral_radius[2] = 0;
  if (dim == 3) {
    inviscid_spectral_radius[2] = std::abs(W) + acoustic_speed * grad_zeta;
    max_spectral_radius = max(max_spectral_radius, inviscid_spectral_radius[2]);
  }
  // const real spectral_radius_inv =
  //     inviscid_spectral_radius[0] + inviscid_spectral_radius[1] + inviscid_spectral_radius[2];

  // Next, compute the viscous spectral radius
  real max_length{grad_xi};
  max_length = max(max_length, grad_eta);
  if (dim == 3)
    max_length = max(max_length, grad_zeta);

  real max_diffuse_vel{0.0};
  const real iRho = 1.0 / bv(i, j, k, 0);
  if constexpr (mixture == MixtureModel::Air) {
    max_diffuse_vel = max(gamma_air, 4.0 / 3.0) * iRho;
    max_diffuse_vel *= zone->mul(i, j, k) / param->Pr;
  } else {
    // real nu = zone->mul(i, j, k) * iRho;
    // real alp = zone->thermal_conductivity(i, j, k) * iRho * zone->gamma(i, j, k) / zone->cp(i, j, k);
    // real D[MAX_SPEC_NUMBER];
    max_diffuse_vel = zone->mul(i, j, k) * iRho;
    max_diffuse_vel = max(max_diffuse_vel,
                          zone->thermal_conductivity(i, j, k) * iRho * zone->gamma(i, j, k) / zone->cp(i, j, k));
    real max_rhoD{0};
    for (int l = 0; l < param->n_spec; ++l) {
      max_rhoD = max(max_rhoD, zone->rho_D(i, j, k, l));
      // D[l] = zone->rho_D(i, j, k, l) * iRho;
    }
    max_diffuse_vel = max(max_diffuse_vel, max_rhoD * iRho);
    // if (print) {
    //   printf("i=%d, nu=%e, alp=%e, D=(%e,%e,%e,%e,%e,%e,%e,%e,%e)\n", i, nu, alp, D[0], D[1], D[2], D[3], D[4],
    //          D[5], D[6], D[7], D[8]);
    // }
  }
  auto &vis_spec_rad = zone->visc_spectr_rad(i, j, k);
  vis_spec_rad[0] = grad_xi * grad_xi * max_diffuse_vel;
  vis_spec_rad[1] = grad_eta * grad_eta * max_diffuse_vel;
  if (dim == 3)
    vis_spec_rad[2] = grad_zeta * grad_zeta * max_diffuse_vel;

  max_spectral_radius = max(max_spectral_radius, max_length * max_length * max_diffuse_vel);
  real dt = param->cfl / max_spectral_radius;

  if constexpr (mixture != MixtureModel::Air) {
    if (param->n_reac > 0) {
      // We need to take the reactions timescale into account
      const real reaction_dt = param->cfl * zone->reaction_timeScale(i, j, k);
      // if (print) {
      //   printf("i=%d, dt=%e, reaction_dt=%e\n", i, dt, reaction_dt);
      // }
      if (reaction_dt > 1e-20 && reaction_dt < 1e-5)
        dt = min(dt, reaction_dt);
    }
  }

  zone->dt_local(i, j, k) = dt;

  // real gamma{gamma_air};
  // if constexpr (mixture != MixtureModel::Air) {
  //   gamma = zone->gamma(i, j, k);
  // }
  // const real coeff_1 = max(gamma, 4.0 / 3.0) / bv(i, j, k, 0);
  // real coeff_2 = zone->mul(i, j, k) / param->Pr;
  // auto &vis_spec_rad = zone->visc_spectr_rad(i, j, k);
  // vis_spec_rad[0] = grad_xi * grad_xi * coeff_1 * coeff_2;
  // vis_spec_rad[1] = grad_eta * grad_eta * coeff_1 * coeff_2;
  // // real spectral_radius_viscous = grad_xi * grad_xi + grad_eta * grad_eta;
  // max_spectral_radius = max(max_spectral_radius, vis_spec_rad[0]);
  // max_spectral_radius = max(max_spectral_radius, vis_spec_rad[1]);
  // if (dim == 3) {
  //   // spectral_radius_viscous += grad_zeta * grad_zeta;
  //   vis_spec_rad[2] = grad_zeta * grad_zeta * coeff_1 * coeff_2;
  //   max_spectral_radius = max(max_spectral_radius, vis_spec_rad[2]);
  // }
  // spectral_radius_viscous *= coeff_1 * coeff_2;

  // zone->dt_local(i, j, k) = param->cfl / (spectral_radius_inv + spectral_radius_viscous);
  // zone->dt_local(i, j, k) = param->cfl / max_spectral_radius;
  //  zone->udv(i, j, k, 0) = zone->dt_local(i, j, k);
}

template<MixtureModel mixture>
__global__ void cfd::limit_flow(DZone *zone, DParameter *param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;

  // Record the computed values. First for flow variables and mass fractions
  constexpr int n_flow_var = 5;
  real var[n_flow_var];
  var[0] = bv(i, j, k, 0);
  var[1] = bv(i, j, k, 1);
  var[2] = bv(i, j, k, 2);
  var[3] = bv(i, j, k, 3);
  var[4] = bv(i, j, k, 4);

  // Find the unphysical values and limit them
  const auto ll = param->limit_flow.ll;
  const auto ul = param->limit_flow.ul;
  bool unphysical{false};
  for (int l = 0; l < n_flow_var; ++l) {
    if (isnan(var[l])) {
      unphysical = true;
      break;
    }
    if (var[l] < ll[l] || var[l] > ul[l]) {
      unphysical = true;
      break;
    }
  }

  __syncthreads();

  if (unphysical) {
    // printf("Unphysical values appear in process %d, block %d, i = %d, j = %d, k = %d.\n", param->myid, blk_id, i, j, k);

    real updated_var[n_flow_var/* + MAX_SPEC_NUMBER + 4*/] = {};
    int kn{0};
    // Compute the sum of all "good" points surrounding the "bad" point
    for (int ka = -1; ka < 2; ++ka) {
      const int k1{k + ka};
      if (k1 < 0 || k1 >= mz) continue;
      for (int ja = -1; ja < 2; ++ja) {
        const int j1{j + ja};
        if (j1 < 0 || j1 >= my) continue;
        for (int ia = -1; ia < 2; ++ia) {
          const int i1{i + ia};
          if (i1 < 0 || i1 >= mx)continue;

          if (isnan(bv(i1, j1, k1, 0)) || isnan(bv(i1, j1, k1, 1)) || isnan(bv(i1, j1, k1, 2)) ||
              isnan(bv(i1, j1, k1, 3)) || isnan(bv(i1, j1, k1, 4)) || bv(i1, j1, k1, 0) < ll[0] ||
              bv(i1, j1, k1, 1) < ll[1] || bv(i1, j1, k1, 2) < ll[2] || bv(i1, j1, k1, 3) < ll[3] ||
              bv(i1, j1, k1, 4) < ll[4] || bv(i1, j1, k1, 0) > ul[0] || bv(i1, j1, k1, 1) > ul[1] ||
              bv(i1, j1, k1, 2) > ul[2] || bv(i1, j1, k1, 3) > ul[3] || bv(i1, j1, k1, 4) > ul[4]) {
            continue;
          }

          updated_var[0] += bv(i1, j1, k1, 0);
          updated_var[1] += bv(i1, j1, k1, 1);
          updated_var[2] += bv(i1, j1, k1, 2);
          updated_var[3] += bv(i1, j1, k1, 3);
          updated_var[4] += bv(i1, j1, k1, 4);

          ++kn;
        }
      }
    }

    // Compute the average of the surrounding points
    if (kn > 0) {
      const real kn_inv{1.0 / kn};
      for (double &l: updated_var) {
        l *= kn_inv;
      }
    } else {
      // The surrounding points are all "bad."
      for (int l = 0; l < 5; ++l) {
        updated_var[l] = max(var[l], ll[l]);
        updated_var[l] = min(updated_var[l], ul[l]);
      }
    }

    // Assign averaged values for the bad point
    bv(i, j, k, 0) = updated_var[0];
    bv(i, j, k, 1) = updated_var[1];
    bv(i, j, k, 2) = updated_var[2];
    bv(i, j, k, 3) = updated_var[3];
    bv(i, j, k, 4) = updated_var[4];
    if constexpr (mixture == MixtureModel::Air) {
      bv(i, j, k, 5) = updated_var[4] * mw_air / (updated_var[0] * R_u);
    } else {
      real R = 0;
      for (int l = 0; l < param->n_spec; ++l) {
        R += zone->sv(i, j, k, l) * param->gas_const[l];
      }
      bv(i, j, k, 5) = updated_var[4] / (updated_var[0] * R);
    }

    compute_cv_from_bv_1_point<mixture>(zone, param, i, j, k);
  }
  __syncthreads();
}

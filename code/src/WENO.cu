#include "InviscidScheme.cuh"
#include "Constants.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Field.h"

namespace cfd {
template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_x(DZone *zone, DParameter *param) {
  const int i = static_cast<int>((blockDim.x - 1) * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->mx;
  if (i >= max_extent) return;

  const int tid = static_cast<int>(threadIdx.x);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  int n_active = min(static_cast<int>(blockDim.x), max_extent - static_cast<int>((blockDim.x - 1) * blockIdx.x) +
                                                   1); // n_active is the number of active threads in the block.
  int n_point = n_active + 2 * ngg - 1; // n_point is the number of points in the shared memory, used for the template.

  extern __shared__ real s[];
  real *cv = s;
  real *p = &cv[n_point * n_var];
  real *cc = &p[n_point];
  real *metric = &cc[n_point];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];

  int il0 = static_cast<int>((blockDim.x - 1) * blockIdx.x) - ngg;
  for (int il = il0 + tid; il <= il0 + n_point - 1; il += n_active) {
    int iSh = il - il0;                // iSh is the shared index
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_var + l] = zone->cv(il, j, k, l);
    }
    p[iSh] = zone->bv(il, j, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cc[iSh] = zone->acoustic_speed(il, j, k);
    else
      cc[iSh] = sqrt(gamma_air * R_air * zone->bv(il, j, k, 5));
    metric[iSh * 3] = zone->metric(il, j, k, 0);
    metric[iSh * 3 + 1] = zone->metric(il, j, k, 1);
    metric[iSh * 3 + 2] = zone->metric(il, j, k, 2);
    jac[iSh] = zone->jac(il, j, k);
    compute_flux(&cv[iSh * n_var], p[iSh], cc[iSh], param, &metric[iSh * 3], jac[iSh], &fp[iSh * n_var],
                 &fm[iSh * n_var]);
  }
  const int i_shared = tid - 1 + ngg;
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i + ii, j, k) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, p, param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_1(param->dim, n_var, cv, i_shared, jac, dt, &fc[tid * n_var], metric, cc, fp);
  }
  __syncthreads();

  if (tid > 0 && i >= zone->iMin && i <= zone->iMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_y(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->my;
  if (j >= max_extent) return;

  const auto tid = static_cast<int>(threadIdx.y);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.y), max_extent - static_cast<int>((blockDim.y - 1) * blockIdx.y) +
                                                          1); // n_active is the number of active threads in the block.
  const int n_point = n_active + 2 * ngg - 1;

  extern __shared__ real s[];
  real *cv = s;
  real *p = &cv[n_point * n_var];
  real *cc = &p[n_point];
  real *metric = &cc[n_point];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];

  const int jl0 = static_cast<int>((blockDim.y - 1) * blockIdx.y) - ngg;
  for (int jl = jl0 + tid; jl <= jl0 + n_point - 1; jl += n_active) {
    int iSh = jl - jl0;                // iSh is the shared index
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_var + l] = zone->cv(i, jl, k, l);
    }
    p[iSh] = zone->bv(i, jl, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cc[iSh] = zone->acoustic_speed(i, jl, k);
    else
      cc[iSh] = sqrt(gamma_air * R_air * zone->bv(i, jl, k, 5));
    metric[iSh * 3] = zone->metric(i, jl, k, 3);
    metric[iSh * 3 + 1] = zone->metric(i, jl, k, 4);
    metric[iSh * 3 + 2] = zone->metric(i, jl, k, 5);
    jac[iSh] = zone->jac(i, jl, k);
    compute_flux(&cv[iSh * n_var], p[iSh], cc[iSh], param, &metric[iSh * 3], jac[iSh], &fp[iSh * n_var],
                 &fm[iSh * n_var]);
  }
  const int i_shared = tid - 1 + ngg;
  __syncthreads();

  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j + ii, k) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, p, param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_1(param->dim, n_var, cv, i_shared, jac, dt, &fc[tid * n_var], metric, cc, fp);
  }
  __syncthreads();

  if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_weno_y1(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->my;
  if (j >= max_extent || i >= zone->mx) return;

  const auto tid = static_cast<int>(threadIdx.y), tx = static_cast<int>(threadIdx.x);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.y), max_extent - static_cast<int>(blockDim.y * blockIdx.y) + 1);
  const int n_point = n_active + 2 * ngg - 1;

  extern __shared__ real s[];
  constexpr int nyy = 32;
  auto fp = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
  auto fm = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

  const int jl0 = static_cast<int>(blockDim.y * blockIdx.y) - ngg;
  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  for (int jl = jl0 + tid; jl <= jl0 + n_point - 1; jl += n_active) {
    int iSh = jl - jl0; // iSh is the shared index
    real q[5 + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER], metric[3];
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      q[l] = cv(i, jl, k, l);
    }
    metric[0] = zone->metric(i, jl, k, 3);
    metric[1] = zone->metric(i, jl, k, 4);
    metric[2] = zone->metric(i, jl, k, 5);

    const real Uk{(q[1] * metric[0] + q[2] * metric[1] + q[3] * metric[2]) / q[0]};
    const real pk = bv(i, jl, k, 4);
    real cGradK = norm3d(metric[0], metric[1], metric[2]);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK *= zone->acoustic_speed(i, jl, k);
    else
      cGradK *= sqrt(gamma_air * R_air * bv(i, jl, k, 5));
    const real lambda0 = abs(Uk) + cGradK;
    const real jac = zone->jac(i, jl, k);

    fp[0][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[0]);
    fp[1][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[1] + pk * metric[0]);
    fp[2][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[2] + pk * metric[1]);
    fp[3][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[3] + pk * metric[2]);
    fp[4][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[4] + pk * Uk);

    fm[0][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[0]);
    fm[1][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[1] + pk * metric[0]);
    fm[2][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[2] + pk * metric[1]);
    fm[3][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[3] + pk * metric[2]);
    fm[4][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[4] + pk * Uk);

    for (int l = 5; l < n_var; ++l) {
      fp[l][iSh][tx] = 0.5 * jac * (Uk * q[l] + lambda0 * q[l]);
      fm[l][iSh][tx] = 0.5 * jac * (Uk * q[l] - lambda0 * q[l]);
    }
  }
  const int i_shared = tid - 1 + ngg;
  __syncthreads();

  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j + ii, k) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    //  const real eps_ref = 1e-6 * param->weno_eps_scale;
    constexpr real eps{1e-8};
    const real jac = zone->jac(i, j, k), jac_r = zone->jac(i, j + 1, k);
    const real eps_ref = eps * param->weno_eps_scale * 0.25 *
                         ((zone->metric(i, j, k, 3) * jac + zone->metric(i, j + 1, k, 3) * jac_r)
                          *
                          (zone->metric(i, j, k, 3) * jac + zone->metric(i, j + 1, k, 3) * jac_r) +
                          (zone->metric(i, j, k, 4) * jac + zone->metric(i, j + 1, k, 4) * jac_r)
                          *
                          (zone->metric(i, j, k, 4) * jac + zone->metric(i, j + 1, k, 4) * jac_r)
                          +
                          (zone->metric(i, j, k, 5) * jac + zone->metric(i, j + 1, k, 5) * jac_r)
                          *
                          (zone->metric(i, j, k, 6) * jac + zone->metric(i, j + 1, k, 5) * jac_r));
    real eps_scaled[3];
    eps_scaled[0] = eps_ref;
    eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
    eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

    auto &gc = zone->gFlux;
    for (int l = 0; l < n_var; ++l) {
      real eps_here{eps_scaled[0]};
      if (l == 1 || l == 2 || l == 3) {
        eps_here = eps_scaled[1];
      } else if (l == 4) {
        eps_here = eps_scaled[2];
      }

      if (param->inviscid_scheme == 71) {
        real vp[7], vm[7];
        vp[0] = fp[l][i_shared - 3][tx];
        vp[1] = fp[l][i_shared - 2][tx];
        vp[2] = fp[l][i_shared - 1][tx];
        vp[3] = fp[l][i_shared][tx];
        vp[4] = fp[l][i_shared + 1][tx];
        vp[5] = fp[l][i_shared + 2][tx];
        vp[6] = fp[l][i_shared + 3][tx];
        vm[0] = fm[l][i_shared - 2][tx];
        vm[1] = fm[l][i_shared - 1][tx];
        vm[2] = fm[l][i_shared][tx];
        vm[3] = fm[l][i_shared + 1][tx];
        vm[4] = fm[l][i_shared + 2][tx];
        vm[5] = fm[l][i_shared + 3][tx];
        vm[6] = fm[l][i_shared + 4][tx];

        gc(i, j, k, l) = WENO7(vp, vm, eps_here, if_shock);
      }
    }
  } else if (sch == 52 || sch == 72) {
    real rho_l = cv(i, j, k, 0);
    real rho_r = cv(i, j + 1, k, 0);
    // First, compute the Roe average of the half-point variables.
    real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
    const real rlc{1 / (rho_l + temp1)};
    const real rrc{1 / (temp1 + rho_r)};
    const real um{rlc * cv(i, j, k, 1) + rrc * cv(i, j + 1, k, 1)};
    const real vm{rlc * cv(i, j, k, 2) + rrc * cv(i, j + 1, k, 2)};
    const real wm{rlc * cv(i, j, k, 3) + rrc * cv(i, j + 1, k, 3)};

    real svm[MAX_SPEC_NUMBER] = {};
    for (int l = 0; l < n_var - 5; ++l) {
      svm[l] = rlc * cv(i, j, k, l + 5) + rrc * cv(i, j + 1, k, l + 5);
    }

    const int n_spec{param->n_spec};
    temp1 = 0; // temp1 = gas_constant (R)
    for (int l = 0; l < n_spec; ++l) {
      temp1 += svm[l] * param->gas_const[l];
    }
    real temp3 = (rlc * bv(i, j, k, 4) + rrc * bv(i, j + 1, k, 4)) / temp1; // temp1 = R, temp3 = T

    // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
    real fChar[5 + MAX_SPEC_NUMBER];
    real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
    compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
    real temp2{0};                                         // temp2 = cp
    for (int l = 0; l < n_spec; ++l) {
      temp2 += svm[l] * fChar[l];
    }
    const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
    const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
    const real gm1{gamma - 1};

    // Next, we compute the left characteristic matrix at i+1/2.
    const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i, j + 1, k)};
    real kx = zone->metric(i, j, k, 3) * jac_l + zone->metric(i, j + 1, k, 3) * jac_r;
    real ky = zone->metric(i, j, k, 4) * jac_l + zone->metric(i, j + 1, k, 4) * jac_r;
    real kz = zone->metric(i, j, k, 5) * jac_l + zone->metric(i, j + 1, k, 5) * jac_r;
    constexpr real eps{1e-40};
    const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kx * kx + ky * ky + kz * kz);
    temp1 = 1 / (jac_l + jac_r); // temp1 is 1/(jac_l + jac_r) in these 4 lines
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    const real Uk_bar{kx * um + ky * vm + kz * wm};

    // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
    //  const real cm2_inv{1.0 / (cm * cm)};
    temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
    // Compute the characteristic flux with L.
    // compute the partial derivative of pressure to species density
    for (int l = 0; l < n_spec; ++l) {
      hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
      // The computations including this alpha_l are all combined with a division by cm2.
      hI_alpI[l] *= temp2;
    }

    // Li Xinliang's flux splitting
    //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
    temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
    if (param->inviscid_scheme == 72) {
      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = -kx;
            L[0] = kx * (1 - temp3 * temp2) - (kz * vm - ky * wm) / cm;
            L[1] = kx * gm1 * um * temp2;
            L[2] = (kx * gm1 * vm + kz * cm) * temp2;
            L[3] = (kx * gm1 * wm - ky * cm) * temp2;
            L[4] = -kx * gm1 * temp2;
            break;
          case 2:
            temp1 = -ky;
            L[0] = ky * (1 - temp3 * temp2) - (kx * wm - kz * um) / cm;
            L[1] = (ky * gm1 * um - kz * cm) * temp2;
            L[2] = ky * gm1 * vm * temp2;
            L[3] = (ky * gm1 * wm + kx * cm) * temp2;
            L[4] = -ky * gm1 * temp2;
            break;
          case 3:
            temp1 = -kz;
            L[0] = kz * (1 - temp3 * temp2) - (ky * um - kx * vm) / cm;
            L[1] = (kz * gm1 * um + ky * cm) * temp2;
            L[2] = (kz * gm1 * vm - kx * cm) * temp2;
            L[3] = kz * gm1 * wm * temp2;
            L[4] = -kz * gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += L[n] * fp[n][i_shared - 3 + m][tx];
            vMinus[m] += L[n] * fm[n][i_shared - 2 + m][tx];
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += temp1 * hI_alpI[n] * fp[5 + n][i_shared - 3 + m][tx];
            vMinus[m] += temp1 * hI_alpI[n] * fm[5 + n][i_shared - 2 + m][tx];
          }
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
      }
      for (int l = 0; l < n_spec; ++l) {
        real vPlus[7], vMinus[7];
        for (int m = 0; m < 7; ++m) {
          vPlus[m] = -svm[l] * fp[0][i_shared - 3 + m][tx] + fp[5 + l][i_shared - 3 + m][tx];
          vMinus[m] = -svm[l] * fm[0][i_shared - 2 + m][tx] + fm[5 + l][i_shared - 2 + m][tx];
        }
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
      }
    } // temp2 is not 1/(c*c) anymore.

    // Project the flux back to physical space
    // We do not compute the right characteristic matrix here, because we explicitly write the components below.
    temp1 = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
    temp3 = fChar[0] - fChar[4];
    auto &gc = zone->gFlux;
    gc(i, j, k, 0) = temp1;
    gc(i, j, k, 1) = um * temp1 - cm * (kx * temp3 + kz * fChar[2] - ky * fChar[3]);
    gc(i, j, k, 2) = vm * temp1 - cm * (ky * temp3 + kx * fChar[3] - kz * fChar[1]);
    gc(i, j, k, 3) = wm * temp1 - cm * (kz * temp3 + ky * fChar[1] - kx * fChar[2]);

    temp2 = rlc * (cv(i, j, k, 4) + bv(i, j, k, 4)) +
            rrc * (cv(i, j + 1, k, 4) + bv(i, j + 1, k, 4)); // temp2 is Roe averaged enthalpy
    gc(i, j, k, 4) = temp2 * temp1 - cm * (Uk_bar * temp3 + (kx * cm / gm1 - kz * vm + ky * wm) * fChar[1]
                                           + (ky * cm / gm1 - kx * wm + kz * um) * fChar[2]
                                           + (kz * cm / gm1 - ky * um + kx * vm) * fChar[3]);

    temp2 = 0;
    for (int l = 0; l < n_spec; ++l) {
      gc(i, j, k, 5 + l) = svm[l] * temp1 + fChar[l + 5];
      temp2 += hI_alpI[l] * fChar[l + 5];
    }
    gc(i, j, k, 4) -= temp2 * cm * cm / gm1;
    // if (i == 0 && j == 0 && k == 0) {
    //   printf("new, fFlux = %e,%e,%e,%e,%e\n", gc(i, j, k, 0), gc(i, j, k, 1), gc(i, j, k, 2),
    //          gc(i, j, k, 3), gc(i, j, k, 4));
    // }
  }
  __syncthreads();

  // if (param->positive_preserving) {
  //   real dt{0};
  //   if (param->dt > 0)
  //     dt = param->dt;
  //   else
  //     dt = zone->dt_local(i, j, k);
  //   positive_preserving_limiter_1(param->dim, n_var, cv, i_shared, jac, dt, &fc[tid * n_var], metric, cc, fp);
  // }
  // __syncthreads();
  //
  // if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
  //   for (int l = 0; l < n_var; ++l) {
  //     zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
  //   }
  // }
}

__global__ void compute_derivative_y(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &gc = zone->gFlux;
  dq(i, j, k, 0) -= gc(i, j, k, 0) - gc(i, j - 1, k, 0);
  dq(i, j, k, 1) -= gc(i, j, k, 1) - gc(i, j - 1, k, 1);
  dq(i, j, k, 2) -= gc(i, j, k, 2) - gc(i, j - 1, k, 2);
  dq(i, j, k, 3) -= gc(i, j, k, 3) - gc(i, j - 1, k, 3);
  dq(i, j, k, 4) -= gc(i, j, k, 4) - gc(i, j - 1, k, 4);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) -= gc(i, j, k, l) - gc(i, j - 1, k, l);
  }
}

template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_z(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
  const int max_extent = zone->mz;
  if (k >= max_extent) return;

  const auto tid = static_cast<int>(threadIdx.z);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.z),
                            max_extent - static_cast<int>((blockDim.z - 1) * blockIdx.z) + 1);
  const int n_point = n_active + 2 * ngg - 1;

  extern __shared__ real s[];
  real *cv = s;
  real *p = &cv[n_point * n_var];
  real *cc = &p[n_point];
  real *metric = &cc[n_point];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];

  const int kl0 = static_cast<int>((blockDim.z - 1) * blockIdx.z) - ngg;
  for (int kl = kl0 + tid; kl <= kl0 + n_point - 1; kl += n_active) {
    int iSh = kl - kl0;                // iSh is the shared index
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_var + l] = zone->cv(i, j, kl, l);
    }
    p[iSh] = zone->bv(i, j, kl, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cc[iSh] = zone->acoustic_speed(i, j, kl);
    else
      cc[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, kl, 5));
    metric[iSh * 3] = zone->metric(i, j, kl, 6);
    metric[iSh * 3 + 1] = zone->metric(i, j, kl, 7);
    metric[iSh * 3 + 2] = zone->metric(i, j, kl, 8);
    jac[iSh] = zone->jac(i, j, kl);
    compute_flux(&cv[iSh * n_var], p[iSh], cc[iSh], param, &metric[iSh * 3], jac[iSh], &fp[iSh * n_var],
                 &fm[iSh * n_var]);
  }
  const int i_shared = tid - 1 + ngg;
  __syncthreads();

  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j, k + ii) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, p, param, metric, jac, &fc[tid * n_var], i_shared, fp, fm, if_shock);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_1(param->dim, n_var, cv, i_shared, jac, dt, &fc[tid * n_var], metric, cc, fp);
  }
  __syncthreads();

  if (tid > 0 && k >= zone->kMin && k <= zone->kMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_weno_z1(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  const int max_extent = zone->mz;
  if (k >= max_extent || i >= zone->mx) return;

  const auto tid = static_cast<int>(threadIdx.z), tx = static_cast<int>(threadIdx.x);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.z), max_extent - static_cast<int>(blockDim.z * blockIdx.z) + 1);
  const int n_point = n_active + 2 * ngg - 1;

  extern __shared__ real s[];
  constexpr int nyy = 32;
  auto fp = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
  auto fm = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

  const int kl0 = static_cast<int>(blockDim.z * blockIdx.z) - ngg;
  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  for (int kl = kl0 + tid; kl <= kl0 + n_point - 1; kl += n_active) {
    int iSh = kl - kl0; // iSh is the shared index
    real q[5 + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER], metric[3];
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      q[l] = cv(i, j, kl, l);
    }
    metric[0] = zone->metric(i, j, kl, 6);
    metric[1] = zone->metric(i, j, kl, 7);
    metric[2] = zone->metric(i, j, kl, 8);

    const real Uk{(q[1] * metric[0] + q[2] * metric[1] + q[3] * metric[2]) / q[0]};
    const real pk = bv(i, j, kl, 4);
    real cGradK = norm3d(metric[0], metric[1], metric[2]);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK *= zone->acoustic_speed(i, j, kl);
    else
      cGradK *= sqrt(gamma_air * R_air * bv(i, j, kl, 5));
    const real lambda0 = abs(Uk) + cGradK;
    const real jac = zone->jac(i, j, kl);

    fp[0][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[0]);
    fp[1][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[1] + pk * metric[0]);
    fp[2][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[2] + pk * metric[1]);
    fp[3][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[3] + pk * metric[2]);
    fp[4][iSh][tx] = 0.5 * jac * ((Uk + lambda0) * q[4] + pk * Uk);

    fm[0][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[0]);
    fm[1][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[1] + pk * metric[0]);
    fm[2][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[2] + pk * metric[1]);
    fm[3][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[3] + pk * metric[2]);
    fm[4][iSh][tx] = 0.5 * jac * ((Uk - lambda0) * q[4] + pk * Uk);

    for (int l = 5; l < n_var; ++l) {
      fp[l][iSh][tx] = 0.5 * jac * (Uk * q[l] + lambda0 * q[l]);
      fm[l][iSh][tx] = 0.5 * jac * (Uk * q[l] - lambda0 * q[l]);
    }
  }
  const int i_shared = tid - 1 + ngg;
  __syncthreads();

  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j, k + ii) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    constexpr real eps{1e-8};
    const real jac = zone->jac(i, j, k), jac_r = zone->jac(i, j, k + 1);
    const real eps_ref = eps * param->weno_eps_scale * 0.25 *
                         ((zone->metric(i, j, k, 3) * jac + zone->metric(i, j, k + 1, 3) * jac_r)
                          *
                          (zone->metric(i, j, k, 3) * jac + zone->metric(i, j, k + 1, 3) * jac_r) +
                          (zone->metric(i, j, k, 4) * jac + zone->metric(i, j, k + 1, 4) * jac_r)
                          *
                          (zone->metric(i, j, k, 4) * jac + zone->metric(i, j, k + 1, 4) * jac_r)
                          +
                          (zone->metric(i, j, k, 5) * jac + zone->metric(i, j, k + 1, 5) * jac_r)
                          *
                          (zone->metric(i, j, k, 6) * jac + zone->metric(i, j, k + 1, 5) * jac_r));
    real eps_scaled[3];
    eps_scaled[0] = eps_ref;
    eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
    eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

    auto &gc = zone->hFlux;
    for (int l = 0; l < n_var; ++l) {
      real eps_here{eps_scaled[0]};
      if (l == 1 || l == 2 || l == 3) {
        eps_here = eps_scaled[1];
      } else if (l == 4) {
        eps_here = eps_scaled[2];
      }

      if (param->inviscid_scheme == 71) {
        real vp[7], vm[7];
        vp[0] = fp[l][i_shared - 3][tx];
        vp[1] = fp[l][i_shared - 2][tx];
        vp[2] = fp[l][i_shared - 1][tx];
        vp[3] = fp[l][i_shared][tx];
        vp[4] = fp[l][i_shared + 1][tx];
        vp[5] = fp[l][i_shared + 2][tx];
        vp[6] = fp[l][i_shared + 3][tx];
        vm[0] = fm[l][i_shared - 2][tx];
        vm[1] = fm[l][i_shared - 1][tx];
        vm[2] = fm[l][i_shared][tx];
        vm[3] = fm[l][i_shared + 1][tx];
        vm[4] = fm[l][i_shared + 2][tx];
        vm[5] = fm[l][i_shared + 3][tx];
        vm[6] = fm[l][i_shared + 4][tx];

        gc(i, j, k, l) = WENO7(vp, vm, eps_here, if_shock);
      }
    }
  } else if (sch == 52 || sch == 72) {
    real rho_l = cv(i, j, k, 0);
    real rho_r = cv(i, j, k + 1, 0);
    // First, compute the Roe average of the half-point variables.
    real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
    const real rlc{1 / (rho_l + temp1)};
    const real rrc{1 / (temp1 + rho_r)};
    const real um{rlc * cv(i, j, k, 1) + rrc * cv(i, j, k + 1, 1)};
    const real vm{rlc * cv(i, j, k, 2) + rrc * cv(i, j, k + 1, 2)};
    const real wm{rlc * cv(i, j, k, 3) + rrc * cv(i, j, k + 1, 3)};

    real svm[MAX_SPEC_NUMBER] = {};
    for (int l = 0; l < n_var - 5; ++l) {
      svm[l] = rlc * cv(i, j, k, l + 5) + rrc * cv(i, j, k + 1, l + 5);
    }

    const int n_spec{param->n_spec};
    temp1 = 0; // temp1 = gas_constant (R)
    for (int l = 0; l < n_spec; ++l) {
      temp1 += svm[l] * param->gas_const[l];
    }
    real temp3 = (rlc * bv(i, j, k, 4) + rrc * bv(i, j, k + 1, 4)) / temp1; // temp1 = R, temp3 = T

    // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
    real fChar[5 + MAX_SPEC_NUMBER];
    real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
    compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
    real temp2{0};                                         // temp2 = cp
    for (int l = 0; l < n_spec; ++l) {
      temp2 += svm[l] * fChar[l];
    }
    const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
    const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
    const real gm1{gamma - 1};

    // Next, we compute the left characteristic matrix at i+1/2.
    const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i, j, k + 1)};
    real kx = zone->metric(i, j, k, 6) * jac_l + zone->metric(i, j, k + 1, 6) * jac_r;
    real ky = zone->metric(i, j, k, 7) * jac_l + zone->metric(i, j, k + 1, 7) * jac_r;
    real kz = zone->metric(i, j, k, 8) * jac_l + zone->metric(i, j, k + 1, 8) * jac_r;
    constexpr real eps{1e-40};
    const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kx * kx + ky * ky + kz * kz);
    temp1 = 1 / (jac_l + jac_r); // temp1 is 1/(jac_l + jac_r) in these 4 lines
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    const real Uk_bar{kx * um + ky * vm + kz * wm};

    // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
    //  const real cm2_inv{1.0 / (cm * cm)};
    temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
    // Compute the characteristic flux with L.
    // compute the partial derivative of pressure to species density
    for (int l = 0; l < n_spec; ++l) {
      hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
      // The computations including this alpha_l are all combined with a division by cm2.
      hI_alpI[l] *= temp2;
    }

    // Li Xinliang's flux splitting
    //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
    temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
    if (param->inviscid_scheme == 72) {
      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = -kx;
            L[0] = kx * (1 - temp3 * temp2) - (kz * vm - ky * wm) / cm;
            L[1] = kx * gm1 * um * temp2;
            L[2] = (kx * gm1 * vm + kz * cm) * temp2;
            L[3] = (kx * gm1 * wm - ky * cm) * temp2;
            L[4] = -kx * gm1 * temp2;
            break;
          case 2:
            temp1 = -ky;
            L[0] = ky * (1 - temp3 * temp2) - (kx * wm - kz * um) / cm;
            L[1] = (ky * gm1 * um - kz * cm) * temp2;
            L[2] = ky * gm1 * vm * temp2;
            L[3] = (ky * gm1 * wm + kx * cm) * temp2;
            L[4] = -ky * gm1 * temp2;
            break;
          case 3:
            temp1 = -kz;
            L[0] = kz * (1 - temp3 * temp2) - (ky * um - kx * vm) / cm;
            L[1] = (kz * gm1 * um + ky * cm) * temp2;
            L[2] = (kz * gm1 * vm - kx * cm) * temp2;
            L[3] = kz * gm1 * wm * temp2;
            L[4] = -kz * gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += L[n] * fp[n][i_shared - 3 + m][tx];
            vMinus[m] += L[n] * fm[n][i_shared - 2 + m][tx];
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += temp1 * hI_alpI[n] * fp[5 + n][i_shared - 3 + m][tx];
            vMinus[m] += temp1 * hI_alpI[n] * fm[5 + n][i_shared - 2 + m][tx];
          }
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
      }
      for (int l = 0; l < n_spec; ++l) {
        real vPlus[7], vMinus[7];
        for (int m = 0; m < 7; ++m) {
          vPlus[m] = -svm[l] * fp[0][i_shared - 3 + m][tx] + fp[5 + l][i_shared - 3 + m][tx];
          vMinus[m] = -svm[l] * fm[0][i_shared - 2 + m][tx] + fm[5 + l][i_shared - 2 + m][tx];
        }
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
      }
    } // temp2 is not 1/(c*c) anymore.

    // Project the flux back to physical space
    // We do not compute the right characteristic matrix here, because we explicitly write the components below.
    temp1 = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
    temp3 = fChar[0] - fChar[4];
    auto &hc = zone->hFlux;
    hc(i, j, k, 0) = temp1;
    hc(i, j, k, 1) = um * temp1 - cm * (kx * temp3 + kz * fChar[2] - ky * fChar[3]);
    hc(i, j, k, 2) = vm * temp1 - cm * (ky * temp3 + kx * fChar[3] - kz * fChar[1]);
    hc(i, j, k, 3) = wm * temp1 - cm * (kz * temp3 + ky * fChar[1] - kx * fChar[2]);

    temp2 = rlc * (cv(i, j, k, 4) + bv(i, j, k, 4)) +
            rrc * (cv(i, j, k + 1, 4) + bv(i, j, k + 1, 4)); // temp2 is Roe averaged enthalpy
    hc(i, j, k, 4) = temp2 * temp1 - cm * (Uk_bar * temp3 + (kx * cm / gm1 - kz * vm + ky * wm) * fChar[1]
                                           + (ky * cm / gm1 - kx * wm + kz * um) * fChar[2]
                                           + (kz * cm / gm1 - ky * um + kx * vm) * fChar[3]);

    temp2 = 0;
    for (int l = 0; l < n_spec; ++l) {
      hc(i, j, k, 5 + l) = svm[l] * temp1 + fChar[l + 5];
      temp2 += hI_alpI[l] * fChar[l + 5];
    }
    hc(i, j, k, 4) -= temp2 * cm * cm / gm1;
    // if (i == 0 && j == 0 && k == 0) {
    //   printf("new, fFlux = %e,%e,%e,%e,%e\n", gc(i, j, k, 0), gc(i, j, k, 1), gc(i, j, k, 2),
    //          gc(i, j, k, 3), gc(i, j, k, 4));
    // }
  }
  // __syncthreads();

  // if (param->positive_preserving) {
  //   real dt{0};
  //   if (param->dt > 0)
  //     dt = param->dt;
  //   else
  //     dt = zone->dt_local(i, j, k);
  //   positive_preserving_limiter_1(param->dim, n_var, cv, i_shared, jac, dt, &fc[tid * n_var], metric, cc, fp);
  // }
  // __syncthreads();
  //
  // if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
  //   for (int l = 0; l < n_var; ++l) {
  //     zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
  //   }
  // }
}

__global__ void compute_derivative_z(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &hc = zone->hFlux;
  dq(i, j, k, 0) -= hc(i, j, k, 0) - hc(i, j, k - 1, 0);
  dq(i, j, k, 1) -= hc(i, j, k, 1) - hc(i, j, k - 1, 1);
  dq(i, j, k, 2) -= hc(i, j, k, 2) - hc(i, j, k - 1, 2);
  dq(i, j, k, 3) -= hc(i, j, k, 3) - hc(i, j, k - 1, 3);
  dq(i, j, k, 4) -= hc(i, j, k, 4) - hc(i, j, k - 1, 4);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) -= hc(i, j, k, l) - hc(i, j, k - 1, l);
  }
}

template<MixtureModel mix_model>
void compute_convective_term_weno(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter) {
  // The implementation of classic WENO.
  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                       // fc
                     + n_computation_per_block * n_var * 2                   // F+/F-
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real);            // metric[3]

  dim3 TPB(block_dim, 1, 1);
  dim3 BPG((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
  compute_convective_term_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);

  constexpr int ny = 32;
  TPB = dim3(4, ny, 1);
  BPG = dim3((extent[0] - 1) / 4 + 1, (extent[1] + 1 - 1) / ny + 1, extent[2]);
  auto shared_mem1 = (ny + 2 * 4 - 1) * 4 * n_var * 2 * sizeof(real); // F+/F-
  compute_convective_term_weno_y1<mix_model><<<BPG, TPB, shared_mem1>>>(zone, param);
  TPB = dim3(16, 8, 8);
  BPG = dim3((extent[0] - 1) / 16 + 1, (extent[1] - 1) / 8 + 1, (extent[2] - 1) / 8 + 1);
  compute_derivative_y<<<BPG, TPB>>>(zone, param);

  if (extent[2] > 1) {
    // TPB = dim3(1, 1, 64);
    // BPG = dim3(extent[0], extent[1], (extent[2] - 1) / (64 - 1) + 1);
    // compute_convective_term_weno_z<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
    TPB = dim3(4, 1, ny);
    BPG = dim3((extent[0] - 1) / 4 + 1, extent[1], (extent[2] + 1 - 1) / ny + 1);
    compute_convective_term_weno_z1<mix_model><<<BPG, TPB, shared_mem1>>>(zone, param);
    TPB = dim3(16, 8, 8);
    BPG = dim3((extent[0] - 1) / 16 + 1, (extent[1] - 1) / 8 + 1, (extent[2] - 1) / 8 + 1);
    compute_derivative_z<<<BPG, TPB>>>(zone, param);
  }
}

__device__ void
compute_flux(const real *Q, const DParameter *param, const real *metric, real jac, real *Fk) {
  const int n_var = param->n_var;
  const real jacUk{jac * (metric[0] * Q[1] + metric[1] * Q[2] + metric[2] * Q[3]) / Q[0]};
  const real pk{Q[n_var]};

  Fk[0] = jacUk * Q[0];
  Fk[1] = jacUk * Q[1] + jac * pk * metric[0];
  Fk[2] = jacUk * Q[2] + jac * pk * metric[1];
  Fk[3] = jacUk * Q[3] + jac * pk * metric[2];
  Fk[4] = jacUk * (Q[4] + pk);

  for (int l = 5; l < n_var; ++l) {
    Fk[l] = jacUk * Q[l];
  }
}

__device__ void compute_flux(const real *Q, const real p, const real cc, const DParameter *param, const real *metric,
  real jac, real *Fp, real *Fm) {
  const int n_var = param->n_var;
  const real Uk{(Q[1] * metric[0] + Q[2] * metric[1] + Q[3] * metric[2]) / Q[0]};
  const real pk{p};
  const real cGradK = cc * sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
  const real lambda0 = abs(Uk) + cGradK;

  Fp[0] = 0.5 * jac * (Uk * Q[0] + lambda0 * Q[0]);
  Fp[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] + lambda0 * Q[1]);
  Fp[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] + lambda0 * Q[2]);
  Fp[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] + lambda0 * Q[3]);
  Fp[4] = 0.5 * jac * (Uk * (Q[4] + pk) + lambda0 * Q[4]);

  Fm[0] = 0.5 * jac * (Uk * Q[0] - lambda0 * Q[0]);
  Fm[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] - lambda0 * Q[1]);
  Fm[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] - lambda0 * Q[2]);
  Fm[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] - lambda0 * Q[3]);
  Fm[4] = 0.5 * jac * (Uk * (Q[4] + pk) - lambda0 * Q[4]);

  for (int l = 5; l < n_var; ++l) {
    Fp[l] = 0.5 * jac * (Uk * Q[l] + lambda0 * Q[l]);
    Fm[l] = 0.5 * jac * (Uk * Q[l] - lambda0 * Q[l]);
  }
}

__device__ void compute_flux(const real *Q, const DParameter *param, const real *metric, real jac, real *Fp, real *Fm,
  real p, real cc) {
  const int n_var = param->n_var;
  const real Uk{(Q[1] * metric[0] + Q[2] * metric[1] + Q[3] * metric[2]) / Q[0]};
  const real pk{p};
  const real cGradK = cc * sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
  const real lambda0 = abs(Uk) + cGradK;

  Fp[0] = 0.5 * jac * (Uk * Q[0] + lambda0 * Q[0]);
  Fp[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] + lambda0 * Q[1]);
  Fp[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] + lambda0 * Q[2]);
  Fp[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] + lambda0 * Q[3]);
  Fp[4] = 0.5 * jac * (Uk * (Q[4] + pk) + lambda0 * Q[4]);

  Fm[0] = 0.5 * jac * (Uk * Q[0] - lambda0 * Q[0]);
  Fm[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] - lambda0 * Q[1]);
  Fm[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] - lambda0 * Q[2]);
  Fm[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] - lambda0 * Q[3]);
  Fm[4] = 0.5 * jac * (Uk * (Q[4] + pk) - lambda0 * Q[4]);

  for (int l = 5; l < n_var; ++l) {
    Fp[l] = 0.5 * jac * (Uk * Q[l] + lambda0 * Q[l]);
    Fm[l] = 0.5 * jac * (Uk * Q[l] - lambda0 * Q[l]);
  }
}

template<MixtureModel mix_model>
__device__ void compute_weno_flux_ch(const real *cv, const real *p, DParameter *param, const real *metric,
  const real *jac, real *fci, int i_shared, const real *Fp, const real *Fm, bool if_shock) {
  const int n_var = param->n_var;

  // First, compute the Roe average of the half-point variables.
  const real *cvl{&cv[i_shared * n_var]};
  const real *cvr{&cv[(i_shared + 1) * n_var]};
  real temp1 = sqrt(cvl[0] * cvr[0]); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (cvl[0] + temp1)};
  const real rrc{1 / (temp1 + cvr[0])};
  const real um{rlc * cvl[1] + rrc * cvr[1]};
  const real vm{rlc * cvl[2] + rrc * cvr[2]};
  const real wm{rlc * cvl[3] + rrc * cvr[3]};

  real svm[MAX_SPEC_NUMBER] = {};
  for (int l = 0; l < n_var - 5; ++l) {
    svm[l] = rlc * cvl[l + 5] + rrc * cvr[l + 5];
  }

  const int n_spec{param->n_spec};
  temp1 = 0; // temp1 = gas_constant (R)
  for (int l = 0; l < n_spec; ++l) {
    temp1 += svm[l] * param->gas_const[l];
  }
  real temp3 = (rlc * p[i_shared] + rrc * p[i_shared + 1]) / temp1; // temp1 = R, temp3 = T

  // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
  real fChar[5 + MAX_SPEC_NUMBER];
  real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
  compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
  real temp2{0};                                         // temp2 = cp
  for (int l = 0; l < n_spec; ++l) {
    temp2 += svm[l] * fChar[l];
  }
  const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
  const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
  const real gm1{gamma - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  //  const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
  real kx = metric[i_shared * 3 + 0] * jac[i_shared] + metric[(i_shared + 1) * 3 + 0] * jac[i_shared + 1];
  real ky = metric[i_shared * 3 + 1] * jac[i_shared] + metric[(i_shared + 1) * 3 + 1] * jac[i_shared + 1];
  real kz = metric[i_shared * 3 + 2] * jac[i_shared] + metric[(i_shared + 1) * 3 + 2] * jac[i_shared + 1];
  constexpr real eps{1e-40};
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kx * kx + ky * ky + kz * kz);
  temp1 = 1 / (jac[i_shared] + jac[i_shared + 1]); // temp1 is 1/(jac_l + jac_r) in these 4 lines
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  //  const real cm2_inv{1.0 / (cm * cm)};
  temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
  // Compute the characteristic flux with L.
  // compute the partial derivative of pressure to species density
  for (int l = 0; l < n_spec; ++l) {
    hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
    // The computations including this alpha_l are all combined with a division by cm2.
    hI_alpI[l] *= temp2;
  }

  // Li Xinliang's flux splitting
  //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
  temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  if (param->inviscid_scheme == 72) {
    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = -kx;
          L[0] = kx * (1 - temp3 * temp2) - (kz * vm - ky * wm) / cm;
          L[1] = kx * gm1 * um * temp2;
          L[2] = (kx * gm1 * vm + kz * cm) * temp2;
          L[3] = (kx * gm1 * wm - ky * cm) * temp2;
          L[4] = -kx * gm1 * temp2;
          break;
        case 2:
          temp1 = -ky;
          L[0] = ky * (1 - temp3 * temp2) - (kx * wm - kz * um) / cm;
          L[1] = (ky * gm1 * um - kz * cm) * temp2;
          L[2] = ky * gm1 * vm * temp2;
          L[3] = (ky * gm1 * wm + kx * cm) * temp2;
          L[4] = -ky * gm1 * temp2;
          break;
        case 3:
          temp1 = -kz;
          L[0] = kz * (1 - temp3 * temp2) - (ky * um - kx * vm) / cm;
          L[1] = (kz * gm1 * um + ky * cm) * temp2;
          L[2] = (kz * gm1 * vm - kx * cm) * temp2;
          L[3] = kz * gm1 * wm * temp2;
          L[4] = -kz * gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[7] = {}, vMinus[7] = {};
      for (int m = 0; m < 7; ++m) {
        for (int n = 0; n < 5; ++n) {
          vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
          vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
        }
        for (int n = 0; n < n_spec; ++n) {
          vPlus[m] += temp1 * hI_alpI[n] * Fp[(i_shared - 3 + m) * n_var + 5 + n];
          vMinus[m] += temp1 * hI_alpI[n] * Fm[(i_shared - 2 + m) * n_var + 5 + n];
        }
      }
      fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
    }
    for (int l = 0; l < n_spec; ++l) {
      real vPlus[7], vMinus[7];
      for (int m = 0; m < 7; ++m) {
        vPlus[m] = -svm[l] * Fp[(i_shared - 3 + m) * n_var] + Fp[(i_shared - 3 + m) * n_var + 5 + l];
        vMinus[m] = -svm[l] * Fm[(i_shared - 2 + m) * n_var] + Fm[(i_shared - 2 + m) * n_var + 5 + l];
      }
      fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
    }
  } // temp2 is not 1/(c*c) anymore.

  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fci[0] = temp1;
  fci[1] = um * temp1 - cm * (kx * temp3 + kz * fChar[2] - ky * fChar[3]);
  fci[2] = vm * temp1 - cm * (ky * temp3 + kx * fChar[3] - kz * fChar[1]);
  fci[3] = wm * temp1 - cm * (kz * temp3 + ky * fChar[1] - kx * fChar[2]);

  temp2 = rlc * (cvl[4] + p[i_shared]) +
          rrc * (cvr[4] + p[i_shared + 1]); // temp2 is Roe averaged enthalpy
  fci[4] = temp2 * temp1 - cm * (Uk_bar * temp3 + (kx * cm / gm1 - kz * vm + ky * wm) * fChar[1]
                                 + (ky * cm / gm1 - kx * wm + kz * um) * fChar[2]
                                 + (kz * cm / gm1 - ky * um + kx * vm) * fChar[3]);

  temp2 = 0;
  for (int l = 0; l < n_spec; ++l) {
    fci[l + 5] = svm[l] * temp1 + fChar[l + 5];
    temp2 += hI_alpI[l] * fChar[l + 5];
  }
  fci[4] -= temp2 * cm * cm / gm1;
}

template<>
__device__ void compute_weno_flux_ch<MixtureModel::Air>(const real *cv, const real *p, DParameter *param,
  const real *metric, const real *jac, real *fci, int i_shared, const real *Fp, const real *Fm, bool if_shock) {
  const int n_var = param->n_var;

  // First, compute the Roe average of the half-point variables.
  const real *cvl{&cv[i_shared * n_var]};
  const real *cvr{&cv[(i_shared + 1) * n_var]};
  real temp1 = sqrt(cvl[0] * cvr[0]); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (cvl[0] + temp1)};
  const real rrc{1 / (temp1 + cvr[0])};
  const real um{rlc * cvl[1] + rrc * cvr[1]};
  const real vm{rlc * cvl[2] + rrc * cvr[2]};
  const real wm{rlc * cvl[3] + rrc * cvr[3]};
  constexpr real gm1{gamma_air - 1};
  const real hm = rlc * (cvl[4] + p[i_shared]) + rrc * (cvr[4] + p[i_shared + 1]);
  const real cm = sqrt(gm1 * (hm - 0.5 * (um * um + vm * vm + wm * wm)));

  // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
  real fChar[5];
  // Next, we compute the left characteristic matrix at i+1/2.
  //  const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
  real kx = metric[i_shared * 3 + 0] * jac[i_shared] + metric[(i_shared + 1) * 3 + 0] * jac[i_shared + 1];
  real ky = metric[i_shared * 3 + 1] * jac[i_shared] + metric[(i_shared + 1) * 3 + 1] * jac[i_shared + 1];
  real kz = metric[i_shared * 3 + 2] * jac[i_shared] + metric[(i_shared + 1) * 3 + 2] * jac[i_shared + 1];
  constexpr real eps{1e-40};
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kx * kx + ky * ky + kz * kz);
  temp1 = 1 / (jac[i_shared] + jac[i_shared + 1]); // temp1 is 1/(jac_l + jac_r) in these 4 lines
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  //  const real cm2_inv{1.0 / (cm * cm)};
  real temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.

  // Li Xinliang's flux splitting
  //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
  real temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  if (param->inviscid_scheme == 72) {
    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = -kx;
          L[0] = kx * (1 - temp3 * temp2) - (kz * vm - ky * wm) / cm;
          L[1] = kx * gm1 * um * temp2;
          L[2] = (kx * gm1 * vm + kz * cm) * temp2;
          L[3] = (kx * gm1 * wm - ky * cm) * temp2;
          L[4] = -kx * gm1 * temp2;
          break;
        case 2:
          temp1 = -ky;
          L[0] = ky * (1 - temp3 * temp2) - (kx * wm - kz * um) / cm;
          L[1] = (ky * gm1 * um - kz * cm) * temp2;
          L[2] = ky * gm1 * vm * temp2;
          L[3] = (ky * gm1 * wm + kx * cm) * temp2;
          L[4] = -ky * gm1 * temp2;
          break;
        case 3:
          temp1 = -kz;
          L[0] = kz * (1 - temp3 * temp2) - (ky * um - kx * vm) / cm;
          L[1] = (kz * gm1 * um + ky * cm) * temp2;
          L[2] = (kz * gm1 * vm - kx * cm) * temp2;
          L[3] = kz * gm1 * wm * temp2;
          L[4] = -kz * gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[7] = {}, vMinus[7] = {};
      for (int m = 0; m < 7; ++m) {
        for (int n = 0; n < 5; ++n) {
          vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
          vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
        }
      }
      fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
    }
  } // temp2 is not 1/(c*c) anymore.

  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fci[0] = temp1;
  fci[1] = um * temp1 - cm * (kx * temp3 + kz * fChar[2] - ky * fChar[3]);
  fci[2] = vm * temp1 - cm * (ky * temp3 + kx * fChar[3] - kz * fChar[1]);
  fci[3] = wm * temp1 - cm * (kz * temp3 + ky * fChar[1] - kx * fChar[2]);

  temp2 = rlc * (cvl[4] + p[i_shared]) +
          rrc * (cvr[4] + p[i_shared + 1]); // temp2 is Roe averaged enthalpy
  fci[4] = temp2 * temp1 - cm * (Uk_bar * temp3 + (kx * cm / gm1 - kz * vm + ky * wm) * fChar[1]
                                 + (ky * cm / gm1 - kx * wm + kz * um) * fChar[2]
                                 + (kz * cm / gm1 - ky * um + kx * vm) * fChar[3]);
}

// template<MixtureModel mix_model>
// __device__ void
// compute_weno_flux_ch(const real *cv, const real *p, DParameter *param, int tid, const real *metric, const real *jac,
//   real *fc, int i_shared, real *Fp, real *Fm, bool if_shock) {
//   const int n_var = param->n_var;
//
//   // 0: acans; 1: li xinliang(own flux splitting); 2: my(same spectral radius)
//   constexpr int method = 1;
//
//   int weno_scheme_i = 4;
//   if (param->inviscid_scheme == 52) {
//     weno_scheme_i = 3;
//   }
//
//   const auto m_l = &metric[i_shared * 3], m_r = &metric[(i_shared + 1) * 3];
//
//   // The first n_var in the cv array is conservative vars, followed by p and cm.
//   const real *cvl{&cv[i_shared * n_var]};
//   const real *cvr{&cv[(i_shared + 1) * n_var]};
//   const real rhoL_inv{1.0 / cvl[0]}, rhoR_inv{1.0 / cvr[0]};
//   // First, compute the Roe average of the half-point variables.
//   const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//   const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//   const real um{rlc * cvl[1] * rhoL_inv + rrc * cvr[1] * rhoR_inv};
//   const real vm{rlc * cvl[2] * rhoL_inv + rrc * cvr[2] * rhoR_inv};
//   const real wm{rlc * cvl[3] * rhoL_inv + rrc * cvr[3] * rhoR_inv};
//   const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
//   const real hl{(cvl[4] + p[i_shared]) * rhoL_inv};
//   const real hr{(cvr[4] + p[i_shared + 1]) * rhoR_inv};
//   const real hm{rlc * hl + rrc * hr};
//
//   real svm[MAX_SPEC_NUMBER] = {};
//   for (int l = 0; l < n_var - 5; ++l) {
//     svm[l] = rlc * cvl[l + 5] * rhoL_inv + rrc * cvr[l + 5] * rhoR_inv;
//   }
//
//   const int n_spec{param->n_spec};
//   real mw_inv = 0;
//   for (int l = 0; l < n_spec; ++l) {
//     mw_inv += svm[l] * param->imw[l];
//   }
//
//   const real tl{p[i_shared] * rhoL_inv};
//   const real tr{p[i_shared + 1] * rhoR_inv};
//   const real tm = (rlc * tl + rrc * tr) / (R_u * mw_inv);
//
//   real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
//   compute_enthalpy_and_cp(tm, h_i, cp_i, param);
//   real cp{0}, cv_tot{0};
//   for (int l = 0; l < n_spec; ++l) {
//     cp += svm[l] * cp_i[l];
//     cv_tot += svm[l] * (cp_i[l] - param->gas_const[l]);
//   }
//   const real gamma = cp / cv_tot;
//   const real cm = sqrt(gamma * R_u * mw_inv * tm);
//   const real gm1{gamma - 1};
//
//   // Next, we compute the left characteristic matrix at i+1/2.
//   const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
//   real kxJ{m_l[0] * jac_l + m_r[0] * jac_r};
//   real kyJ{m_l[1] * jac_l + m_r[1] * jac_r};
//   real kzJ{m_l[2] * jac_l + m_r[2] * jac_r};
//   real kx{kxJ / (jac_l + jac_r)};
//   real ky{kyJ / (jac_l + jac_r)};
//   real kz{kzJ / (jac_l + jac_r)};
//   const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
//   kx /= gradK;
//   ky /= gradK;
//   kz /= gradK;
//   const real Uk_bar{kx * um + ky * vm + kz * wm};
//   const real alpha{gm1 * ekm};
//
//   // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
//   const real cm2_inv{1.0 / (cm * cm)};
//   // Compute the characteristic flux with L.
//   real fChar[5 + MAX_SPEC_NUMBER];
//   constexpr real eps{1e-40};
//   const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);
//
//   real alpha_l[MAX_SPEC_NUMBER];
//   // compute the partial derivative of pressure to species density
//   for (int l = 0; l < n_spec; ++l) {
//     alpha_l[l] = gamma * param->gas_const[l] * tm - (gamma - 1) * h_i[l];
//     // The computations including this alpha_l are all combined with a division by cm2.
//     alpha_l[l] *= cm2_inv;
//   }
//
//   if constexpr (method == 1) {
//     // Li Xinliang's flux splitting
//     // if (param->positive_preserving) {
//     //   real spectralRadThis = abs((m_l[0] * cvl[1] + m_l[1] * cvl[2] + m_l[2] * cvl[3]) * rhoL_inv +
//     //                              cvl[n_var + 1] * sqrt(m_l[0] * m_l[0] + m_l[1] * m_l[1] + m_l[2] * m_l[2]));
//     //   real spectralRadNext = abs((m_r[0] * cvr[1] + m_r[1] * cvr[2] + m_r[2] * cvr[3]) * rhoR_inv +
//     //                              cvr[n_var + 1] * sqrt(m_r[0] * m_r[0] + m_r[1] * m_r[1] + m_r[2] * m_r[2]));
//     //   for (int l = 0; l < n_var - 5; ++l) {
//     //     f_1st[tid * (n_var - 5) + l] =
//     //         0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cvl[l + 5] * jac_l) +
//     //         0.5 * (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cvr[l + 5] * jac_r);
//     //   }
//     // }
//
//     if (param->inviscid_scheme == 52) {
//       for (int l = 0; l < 5; ++l) {
//         real coeff_alpha_s{0.5};
//         real L[5];
//         switch (l) {
//           case 0:
//             L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           case 1:
//             coeff_alpha_s = -kx;
//             L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//             L[1] = kx * gm1 * um * cm2_inv;
//             L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
//             L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
//             L[4] = -kx * gm1 * cm2_inv;
//             break;
//           case 2:
//             coeff_alpha_s = -ky;
//             L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//             L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
//             L[2] = ky * gm1 * vm * cm2_inv;
//             L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
//             L[4] = -ky * gm1 * cm2_inv;
//             break;
//           case 3:
//             coeff_alpha_s = -kz;
//             L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//             L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
//             L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
//             L[3] = kz * gm1 * wm * cm2_inv;
//             L[4] = -kz * gm1 * cm2_inv;
//             break;
//           case 4:
//             L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           default:
//             break;
//         }
//         real vPlus[5] = {}, vMinus[5] = {};
//         for (int m = 0; m < 5; ++m) {
//           for (int n = 0; n < 5; ++n) {
//             vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
//             vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
//           }
//           for (int n = 0; n < n_spec; ++n) {
//             vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 2 + m) * n_var + 5 + n];
//             vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 1 + m) * n_var + 5 + n];
//           }
//         }
//         fChar[l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       }
//       for (int l = 0; l < n_spec; ++l) {
//         real vPlus[5], vMinus[5];
//         for (int m = 0; m < 5; ++m) {
//           vPlus[m] = -svm[l] * Fp[(i_shared - 2 + m) * n_var] + Fp[(i_shared - 2 + m) * n_var + 5 + l];
//           vMinus[m] = -svm[l] * Fm[(i_shared - 1 + m) * n_var] + Fm[(i_shared - 1 + m) * n_var + 5 + l];
//         }
//         fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     } else if (param->inviscid_scheme == 72) {
//       for (int l = 0; l < 5; ++l) {
//         real coeff_alpha_s{0.5};
//         real L[5];
//         switch (l) {
//           case 0:
//             L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           case 1:
//             coeff_alpha_s = -kx;
//             L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//             L[1] = kx * gm1 * um * cm2_inv;
//             L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
//             L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
//             L[4] = -kx * gm1 * cm2_inv;
//             break;
//           case 2:
//             coeff_alpha_s = -ky;
//             L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//             L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
//             L[2] = ky * gm1 * vm * cm2_inv;
//             L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
//             L[4] = -ky * gm1 * cm2_inv;
//             break;
//           case 3:
//             coeff_alpha_s = -kz;
//             L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//             L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
//             L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
//             L[3] = kz * gm1 * wm * cm2_inv;
//             L[4] = -kz * gm1 * cm2_inv;
//             break;
//           case 4:
//             L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           default:
//             break;
//         }
//
//         real vPlus[7] = {}, vMinus[7] = {};
//         for (int m = 0; m < 7; ++m) {
//           for (int n = 0; n < 5; ++n) {
//             vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
//             vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
//           }
//           for (int n = 0; n < n_spec; ++n) {
//             vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 3 + m) * n_var + 5 + n];
//             vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 2 + m) * n_var + 5 + n];
//           }
//         }
//         fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//       for (int l = 0; l < n_spec; ++l) {
//         real vPlus[7], vMinus[7];
//         for (int m = 0; m < 7; ++m) {
//           vPlus[m] = -svm[l] * Fp[(i_shared - 3 + m) * n_var] + Fp[(i_shared - 3 + m) * n_var + 5 + l];
//           vMinus[m] = -svm[l] * Fm[(i_shared - 2 + m) * n_var] + Fm[(i_shared - 2 + m) * n_var + 5 + l];
//         }
//         fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     }
//   } else {
//     // My method
//     const int weno_size = 2 * weno_scheme_i;
//     real spec_rad[3] = {}, spectralRadThis, spectralRadNext;
//     bool pp_limiter{param->positive_preserving};
//     for (int l = 0; l < weno_size; ++l) {
//       const auto ls = i_shared + l - weno_scheme_i + 1;
//       const real *Q = &cv[ls * (n_var + 2)];
//       real c = Q[n_var + 1];
//       real grad_k = norm3d(metric[ls * 3], metric[ls * 3 + 1], metric[ls * 3 + 2]);
//       real Uk = (metric[ls * 3] * Q[1] + metric[ls * 3 + 1] * Q[2] + metric[ls * 3 + 2] * Q[3]) / Q[0];
//       real ukPc = abs(Uk + c * grad_k);
//       real ukMc = abs(Uk - c * grad_k);
//       spec_rad[0] = max(spec_rad[0], ukMc);
//       spec_rad[1] = max(spec_rad[1], abs(Uk));
//       spec_rad[2] = max(spec_rad[2], ukPc);
//       if (pp_limiter && l == weno_scheme_i - 1)
//         spectralRadThis = ukPc;
//       if (pp_limiter && l == weno_scheme_i)
//         spectralRadNext = ukPc;
//     }
//
//     // if (pp_limiter) {
//     //   for (int l = 0; l < n_var - 5; ++l) {
//     //     f_1st[tid * (n_var - 5) + l] =
//     //         0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cv[i_shared * (n_var + 2) + l + 5] * jac_l) +
//     //         0.5 *
//     //         (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cv[(i_shared + 1) * (n_var + 2) + l + 5] * jac_r);
//     //   }
//     // }
//
//     for (int l = 0; l < 5; ++l) {
//       real lambda_l{spec_rad[1]};
//       real coeff_alpha_s{0.5};
//       real L[5];
//       switch (l) {
//         case 0:
//           lambda_l = spec_rad[0];
//           L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//           L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//           L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//           L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//           L[4] = gm1 * cm2_inv * 0.5;
//           break;
//         case 1:
//           coeff_alpha_s = -kx;
//           L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//           L[1] = kx * gm1 * um * cm2_inv;
//           L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
//           L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
//           L[4] = -kx * gm1 * cm2_inv;
//           break;
//         case 2:
//           coeff_alpha_s = -ky;
//           L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//           L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
//           L[2] = ky * gm1 * vm * cm2_inv;
//           L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
//           L[4] = -ky * gm1 * cm2_inv;
//           break;
//         case 3:
//           coeff_alpha_s = -kz;
//           L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//           L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
//           L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
//           L[3] = kz * gm1 * wm * cm2_inv;
//           L[4] = -kz * gm1 * cm2_inv;
//           break;
//         case 4:
//           lambda_l = spec_rad[2];
//           L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//           L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//           L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//           L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//           L[4] = gm1 * cm2_inv * 0.5;
//           break;
//         default:
//           break;
//       }
//
//       real vPlus[7] = {}, vMinus[7] = {};
//       for (int m = 0; m < weno_size - 1; ++m) {
//         const auto is1 = i_shared - weno_scheme_i + m + 1, is2 = i_shared - weno_scheme_i + m + 2;
//         for (int n = 0; n < 5; ++n) {
//           vPlus[m] += L[n] * (Fp[is1 * n_var + n] + lambda_l * cv[is1 * (n_var + 2) + n] * jac[is1]);
//           vMinus[m] += L[n] * (Fp[is2 * n_var + n] - lambda_l * cv[is2 * (n_var + 2) + n] * jac[is2]);
//         }
//         for (int n = 0; n < n_spec; ++n) {
//           vPlus[m] += coeff_alpha_s * alpha_l[n] * (
//             Fp[is1 * n_var + 5 + n] + lambda_l * cv[is1 * (n_var + 2) + 5 + n] * jac[is1]);
//           vMinus[m] += coeff_alpha_s * alpha_l[n] * (
//             Fp[is2 * n_var + 5 + n] - lambda_l * cv[is2 * (n_var + 2) + 5 + n] * jac[is2]);
//         }
//         vPlus[m] *= 0.5;
//         vMinus[m] *= 0.5;
//       }
//       if (weno_scheme_i == 3) {
//         // WENO-5
//         fChar[l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       } else if (weno_scheme_i == 4) {
//         // WENO-7
//         fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     }
//     for (int l = 0; l < n_spec; ++l) {
//       const real lambda_l{spec_rad[1]};
//       real vPlus[7] = {}, vMinus[7] = {};
//       for (int m = 0; m < weno_size - 1; ++m) {
//         const auto is1 = i_shared - weno_scheme_i + m + 1, is2 = i_shared - weno_scheme_i + m + 2;
//         vPlus[m] = 0.5 * (Fp[is1 * n_var + 5 + l] + lambda_l * cv[is1 * (n_var + 2) + 5 + l] * jac[is1] -
//                           svm[l] * (Fp[is1 * n_var] + lambda_l * cv[is1 * (n_var + 2)] * jac[is1]));
//         vMinus[m] = 0.5 * (Fp[is2 * n_var + 5 + l] -
//                            lambda_l * cv[is2 * (n_var + 2) + 5 + l] * jac[is2] -
//                            svm[l] * (Fp[is2 * n_var] - lambda_l * cv[is2 * (n_var + 2)] * jac[is2]));
//       }
//       if (weno_scheme_i == 3) {
//         // WENO-5
//         fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       } else if (weno_scheme_i == 4) {
//         // WENO-7
//         fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     }
//   }
//
//   // Project the flux back to physical space
//   // We do not compute the right characteristic matrix here, because we explicitly write the components below.
//   auto fci = &fc[tid * n_var];
//   fci[0] = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//   fci[1] = (um - kx * cm) * fChar[0] + kx * um * fChar[1] + (ky * um - kz * cm) * fChar[2] +
//            (kz * um + ky * cm) * fChar[3] + (um + kx * cm) * fChar[4];
//   fci[2] = (vm - ky * cm) * fChar[0] + (kx * vm + kz * cm) * fChar[1] + ky * vm * fChar[2] +
//            (kz * vm - kx * cm) * fChar[3] + (vm + ky * cm) * fChar[4];
//   fci[3] = (wm - kz * cm) * fChar[0] + (kx * wm - ky * cm) * fChar[1] + (ky * wm + kx * cm) * fChar[2] +
//            kz * wm * fChar[3] + (wm + kz * cm) * fChar[4];
//
//   fci[4] = (hm - Uk_bar * cm) * fChar[0] + (kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm) * fChar[1] +
//            (ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm) * fChar[2] +
//            (kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm) * fChar[3] +
//            (hm + Uk_bar * cm) * fChar[4];
//   real add{0};
//   const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//   for (int l = 0; l < n_spec; ++l) {
//     fci[5 + l] = svm[l] * coeff_add + fChar[l + 5];
//     add += alpha_l[l] * fChar[l + 5];
//   }
//   fci[4] -= add * cm * cm / gm1;
// }
//
// // The above function can actually realize the following ability, but the speed is slower than the specific version.
// // Thus, we keep the current version.
// template<>
// __device__ void compute_weno_flux_ch<MixtureModel::Air>(const real *cv, const real *p, DParameter *param, int tid,
//   const real *metric,
//   const real *jac, real *fc, int i_shared, real *Fp, real *Fm,
//   bool if_shock) {
//   const int n_var = param->n_var;
//
//   // 0: acans; 1: li xinliang(own flux splitting); 2: my(same spectral radius)
//   constexpr int method = 2;
//
//   if constexpr (method == 1) {
//     // compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var],
//                  // &Fm[i_shared * n_var], TODO, TODO);
//     // for (size_t i = 0; i < n_add; i++) {
//       // compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
//                    // &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var], TODO, TODO);
//     // }
//   } else if constexpr (method == 2) {
//     compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var]);
//     // for (size_t i = 0; i < n_add; i++) {
//       // compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
//                    // &Fp[ig_shared[i] * n_var]);
//     // }
//   }
//
//   // The first n_var in the cv array is conservative vars, followed by p and cm.
//   const real *cvl{&cv[i_shared * (n_var + 2)]};
//   const real *cvr{&cv[(i_shared + 1) * (n_var + 2)]};
//   // First, compute the Roe average of the half-point variables.
//   const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//   const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
//   const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
//   const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
//   const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
//   const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
//   constexpr real gm1{gamma_air - 1};
//   const real hl{(cvl[4] + cvl[n_var]) / cvl[0]};
//   const real hr{(cvr[4] + cvr[n_var]) / cvr[0]};
//   const real hm{rlc * hl + rrc * hr};
//   const real cm2{gm1 * (hm - ekm)};
//   const real cm{sqrt(cm2)};
//
//   // Next, we compute the left characteristic matrix at i+1/2.
//   real kx{
//     (jac[i_shared] * metric[i_shared * 3] + jac[i_shared + 1] * metric[(i_shared + 1) * 3]) /
//     (jac[i_shared] + jac[i_shared + 1])
//   };
//   real ky{
//     (jac[i_shared] * metric[i_shared * 3 + 1] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1]) /
//     (jac[i_shared] + jac[i_shared + 1])
//   };
//   real kz{
//     (jac[i_shared] * metric[i_shared * 3 + 2] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2]) /
//     (jac[i_shared] + jac[i_shared + 1])
//   };
//   const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
//   kx /= gradK;
//   ky /= gradK;
//   kz /= gradK;
//   const real Uk_bar{kx * um + ky * vm + kz * wm};
//   const real alpha{gm1 * ekm};
//   const real cm2_inv{1.0 / (cm * cm)};
//
//   // Compute the characteristic flux with L.
//   real fChar[5];
//   constexpr real eps{1e-40};
//   const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
//   const real eps_scaled = eps * param->weno_eps_scale * 0.25 *
//                           ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
//                            (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
//                            (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
//                            (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
//                            (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
//                            (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
//
//   if constexpr (method == 0) {
//     // ACANS version
//     real ap[3], an[3];
//     const auto max_spec_rad = abs(Uk_bar) + cm;
//     ap[0] = 0.5 * (Uk_bar - cm + max_spec_rad) * gradK;
//     ap[1] = 0.5 * (Uk_bar + max_spec_rad) * gradK;
//     ap[2] = 0.5 * (Uk_bar + cm + max_spec_rad) * gradK;
//     an[0] = 0.5 * (Uk_bar - cm - max_spec_rad) * gradK;
//     an[1] = 0.5 * (Uk_bar - max_spec_rad) * gradK;
//     an[2] = 0.5 * (Uk_bar + cm - max_spec_rad) * gradK;
//     if (param->inviscid_scheme == 52) {
//       for (int l = 0; l < 5; ++l) {
//         real vPlus[5] = {}, vMinus[5] = {};
//         // ACANS version
//         real lambda_p{ap[1]}, lambda_n{an[1]};
//         if (l == 0) {
//           lambda_p = ap[0];
//           lambda_n = an[0];
//         } else if (l == 4) {
//           lambda_p = ap[2];
//           lambda_n = an[2];
//         }
//         for (int m = 0; m < 5; m++) {
//           for (int n = 0; n < 5; ++n) {
//             // vPlus[m] += lambda_p * LR(l, n) * cv[(i_shared - 2 + m) * (n_var + 2) + n] * 0.5 *
//             //     (jac[i_shared] + jac[i_shared + 1]);
//             // vMinus[m] += lambda_n * LR(l, n) * cv[(i_shared - 1 + m) * (n_var + 2) + n] * 0.5 *
//             //     (jac[i_shared] + jac[i_shared + 1]);
//           }
//         }
//         fChar[l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     } else if (param->inviscid_scheme == 72) {
//       for (int l = 0; l < 5; ++l) {
//         real vPlus[7] = {}, vMinus[7] = {};
//         // ACANS version
//         real lambda_p{ap[1]}, lambda_n{an[1]};
//         if (l == 0) {
//           lambda_p = ap[0];
//           lambda_n = an[0];
//         } else if (l == 4) {
//           lambda_p = ap[2];
//           lambda_n = an[2];
//         }
//         for (int m = 0; m < 7; m++) {
//           for (int n = 0; n < 5; ++n) {
//             // vPlus[m] += lambda_p * LR(l, n) * cv[(i_shared - 3 + m) * (n_var + 2) + n] * 0.5 *
//             //     (jac[i_shared] + jac[i_shared + 1]);
//             // vMinus[m] += lambda_n * LR(l, n) * cv[(i_shared - 2 + m) * (n_var + 2) + n] * 0.5 *
//             //     (jac[i_shared] + jac[i_shared + 1]);
//           }
//         }
//         fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     }
//   } else if constexpr (method == 1) {
//     // Li Xinliang version
//     // if (param->inviscid_scheme == 52) {
//     //   for (int l = 0; l < 5; ++l) {
//     //     real vPlus[5] = {}, vMinus[5] = {};
//     //     for (int m = 0; m < 5; m++) {
//     //       for (int n = 0; n < 5; n++) {
//     //         vPlus[m] += LR(l, n) * Fp[(i_shared - 2 + m) * 5 + n];
//     //         vMinus[m] += LR(l, n) * Fm[(i_shared - 1 + m) * 5 + n];
//     //       }
//     //     }
//     //     fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
//     //   }
//     // } else if (param->inviscid_scheme == 72) {
//     //   for (int l = 0; l < 5; ++l) {
//     //     real vPlus[7] = {}, vMinus[7] = {};
//     //     for (int m = 0; m < 7; ++m) {
//     //       for (int n = 0; n < 5; ++n) {
//     //         vPlus[m] += LR(l, n) * Fp[(i_shared - 3 + m) * 5 + n];
//     //         vMinus[m] += LR(l, n) * Fm[(i_shared - 2 + m) * 5 + n];
//     //       }
//     //     }
//     //     fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//     //   }
//     // }
//   } else {
//     // My method
//     real spec_rad[3] = {};
//
//     if (param->inviscid_scheme == 52) {
//       for (int l = -2; l < 4; ++l) {
//         const real *Q = &cv[(i_shared + l) * (n_var + 2)];
//         real c = sqrt(gamma_air * Q[n_var] / Q[0]);
//         real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
//                            metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
//                            metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
//         real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
//                    metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
//         real ukPc = abs(Uk + c * grad_k);
//         real ukMc = abs(Uk - c * grad_k);
//         spec_rad[0] = max(spec_rad[0], ukMc);
//         spec_rad[1] = max(spec_rad[1], abs(Uk));
//         spec_rad[2] = max(spec_rad[2], ukPc);
//       }
//       spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
//       spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
//       spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));
//
//       for (int l = 0; l < 5; ++l) {
//         real lambda_l{spec_rad[1]};
//         if (l == 0) {
//           lambda_l = spec_rad[0];
//         } else if (l == 4) {
//           lambda_l = spec_rad[2];
//         }
//
//         real vPlus[5] = {}, vMinus[5] = {};
//         for (int m = 0; m < 5; ++m) {
//           for (int n = 0; n < 5; ++n) {
//             // vPlus[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * 5 + n] + lambda_l * cv[(i_shared - 2 + m) * 7 + n] *
//             //                         jac[i_shared - 2 + m]);
//             // vMinus[m] += LR(l, n) * (Fm[(i_shared - 1 + m) * 5 + n] - lambda_l * cv[(i_shared - 1 + m) * 7 + n] *
//             //                          jac[i_shared - 1 + m]);
//           }
//           vPlus[m] *= 0.5;
//           vMinus[m] *= 0.5;
//         }
//         fChar[l] = WENO5(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     } else if (param->inviscid_scheme == 72) {
//       for (int l = -3; l < 5; ++l) {
//         const real *Q = &cv[(i_shared + l) * (n_var + 2)];
//         real c = Q[n_var + 1];
//         real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
//                            metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
//                            metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
//         real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
//                    metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
//         real ukPc = abs(Uk + c * grad_k);
//         real ukMc = abs(Uk - c * grad_k);
//         spec_rad[0] = max(spec_rad[0], ukMc);
//         spec_rad[1] = max(spec_rad[1], abs(Uk));
//         spec_rad[2] = max(spec_rad[2], ukPc);
//       }
//
//       for (int l = 0; l < 5; ++l) {
//         real lambda_l{spec_rad[1]};
//         real L[5];
//         switch (l) {
//           case 0:
//             lambda_l = spec_rad[0];
//             L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           case 1:
//             L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//             L[1] = kx * gm1 * um * cm2_inv;
//             L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
//             L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
//             L[4] = -kx * gm1 * cm2_inv;
//             break;
//           case 2:
//             L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//             L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
//             L[2] = ky * gm1 * vm * cm2_inv;
//             L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
//             L[4] = -ky * gm1 * cm2_inv;
//             break;
//           case 3:
//             L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//             L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
//             L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
//             L[3] = kz * gm1 * wm * cm2_inv;
//             L[4] = -kz * gm1 * cm2_inv;
//             break;
//           case 4:
//             lambda_l = spec_rad[2];
//             L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//             L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//             L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//             L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//             L[4] = gm1 * cm2_inv * 0.5;
//             break;
//           default:
//             break;
//         }
//
//         real vPlus[7] = {}, vMinus[7] = {};
//         for (int m = 0; m < 7; ++m) {
//           for (int n = 0; n < 5; ++n) {
//             vPlus[m] += L[n] * (Fp[(i_shared - 3 + m) * 5 + n] + lambda_l * cv[(i_shared - 3 + m) * 7 + n] *
//                                 jac[i_shared - 3 + m]);
//             vMinus[m] += L[n] * (Fm[(i_shared - 2 + m) * 5 + n] - lambda_l * cv[(i_shared - 2 + m) * 7 + n] *
//                                  jac[i_shared - 2 + m]);
//           }
//           vPlus[m] *= 0.5;
//           vMinus[m] *= 0.5;
//         }
//         fChar[l] = WENO7(vPlus, vMinus, eps_scaled, if_shock);
//       }
//     }
//   }
//
//   // Project the flux back to physical space
//   auto fci = &fc[tid * n_var];
//   fci[0] = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//   fci[1] = (um - kx * cm) * fChar[0] + kx * um * fChar[1] + (ky * um - kz * cm) * fChar[2] +
//            (kz * um + ky * cm) * fChar[3] + (um + kx * cm) * fChar[4];
//   fci[2] = (vm - ky * cm) * fChar[0] + (kx * vm + kz * cm) * fChar[1] + ky * vm * fChar[2] +
//            (kz * vm - kx * cm) * fChar[3] + (vm + ky * cm) * fChar[4];
//   fci[3] = (wm - kz * cm) * fChar[0] + (kx * wm - ky * cm) * fChar[1] + (ky * wm + kx * cm) * fChar[2] +
//            kz * wm * fChar[3] + (wm + kz * cm) * fChar[4];
//
//   fci[4] = (hm - Uk_bar * cm) * fChar[0] + (kx * alpha / gm1 + (kz * vm - ky * wm) * cm) * fChar[1] +
//            (ky * alpha / gm1 + (kx * wm - kz * um) * cm) * fChar[2] +
//            (kz * alpha / gm1 + (ky * um - kx * vm) * cm) * fChar[3] +
//            (hm + Uk_bar * cm) * fChar[4];
// }

__device__ void
compute_weno_flux_cp(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
  int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st, bool if_shock) {
  const int n_var = param->n_var;

  // compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var],
  // &Fm[i_shared * n_var], TODO, TODO);
  for (size_t i = 0; i < n_add; i++) {
    // compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
    // &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var], TODO, TODO);
  }
  __syncthreads();

  //  const real eps_ref = 1e-6 * param->weno_eps_scale;
  constexpr real eps{1e-20};
  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
  const real eps_ref = eps * param->weno_eps_scale * 0.25 *
                       ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
                        (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
  real eps_scaled[3];
  eps_scaled[0] = eps_ref;
  eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
  eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

  if (param->positive_preserving) {
    for (int l = 0; l < n_var - 5; ++l) {
      f_1st[tid * (n_var - 5) + l] = 0.5 * Fp[i_shared * n_var + l + 5] + 0.5 * Fm[(i_shared + 1) * n_var + l + 5];
    }
  }

  const auto fci = &fc[tid * n_var];

  for (int l = 0; l < n_var; ++l) {
    real eps_here{eps_scaled[0]};
    if (l == 1 || l == 2 || l == 3) {
      eps_here = eps_scaled[1];
    } else if (l == 4) {
      eps_here = eps_scaled[2];
    }

    if (param->inviscid_scheme == 51) {
      real vp[5], vm[5];
      vp[0] = Fp[(i_shared - 2) * n_var + l];
      vp[1] = Fp[(i_shared - 1) * n_var + l];
      vp[2] = Fp[i_shared * n_var + l];
      vp[3] = Fp[(i_shared + 1) * n_var + l];
      vp[4] = Fp[(i_shared + 2) * n_var + l];
      vm[0] = Fm[(i_shared - 1) * n_var + l];
      vm[1] = Fm[i_shared * n_var + l];
      vm[2] = Fm[(i_shared + 1) * n_var + l];
      vm[3] = Fm[(i_shared + 2) * n_var + l];
      vm[4] = Fm[(i_shared + 3) * n_var + l];

      fci[l] = WENO5(vp, vm, eps_here, if_shock);
    } else if (param->inviscid_scheme == 71) {
      real vp[7], vm[7];
      vp[0] = Fp[(i_shared - 3) * n_var + l];
      vp[1] = Fp[(i_shared - 2) * n_var + l];
      vp[2] = Fp[(i_shared - 1) * n_var + l];
      vp[3] = Fp[i_shared * n_var + l];
      vp[4] = Fp[(i_shared + 1) * n_var + l];
      vp[5] = Fp[(i_shared + 2) * n_var + l];
      vp[6] = Fp[(i_shared + 3) * n_var + l];
      vm[0] = Fm[(i_shared - 2) * n_var + l];
      vm[1] = Fm[(i_shared - 1) * n_var + l];
      vm[2] = Fm[i_shared * n_var + l];
      vm[3] = Fm[(i_shared + 1) * n_var + l];
      vm[4] = Fm[(i_shared + 2) * n_var + l];
      vm[5] = Fm[(i_shared + 3) * n_var + l];
      vm[6] = Fm[(i_shared + 4) * n_var + l];

      fci[l] = WENO7(vp, vm, eps_here, if_shock);
    }
  }
}

__device__ void compute_weno_flux_cp(DParameter *param, const real *metric, const real *jac, real *fci, int i_shared,
  const real *Fp, const real *Fm, bool if_shock) {
  const int n_var = param->n_var;

  //  const real eps_ref = 1e-6 * param->weno_eps_scale;
  constexpr real eps{1e-8};
  const real eps_ref = eps * param->weno_eps_scale;
  // * 0.25 *
  // ((metric[i_shared * 3] * jac[i_shared] + metric[(i_shared + 1) * 3] * jac[i_shared + 1]) *
  // (metric[i_shared * 3] * jac[i_shared] + metric[(i_shared + 1) * 3] * jac[i_shared + 1]) +
  // (metric[i_shared * 3 + 1] * jac[i_shared] + metric[(i_shared + 1) * 3 + 1] * jac[i_shared + 1])
  // *
  // (metric[i_shared * 3 + 1] * jac[i_shared] + metric[(i_shared + 1) * 3 + 1] * jac[i_shared + 1])
  // +
  // (metric[i_shared * 3 + 2] * jac[i_shared] + metric[(i_shared + 1) * 3 + 2] * jac[i_shared + 1])
  // *
  // (metric[i_shared * 3 + 2] * jac[i_shared] + metric[(i_shared + 1) * 3 + 2] * jac[
  // i_shared + 1]));
  real eps_scaled[3];
  eps_scaled[0] = eps_ref;
  eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
  eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

  for (int l = 0; l < n_var; ++l) {
    real eps_here{eps_scaled[0]};
    if (l == 1 || l == 2 || l == 3) {
      eps_here = eps_scaled[1];
    } else if (l == 4) {
      eps_here = eps_scaled[2];
    }

    if (param->inviscid_scheme == 51) {
      real vp[5], vm[5];
      vp[0] = Fp[(i_shared - 2) * n_var + l];
      vp[1] = Fp[(i_shared - 1) * n_var + l];
      vp[2] = Fp[i_shared * n_var + l];
      vp[3] = Fp[(i_shared + 1) * n_var + l];
      vp[4] = Fp[(i_shared + 2) * n_var + l];
      vm[0] = Fm[(i_shared - 1) * n_var + l];
      vm[1] = Fm[i_shared * n_var + l];
      vm[2] = Fm[(i_shared + 1) * n_var + l];
      vm[3] = Fm[(i_shared + 2) * n_var + l];
      vm[4] = Fm[(i_shared + 3) * n_var + l];

      fci[l] = WENO5(vp, vm, eps_here, if_shock);
    } else if (param->inviscid_scheme == 71) {
      real vp[7], vm[7];
      vp[0] = Fp[(i_shared - 3) * n_var + l];
      vp[1] = Fp[(i_shared - 2) * n_var + l];
      vp[2] = Fp[(i_shared - 1) * n_var + l];
      vp[3] = Fp[i_shared * n_var + l];
      vp[4] = Fp[(i_shared + 1) * n_var + l];
      vp[5] = Fp[(i_shared + 2) * n_var + l];
      vp[6] = Fp[(i_shared + 3) * n_var + l];
      vm[0] = Fm[(i_shared - 2) * n_var + l];
      vm[1] = Fm[(i_shared - 1) * n_var + l];
      vm[2] = Fm[i_shared * n_var + l];
      vm[3] = Fm[(i_shared + 1) * n_var + l];
      vm[4] = Fm[(i_shared + 2) * n_var + l];
      vm[5] = Fm[(i_shared + 3) * n_var + l];
      vm[6] = Fm[(i_shared + 4) * n_var + l];

      fci[l] = WENO7(vp, vm, eps_here, if_shock);
    }
  }
}

__device__ void
positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
  real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac) {
  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

  const int ns = n_var - 5;
  const int offset_yq_l = i_shared * (n_var + 2) + 5;
  const int offset_yq_r = (i_shared + 1) * (n_var + 2) + 5;
  real *fc_yq_i = &fc[tid * n_var + 5];

  for (int l = 0; l < ns; ++l) {
    real theta_p = 1.0, theta_m = 1.0;
    // if (idx_in_mesh > -1) {
    const real up = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * fc_yq_i[l];
    if (up < 0) {
      const real up_lf = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * f_1st[tid * ns + l];
      if (abs(up - up_lf) > 1e-20) {
        theta_p = (0 - up_lf) / (up - up_lf);
        if (theta_p > 1)
          theta_p = 1.0;
        else if (theta_p < 0)
          theta_p = 0;
      }
    }
    // }

    // if (idx_in_mesh < max_extent - 1) {
    const real um = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * fc_yq_i[l];
    if (um < 0) {
      const real um_lf = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * f_1st[tid * ns + l];
      if (abs(um - um_lf) > 1e-20) {
        theta_m = (0 - um_lf) / (um - um_lf);
        if (theta_m > 1)
          theta_m = 1.0;
        else if (theta_m < 0)
          theta_m = 0;
      }
    }
    // }

    fc_yq_i[l] = min(theta_p, theta_m) * (fc_yq_i[l] - f_1st[tid * ns + l]) + f_1st[tid * ns + l];
  }
}

__device__ void positive_preserving_limiter_1(int dim, int n_var, const real *cv, int i_shared, const real *jac,
  real dt, real *fci, const real *metric, const real *cc, const real *Fp) {
  const real alpha = dim == 3 ? 1.0 / 3.0 : 0.5;
  const auto *cvl = &cv[i_shared * n_var];
  const auto *cvr = &cv[(i_shared + 1) * n_var];

  for (int l = 0; l < n_var - 5; ++l) {
    real f1{0.0};
    bool f1_computed{false};
    real theta_p = 1.0, theta_m = 1.0;
    const real up = 0.5 * alpha * cvl[l + 5] * jac[i_shared] - dt * fci[l + 5];
    if (up < 0) {
      const real temp1 = abs((metric[i_shared * 3] * cvl[1] +
                              metric[i_shared * 3 + 1] * cvl[2] +
                              metric[i_shared * 3 + 2] * cvl[3]) / cvl[0] +
                             cc[i_shared] * norm3d(metric[i_shared * 3], metric[i_shared * 3 + 1],
                                                   metric[i_shared * 3 + 2])); // spectralRadThis
      const real temp2 = abs((metric[(i_shared + 1) * 3] * cvr[1] +
                              metric[(i_shared + 1) * 3 + 1] * cvr[2] +
                              metric[(i_shared + 1) * 3 + 2] * cvr[3]) / cvr[0] +
                             cc[i_shared + 1] * norm3d(metric[(i_shared + 1) * 3], metric[(i_shared + 1) * 3 + 1],
                                                       metric[(i_shared + 1) * 3 + 2])); // spectralRadNext
      f1 = 0.5 * (Fp[i_shared * n_var + l + 5] + temp1 * cvl[l + 5] * jac[i_shared]) +
           0.5 *
           (Fp[(i_shared + 1) * n_var + l + 5] - temp2 * cvr[l + 5] * jac[i_shared + 1]);
      f1_computed = true;
      const real up_lf = 0.5 * alpha * cvl[l + 5] * jac[i_shared] - dt * f1;
      if (abs(up - up_lf) > 1e-20) {
        theta_p = (0 - up_lf) / (up - up_lf);
        if (theta_p > 1)
          theta_p = 1.0;
        else if (theta_p < 0)
          theta_p = 0;
      }
    }

    const real um =
        0.5 * alpha * cvr[l + 5] * jac[i_shared + 1] + dt * fci[l + 5];
    if (um < 0) {
      if (!f1_computed) {
        real temp1 = abs((metric[i_shared * 3] * cvl[1] +
                          metric[i_shared * 3 + 1] * cvl[2] +
                          metric[i_shared * 3 + 2] * cvl[3]) / cvl[0] +
                         cc[i_shared] * norm3d(metric[i_shared * 3], metric[i_shared * 3 + 1],
                                               metric[i_shared * 3 + 2])); // spectralRadThis
        real temp2 = abs((metric[(i_shared + 1) * 3] * cvr[1] +
                          metric[(i_shared + 1) * 3 + 1] * cvr[2] +
                          metric[(i_shared + 1) * 3 + 2] * cvr[3]) / cvr[0] +
                         cc[i_shared + 1] * norm3d(metric[(i_shared + 1) * 3], metric[(i_shared + 1) * 3 + 1],
                                                   metric[(i_shared + 1) * 3 + 2])); // spectralRadNext
        f1 = 0.5 * (Fp[i_shared * n_var + l + 5] + temp1 * cvl[l + 5] * jac[i_shared]) +
             0.5 *
             (Fp[(i_shared + 1) * n_var + l + 5] - temp2 * cvr[l + 5] * jac[i_shared + 1]);
      }
      const real um_lf =
          0.5 * alpha * cvr[l + 5] * jac[i_shared + 1] + dt * f1;
      if (abs(um - um_lf) > 1e-20) {
        theta_m = (0 - um_lf) / (um - um_lf);
        if (theta_m > 1)
          theta_m = 1.0;
        else if (theta_m < 0)
          theta_m = 0;
      }
    }

    fci[l + 5] = min(theta_p, theta_m) * (fci[l + 5] - f1) + f1;
  }
}

__device__ real WENO5(const real *vp, const real *vm, real eps, bool if_shock) {
  if (if_shock) {
    constexpr real one6th{1.0 / 6};
    real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
    real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
    real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
    constexpr real thirteen12th{13.0 / 12};
    real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
                 0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
    real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
                 0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
    real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
                 0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
    constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
    // real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
    // real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
    // real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
    // real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
    real a0{three10th / ((eps + beta0) * (eps + beta0))};
    real a1{six10th / ((eps + beta1) * (eps + beta1))};
    real a2{one10th / ((eps + beta2) * (eps + beta2))};
    const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
    v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
    v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
    beta0 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
            0.25 * (3 * vm[2] - 4 * vm[3] + vm[4]) * (3 * vm[2] - 4 * vm[3] + vm[4]);
    beta1 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
            0.25 * (vm[1] - vm[3]) * (vm[1] - vm[3]);
    beta2 = thirteen12th * (vm[0] + vm[2] - 2 * vm[1]) * (vm[0] + vm[2] - 2 * vm[1]) +
            0.25 * (vm[0] - 4 * vm[1] + 3 * vm[2]) * (vm[0] - 4 * vm[1] + 3 * vm[2]);
    // tau5sqr = (beta0 - beta2) * (beta0 - beta2);
    // a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
    // a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
    // a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
    a0 = one10th / ((eps + beta0) * (eps + beta0));
    a1 = six10th / ((eps + beta1) * (eps + beta1));
    a2 = three10th / ((eps + beta2) * (eps + beta2));
    const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    return fPlus + fMinus;
  }
  constexpr real one6th{1.0 / 6};
  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
  const real fPlus{0.3 * v0 + 0.6 * v1 + 0.1 * v2};

  v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
  v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
  v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
  const real fMinus{0.1 * v0 + 0.6 * v1 + 0.3 * v2};

  return fPlus + fMinus;
}

__device__ real WENO7(const real *vp, const real *vm, real eps, bool if_shock) {
  if (if_shock) {
    // Shocked, use WENO
    constexpr real one6th{1.0 / 6};
    constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};

    // Re-organize the data to improve locality
    // 1st order derivative
    real s1{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
    // 2nd order derivative
    real s2{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
    // 3rd order derivative
    real s3{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
    real beta0{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4]);
    s2 = vp[2] - 2 * vp[3] + vp[4];
    s3 = -vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4];
    real beta1{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5]);
    s3 = -vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5];
    real beta2{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6]);
    s2 = 2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6];
    s3 = -vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6];
    real beta3{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    // real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
    constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
    // real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
    // real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
    // real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
    // real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};
    real a0 = c0 / ((eps + beta0) * (eps + beta0));
    real a1 = c1 / ((eps + beta1) * (eps + beta1));
    real a2 = c2 / ((eps + beta2) * (eps + beta2));
    real a3 = c3 / ((eps + beta3) * (eps + beta3));
    // real a0 = Ip < 1 ? c0 / ((eps + beta0) * (eps + beta0)) : 0;
    // real a1 = Ip < 2 ? c1 / ((eps + beta1) * (eps + beta1)) : 0;
    // real a2 = (Ip < 3 && Im < 3) ? c2 / ((eps + beta2) * (eps + beta2)) : 0;
    // real a3 = Im < 2 ? c3 / ((eps + beta3) * (eps + beta3)) : 0;

    constexpr real one12th{1.0 / 12};
    real v0{-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3]};
    real v1{vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4]};
    real v2{-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5]};
    real v3{3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6]};
    const real fPlus{one12th * (a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

    // Minus part
    s1 = one6th * (-2 * vm[6] + 9 * vm[5] - 18 * vm[4] + 11 * vm[3]);
    s2 = -vm[6] + 4 * vm[5] - 5 * vm[4] + 2 * vm[3];
    s3 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
    beta0 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (vm[5] - 6 * vm[4] + 3 * vm[3] + 2 * vm[2]);
    s2 = vm[4] - 2 * vm[3] + vm[2];
    s3 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
    beta1 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (-2 * vm[4] - 3 * vm[3] + 6 * vm[2] - vm[1]);
    s3 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
    beta2 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (-11 * vm[3] + 18 * vm[2] - 9 * vm[1] + 2 * vm[0]);
    s2 = 2 * vm[3] - 5 * vm[2] + 4 * vm[1] - vm[0];
    s3 = -vm[3] + 3 * vm[2] - 3 * vm[1] + vm[0];
    beta3 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    // tau7sqr = (beta0 - beta3) * (beta0 - beta3);
    // a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
    // a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
    // a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
    // a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));
    a0 = c0 / ((eps + beta0) * (eps + beta0));
    a1 = c1 / ((eps + beta1) * (eps + beta1));
    a2 = c2 / ((eps + beta2) * (eps + beta2));
    a3 = c3 / ((eps + beta3) * (eps + beta3));
    // a0 = Im < 1 ? c0 / ((eps + beta0) * (eps + beta0)) : 0;
    // a1 = Im < 2 ? c1 / ((eps + beta1) * (eps + beta1)) : 0;
    // a2 = (Ip < 3 && Im < 3) ? c2 / ((eps + beta2) * (eps + beta2)) : 0;
    // a3 = Ip < 2 ? c3 / ((eps + beta3) * (eps + beta3)) : 0;

    v0 = -3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3];
    v1 = vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2];
    v2 = -vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1];
    v3 = 3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0];
    const real fMinus{one12th * (a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

    return fPlus + fMinus;
  }
  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
  constexpr real one12th{1.0 / 12};
  real v3{0}, v2{0}, v1{0}, v0{0};
  v3 = 3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6];
  v2 = -vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5];
  v1 = vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4];
  v0 = -3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3];
  const real fPlus{one12th * (c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3)};

  // Minus part
  v0 = -3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3];
  v1 = vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2];
  v2 = -vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1];
  v3 = 3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0];
  const real fMinus{one12th * (c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3)};

  return fPlus + fMinus;
}

__device__ real WENO(const real *vp, const real *vm, real eps, bool if_shock, int weno_scheme_i) {
  if (weno_scheme_i == 1) {
    // WENO-1
    return vp[0] + vm[0];
  }
  if (weno_scheme_i == 2) {
    // WENO-3
    if (if_shock) {
      real beta0 = (vp[1] - vp[0]) * (vp[1] - vp[0]);
      real beta1 = (vp[2] - vp[1]) * (vp[2] - vp[1]);
      real a0 = 1.0 / 3.0 / ((eps + beta0) * (eps + beta0));
      real a1 = 2.0 / 3.0 / ((eps + beta1) * (eps + beta1));
      real v0 = 0.5 * (-vp[0] + 3 * vp[1]);
      real v1 = 0.5 * (vp[1] + vp[2]);
      const real fPlus = (a0 * v0 + a1 * v1) / (a0 + a1);

      beta0 = (vm[2] - vm[1]) * (vm[2] - vm[1]);
      beta1 = (vm[3] - vm[2]) * (vm[3] - vm[2]);
      a0 = 1.0 / 3.0 / ((eps + beta0) * (eps + beta0));
      a1 = 2.0 / 3.0 / ((eps + beta1) * (eps + beta1));
      v0 = 0.5 * (-vm[2] + 3 * vm[1]);
      v1 = 0.5 * (vm[0] + vm[1]);
      const real fMinus = (a0 * v0 + a1 * v1) / (a0 + a1);
      return fPlus + fMinus;
    } else {
      real v0 = 0.5 * (-vp[0] + 3 * vp[1]);
      real v1 = 0.5 * (vp[1] + vp[2]);
      const real fPlus = (v0 + 2 * v1) / 3.0;

      v0 = 0.5 * (-vm[2] + 3 * vm[1]);
      v1 = 0.5 * (vm[0] + vm[1]);
      const real fMinus = (v0 + 2 * v1) / 3.0;
      return fPlus + fMinus;
    }
  }
  if (weno_scheme_i == 3) {
    // WENO-5
    return WENO5(vp, vm, eps, if_shock);
  }
  if (weno_scheme_i == 4) {
    // WENO-7
    return WENO7(vp, vm, eps, if_shock);
  }
  // Default to WENO-5
  return WENO5(vp, vm, eps, if_shock);
}

__device__ real WENO5_new(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
  constexpr real thirteen12th{13.0 / 12};
  real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
               0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
  real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
               0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
  real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
               0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
  constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
  real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
  real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
  real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
  real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  v0 = one6th * (11 * vm[3] - 7 * vm[4] + 2 * vm[5]);
  v1 = one6th * (2 * vm[3] + 5 * vm[3] - vm[4]);
  v2 = one6th * (-vm[1] + 5 * vm[2] + 2 * vm[3]);
  beta0 = thirteen12th * (vm[3] + vm[5] - 2 * vm[4]) * (vm[3] + vm[5] - 2 * vm[4]) +
          0.25 * (3 * vm[3] - 4 * vm[4] + vm[5]) * (3 * vm[3] - 4 * vm[4] + vm[5]);
  beta1 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
          0.25 * (vm[2] - vm[4]) * (vm[2] - vm[4]);
  beta2 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
          0.25 * (vm[1] - 4 * vm[2] + 3 * vm[3]) * (vm[1] - 4 * vm[2] + 3 * vm[3]);
  tau5sqr = (beta0 - beta2) * (beta0 - beta2);
  a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
  a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
  a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  return fPlus + fMinus;
}

__device__ real WENO7_new(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};

  // Re-organize the data to improve locality
  // 1st order derivative
  real s1{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
  // 2nd order derivative
  real s2{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
  // 3rd order derivative
  real s3{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
  real beta0{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4]);
  s2 = vp[2] - 2 * vp[3] + vp[4];
  s3 = -vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4];
  real beta1{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5]);
  s3 = -vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5];
  real beta2{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6]);
  s2 = 2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6];
  s3 = -vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6];
  real beta3{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
  real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
  real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
  real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
  real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};

  constexpr real one12th{1.0 / 12};
  real v0{one12th * (-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3])};
  real v1{one12th * (vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4])};
  real v2{one12th * (-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5])};
  real v3{one12th * (3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6])};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  // Minus part
  s1 = one6th * (-2 * vm[7] + 9 * vm[6] - 18 * vm[5] + 11 * vm[4]);
  s2 = -vm[7] + 4 * vm[6] - 5 * vm[5] + 2 * vm[4];
  s3 = -vm[7] + 3 * vm[6] - 3 * vm[5] + vm[4];
  beta0 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (vm[6] - 6 * vm[5] + 3 * vm[4] + 2 * vm[3]);
  s2 = vm[5] - 2 * vm[4] + vm[3];
  s3 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
  beta1 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-2 * vm[5] - 3 * vm[4] + 6 * vm[3] - vm[2]);
  s3 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
  beta2 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-11 * vm[4] + 18 * vm[3] - 9 * vm[2] + 2 * vm[1]);
  s2 = 2 * vm[4] - 5 * vm[3] + 4 * vm[2] - vm[1];
  s3 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
  beta3 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  tau7sqr = (beta0 - beta3) * (beta0 - beta3);
  a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
  a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
  a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
  a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));

  v0 = one12th * (-3 * vm[7] + 13 * vm[6] - 23 * vm[5] + 25 * vm[4]);
  v1 = one12th * (vm[6] - 5 * vm[5] + 13 * vm[4] + 3 * vm[3]);
  v2 = one12th * (-vm[5] + 7 * vm[4] + 7 * vm[3] - vm[2]);
  v3 = one12th * (3 * vm[4] + 13 * vm[3] - 5 * vm[2] + vm[1]);
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  return fPlus + fMinus;
}

template void
compute_convective_term_weno<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);
}

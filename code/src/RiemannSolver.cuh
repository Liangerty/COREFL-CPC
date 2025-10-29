#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "Constants.h"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model>
__device__ void
riemannSolver_ausmPlus(const real *pv_l, const real *pv_r, DParameter *param, int tid, const real *metric,
                       const real *jac, real *fc, int i_shared) {
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

template<MixtureModel mix_model>
__device__ void
riemannSolver_hllc(const real *pv_l, const real *pv_r, DParameter *param, int tid, const real *metric, const real *jac,
                   real *fc, int i_shared) {
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

template<MixtureModel mixtureModel>
__device__ void
compute_half_sum_left_right_flux(const real *pv_l, const real *pv_r, DParameter *param, const real *jac,
                                 const real *metric, int i_shared, real *fc) {
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

template<MixtureModel mix_model>
__device__ void
riemannSolver_Roe(DZone *zone, real *pv, int tid, DParameter *param, real *fc, real *metric, const real *jac,
                  const real *entropy_fix_delta) {
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

template<MixtureModel mix_model>
__device__ void
riemannSolver_laxFriedrich(const real *pv_l, const real *pv_r, DParameter *param, int tid, const real *metric,
                           const real *jac, real *fc, int i_shared) {
  printf("LF flux for mixture is not implemented yet. Please use AUSM+ or Roe instead.\n");
}

template<>
__device__ void
riemannSolver_laxFriedrich<MixtureModel::Air>(const real *pv_l, const real *pv_r, DParameter *param, int tid,
                                              const real *metric, const real *jac, real *fc, int i_shared) {
  const int n_var = param->n_var;
  const int n_reconstruct{n_var};

  // The metrics are just the average of the two adjacent cells.
  const real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  const real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  const real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);

  // compute the left and right contravariance velocity
  const real Ukl{pv_l[1] * kx + pv_l[2] * ky + pv_l[3] * kz};
  const real Ukr{pv_r[1] * kx + pv_r[2] * ky + pv_r[3] * kz};
  const real cl{std::sqrt(gamma_air * pv_l[4] / pv_l[0])};
  const real cr{std::sqrt(gamma_air * pv_r[4] / pv_r[0])};
  const real spectral_radius{max(std::abs(Ukl) + cl * gradK, std::abs(Ukr) + cr * gradK)};

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
}

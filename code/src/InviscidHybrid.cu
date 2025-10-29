#include "InviscidScheme.cuh"
#include "Field.h"
#include "DParameter.cuh"
#include "Constants.h"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model> __device__ void hybrid_weno_part(const real *pv, const real *rhoE, int i_shared,
  const DParameter *param, const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci,
  real *f_1st) {
  const int n_var = param->n_var;
  // compute the roe average
  const auto *pvl = &pv[i_shared * n_var], *pvr = &pv[(i_shared + 1) * n_var];
  const real rlc = sqrt(pvl[0]) / (sqrt(pvl[0]) + sqrt(pvr[0]));
  const real rrc = 1.0 - rlc;
  const real um = rlc * pvl[1] + rrc * pvr[1];
  const real vm = rlc * pvl[2] + rrc * pvr[2];
  const real wm = rlc * pvl[3] + rrc * pvr[3];
  const real ekm = 0.5 * (um * um + vm * vm + wm * wm);
  const real hl = (pvl[4] + rhoE[i_shared]) / pvl[0];
  const real hr = (pvr[4] + rhoE[i_shared + 1]) / pvr[0];
  const real hm = rlc * hl + rrc * hr;

  real svm[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER] = {};
  for (int l = 0; l < param->n_scalar; ++l) {
    svm[l] = rlc * pvl[5 + l] + rrc * pvr[5 + l];
  }
  const int n_spec{param->n_spec};
  real mw_inv = 0;
  for (int l = 0; l < n_spec; ++l) {
    mw_inv += svm[l] * param->imw[l];
  }

  const real tl{pvl[4] / pvl[0]}, tr{pvr[4] / pvr[0]};
  const real tm = (rlc * tl + rrc * tr) / (R_u * mw_inv);

  real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
  compute_enthalpy_and_cp(tm, h_i, cp_i, param);
  // compute_enthalpy_and_cp_1(tm, h_i, cp_i, param);
  real cp{0}, cv_tot{0};
  for (int l = 0; l < n_spec; ++l) {
    cp += svm[l] * cp_i[l];
    cv_tot += svm[l] * (cp_i[l] - param->gas_const[l]);
  }
  const real gamma = cp / cv_tot;
  const real cm = sqrt(gamma * R_u * mw_inv * tm);
  const real gm1{gamma - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
  real kxJ{metric[i_shared * 3] * jac_l + metric[(i_shared + 1) * 3] * jac_r};
  real kyJ{metric[i_shared * 3 + 1] * jac_l + metric[(i_shared + 1) * 3 + 1] * jac_r};
  real kzJ{metric[i_shared * 3 + 2] * jac_l + metric[(i_shared + 1) * 3 + 2] * jac_r};
  real kx{kxJ / (jac_l + jac_r)};
  real ky{kyJ / (jac_l + jac_r)};
  real kz{kzJ / (jac_l + jac_r)};
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  const real cm2_inv{1.0 / (cm * cm)};
  // Compute the characteristic flux with L.
  real fChar[5 + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER];
  constexpr real eps{1e-40};
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);

  real alpha_l[MAX_SPEC_NUMBER];
  // compute the partial derivative of pressure to species density
  for (int l = 0; l < n_spec; ++l) {
    alpha_l[l] = gamma * param->gas_const[l] * tm - (gamma - 1) * h_i[l];
    // The computations including this alpha_l are all combined with a division by cm2.
    alpha_l[l] *= cm2_inv;
  }

  constexpr int max_weno_size = 8; // For WENO7
  int weno_scheme = 3, weno_size = 6;
  if (param->inviscid_scheme == 71 || param->inviscid_scheme == 72) {
    weno_scheme = 4;
    weno_size = 8;
  }
  real spec_rad[3]{}, specRadThis, specRadNext;
  bool pp_limiter{param->positive_preserving};
  for (int m = 0; m < weno_size; m++) {
    const int is = i_shared + m - weno_scheme + 1;
    const real Uk = uk[is];
    const real UkPc = abs(Uk + cGradK[is]);
    const real UkMc = abs(Uk - cGradK[is]);
    spec_rad[0] = max(spec_rad[0], UkMc);
    spec_rad[1] = max(spec_rad[1], abs(Uk));
    spec_rad[2] = max(spec_rad[2], UkPc);
    if (pp_limiter && is == i_shared)
      specRadThis = UkPc;
    if (pp_limiter && is == i_shared + 1)
      specRadNext = UkPc;
  }
  if (pp_limiter) {
    for (int l = 0; l < n_var - 5; l++) { // f_1st[l - 5] = 0.5 * (vPlus[weno_scheme - 1] + vMinus[weno_scheme]);
      f_1st[l] = 0.5 * jac[i_shared] * pv[i_shared * n_var] * pv[i_shared * n_var + 5 + l] * (
                   uk[i_shared] + specRadThis)
                 + 0.5 * jac[i_shared + 1] * pv[(i_shared + 1) * n_var] * pv[(i_shared + 1) * n_var + 5 + l] * (
                   uk[i_shared + 1] - specRadNext);
    }
  }

  for (int l = 0; l < 5; ++l) {
    real coeff_alpha_s{0.5}, lambda{spec_rad[1]};
    real L[5];
    switch (l) {
      case 0:
        lambda = spec_rad[0];
        L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
        L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
        L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
        L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
        L[4] = gm1 * cm2_inv * 0.5;
        break;
      case 1:
        coeff_alpha_s = -kx;
        L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
        L[1] = kx * gm1 * um * cm2_inv;
        L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
        L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
        L[4] = -kx * gm1 * cm2_inv;
        break;
      case 2:
        coeff_alpha_s = -ky;
        L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
        L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
        L[2] = ky * gm1 * vm * cm2_inv;
        L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
        L[4] = -ky * gm1 * cm2_inv;
        break;
      case 3:
        coeff_alpha_s = -kz;
        L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
        L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
        L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
        L[3] = kz * gm1 * wm * cm2_inv;
        L[4] = -kz * gm1 * cm2_inv;
        break;
      case 4:
        lambda = spec_rad[2];
        L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
        L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
        L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
        L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
        L[4] = gm1 * cm2_inv * 0.5;
        break;
      default:
        break;
    }

    real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
    for (int m = 0; m < weno_size; ++m) {
      const int is = i_shared + m - weno_scheme + 1;
      const auto *bv = &pv[is * n_var];
      real Uk = uk[is];
      // const real lambda = abs(Uk) + cGradK[is];
      real F[5]; //  + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER
      F[0] = Uk * bv[0];
      F[1] = Uk * bv[1] * bv[0] + bv[4] * metric[is * 3];
      F[2] = Uk * bv[2] * bv[0] + bv[4] * metric[is * 3 + 1];
      F[3] = Uk * bv[3] * bv[0] + bv[4] * metric[is * 3 + 2];
      F[4] = Uk * (rhoE[is] + bv[4]);

      vPlus[m] = L[0] * (F[0] + lambda * bv[0]) + L[1] * (F[1] + lambda * bv[1] * bv[0]) +
                 L[2] * (F[2] + lambda * bv[2] * bv[0]) + L[3] * (F[3] + lambda * bv[3] * bv[0]) +
                 L[4] * (F[4] + lambda * rhoE[is]);
      vMinus[m] = L[0] * (F[0] - lambda * bv[0]) + L[1] * (F[1] - lambda * bv[1] * bv[0]) +
                  L[2] * (F[2] - lambda * bv[2] * bv[0]) + L[3] * (F[3] - lambda * bv[3] * bv[0]) +
                  L[4] * (F[4] - lambda * rhoE[is]);
      for (int n = 0; n < n_spec; ++n) {
        vPlus[m] += coeff_alpha_s * alpha_l[n] * bv[0] * bv[5 + n] * (Uk + lambda);
        vMinus[m] += coeff_alpha_s * alpha_l[n] * bv[0] * bv[5 + n] * (Uk - lambda);
      }
      vPlus[m] *= 0.5 * jac[is];
      vMinus[m] *= 0.5 * jac[is];
    }
    if (weno_size == 6)
      fChar[l] = WENO5_new(vPlus, vMinus, eps_scaled);
    else
      fChar[l] = WENO7_new(vPlus, vMinus, eps_scaled);
  }
  for (int l = 0; l < n_spec; ++l) {
    real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
    for (int m = 0; m < weno_size; ++m) {
      const int is = i_shared + m - weno_scheme + 1;
      const auto *bv = &pv[is * n_var];
      const real Uk = uk[is];
      const real lambda = spec_rad[1];
      vPlus[m] = 0.5 * jac[is] * bv[0] * (-svm[l] + bv[5 + l]) * (Uk + lambda);
      vMinus[m] = 0.5 * jac[is] * bv[0] * (-svm[l] + bv[5 + l]) * (Uk - lambda);
    }
    if (weno_size == 6)
      fChar[5 + l] = WENO5_new(vPlus, vMinus, eps_scaled);
    else
      fChar[5 + l] = WENO7_new(vPlus, vMinus, eps_scaled);
  }
  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  fci[0] = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  fci[1] = (um - kx * cm) * fChar[0] + kx * um * fChar[1] + (ky * um - kz * cm) * fChar[2] +
           (kz * um + ky * cm) * fChar[3] + (um + kx * cm) * fChar[4];
  fci[2] = (vm - ky * cm) * fChar[0] + (kx * vm + kz * cm) * fChar[1] + ky * vm * fChar[2] +
           (kz * vm - kx * cm) * fChar[3] + (vm + ky * cm) * fChar[4];
  fci[3] = (wm - kz * cm) * fChar[0] + (kx * wm - ky * cm) * fChar[1] + (ky * wm + kx * cm) * fChar[2] +
           kz * wm * fChar[3] + (wm + kz * cm) * fChar[4];

  fci[4] = (hm - Uk_bar * cm) * fChar[0] + (kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm) * fChar[1] +
           (ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm) * fChar[2] +
           (kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm) * fChar[3] +
           (hm + Uk_bar * cm) * fChar[4];
  real add{0};
  const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  for (int l = 0; l < n_spec; ++l) {
    fci[5 + l] = svm[l] * coeff_add + fChar[l + 5];
    add += alpha_l[l] * fChar[l + 5];
  }
  fci[4] -= add * cm * cm / gm1;
}

// template<> __device__ void hybrid_weno_part<MixtureModel::Air>(const real *pv, const real *rhoE, int i_shared,
//   const DParameter *param, const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci) {
//   const int n_var = param->n_var;
//   // compute the roe average
//   const auto *pvl = &pv[i_shared * n_var], *pvr = &pv[(i_shared + 1) * n_var];
//   const real rlc = sqrt(pvl[0]) / (sqrt(pvl[0]) + sqrt(pvr[0]));
//   const real rrc = 1.0 - rlc;
//   const real um = rlc * pvl[1] + rrc * pvr[1];
//   const real vm = rlc * pvl[2] + rrc * pvr[2];
//   const real wm = rlc * pvl[3] + rrc * pvr[3];
//   const real hm = rlc * (pvl[4] + rhoE[i_shared]) / pvl[0] + rrc * (pvr[4] + rhoE[i_shared + 1]) / pvr[0];
//   constexpr real gm1{gamma_air - 1};
//   const real cm{sqrt(gm1 * (hm - 0.5 * (um * um + vm * vm + wm * wm)))};

//   // Next, we compute the left characteristic matrix at i+1/2.
//   const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
//   const real kxJ{metric[i_shared * 3] * jac_l + metric[(i_shared + 1) * 3] * jac_r};
//   const real kyJ{metric[i_shared * 3 + 1] * jac_l + metric[(i_shared + 1) * 3 + 1] * jac_r};
//   const real kzJ{metric[i_shared * 3 + 2] * jac_l + metric[(i_shared + 1) * 3 + 2] * jac_r};
//   real kx{kxJ / (jac_l + jac_r)};
//   real ky{kyJ / (jac_l + jac_r)};
//   real kz{kzJ / (jac_l + jac_r)};
//   const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
//   kx /= gradK;
//   ky /= gradK;
//   kz /= gradK;
//   const real Uk_bar{kx * um + ky * vm + kz * wm};
//   const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};

//   // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
//   const real cm2_inv{1.0 / (cm * cm)};
//   // Compute the characteristic flux with L.
//   real fChar[5];
//   constexpr real eps{1e-40};
//   const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);

//   constexpr int max_weno_size = 8; // For WENO7
//   int weno_scheme = 3, weno_size = 6;
//   if (param->inviscid_scheme == 71 || param->inviscid_scheme == 72) {
//     weno_scheme = 4;
//     weno_size = 8;
//   }

//   for (int l = 0; l < 5; ++l) {
//     real L[5];
//     switch (l) {
//       case 0:
//         L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
//         L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
//         L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
//         L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
//         L[4] = gm1 * cm2_inv * 0.5;
//         break;
//       case 1:
//         L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
//         L[1] = kx * gm1 * um * cm2_inv;
//         L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
//         L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
//         L[4] = -kx * gm1 * cm2_inv;
//         break;
//       case 2:
//         L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
//         L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
//         L[2] = ky * gm1 * vm * cm2_inv;
//         L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
//         L[4] = -ky * gm1 * cm2_inv;
//         break;
//       case 3:
//         L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
//         L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
//         L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
//         L[3] = kz * gm1 * wm * cm2_inv;
//         L[4] = -kz * gm1 * cm2_inv;
//         break;
//       case 4:
//         L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
//         L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
//         L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
//         L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
//         L[4] = gm1 * cm2_inv * 0.5;
//         break;
//       default:
//         break;
//     }

//     real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
//     for (int m = 0; m < weno_size; ++m) {
//       const int is = i_shared + m - weno_scheme + 1;
//       const auto *bv = &pv[is * n_var];
//       const real Uk = uk[is];
//       const real lambda = abs(Uk) + cGradK[is];
//       real F[5];
//       F[0] = Uk * bv[0];
//       F[1] = Uk * bv[1] * bv[0] + bv[4] * metric[is * 3];
//       F[2] = Uk * bv[2] * bv[0] + bv[4] * metric[is * 3 + 1];
//       F[3] = Uk * bv[3] * bv[0] + bv[4] * metric[is * 3 + 2];
//       F[4] = Uk * (rhoE[is] + bv[4]);

//       vPlus[m] = L[0] * (F[0] + lambda * bv[0]) + L[1] * (F[1] + lambda * bv[1] * bv[0]) +
//                  L[2] * (F[2] + lambda * bv[2] * bv[0]) + L[3] * (F[3] + lambda * bv[3] * bv[0]) +
//                  L[4] * (F[4] + lambda * rhoE[is]);
//       vMinus[m] = L[0] * (F[0] - lambda * bv[0]) + L[1] * (F[1] - lambda * bv[1] * bv[0]) +
//                   L[2] * (F[2] - lambda * bv[2] * bv[0]) + L[3] * (F[3] - lambda * bv[3] * bv[0]) +
//                   L[4] * (F[4] - lambda * rhoE[is]);
//       vPlus[m] *= 0.5 * jac[is];
//       vMinus[m] *= 0.5 * jac[is];
//     }
//     if (weno_size == 6)
//       fChar[l] = WENO5_new(vPlus, vMinus, eps_scaled);
//     else
//       fChar[l] = WENO7_new(vPlus, vMinus, eps_scaled);
//   }
//   for (int l = 0; l < param->n_scalar; ++l) {
//     // For passive scalars, just use the component form
//     real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
//     for (int m = 0; m < weno_size; ++m) {
//       const int is = i_shared + m - weno_scheme + 1;
//       const auto *bv = &pv[is * n_var];
//       const real Uk = uk[is];
//       const real lambda = abs(Uk) + cGradK[is];
//       vPlus[m] = 0.5 * jac[is] * bv[0] * bv[5 + l] * (Uk + lambda);
//       vMinus[m] = 0.5 * jac[is] * bv[0] * bv[5 + l] * (Uk - lambda);
//     }
//     if (weno_size == 5)
//       fci[5 + l] = WENO5_new(vPlus, vMinus, eps_scaled);
//     else
//       fci[5 + l] = WENO7_new(vPlus, vMinus, eps_scaled);
//   }
//   // Project the flux back to physical space
//   // We do not compute the right characteristic matrix here, because we explicitly write the components below.
//   fci[0] = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
//   fci[1] = (um - kx * cm) * fChar[0] + kx * um * fChar[1] + (ky * um - kz * cm) * fChar[2] +
//            (kz * um + ky * cm) * fChar[3] + (um + kx * cm) * fChar[4];
//   fci[2] = (vm - ky * cm) * fChar[0] + (kx * vm + kz * cm) * fChar[1] + ky * vm * fChar[2] +
//            (kz * vm - kx * cm) * fChar[3] + (vm + ky * cm) * fChar[4];
//   fci[3] = (wm - kz * cm) * fChar[0] + (kx * wm - ky * cm) * fChar[1] + (ky * wm + kx * cm) * fChar[2] +
//            kz * wm * fChar[3] + (wm + kz * cm) * fChar[4];

//   fci[4] = (hm - Uk_bar * cm) * fChar[0] + (kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm) * fChar[1] +
//            (ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm) * fChar[2] +
//            (kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm) * fChar[3] +
//            (hm + Uk_bar * cm) * fChar[4];
// }

__device__ void hybrid_weno_part_cp(const real *pv, const real *rhoE, int i_shared, const DParameter *param,
  const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci, real *f_1st) {
  const int n_var = param->n_var;

  constexpr real eps{1e-40};
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

  int weno_scheme = 3, weno_size = 6;
  if (param->inviscid_scheme == 71 || param->inviscid_scheme == 72) {
    weno_scheme = 4;
    weno_size = 8;
  }

  for (int l = 0; l < n_var; ++l) {
    constexpr int max_weno_size = 8;
    real eps_here{eps_scaled[0]};
    if (l == 1 || l == 2 || l == 3) {
      eps_here = eps_scaled[1];
    } else if (l == 4) {
      eps_here = eps_scaled[2];
    }

    real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
    for (int m = 0; m < weno_size; ++m) {
      const int is = i_shared + m - weno_scheme + 1;
      const auto *bv = &pv[is * n_var];
      const real Uk = uk[is];
      const real lambda = abs(Uk) + cGradK[is];

      real f, lam;
      switch (l) {
        case 0:
          f = Uk * bv[0];
          lam = lambda * bv[0];
          break;
        case 1:
          f = Uk * bv[1] * bv[0] + bv[4] * metric[is * 3];
          lam = lambda * bv[1] * bv[0];
          break;
        case 2:
          f = Uk * bv[2] * bv[0] + bv[4] * metric[is * 3 + 1];
          lam = lambda * bv[2] * bv[0];
          break;
        case 3:
          f = Uk * bv[3] * bv[0] + bv[4] * metric[is * 3 + 2];
          lam = lambda * bv[3] * bv[0];
          break;
        case 4:
          f = Uk * (rhoE[is] + bv[4]);
          lam = lambda * rhoE[is];
          break;
        default:
          f = Uk * bv[l] * bv[0];
          lam = lambda * bv[l] * bv[0];
      }
      vPlus[m] = 0.5 * jac[is] * (f + lam);
      vMinus[m] = 0.5 * jac[is] * (f - lam);
    }
    if (param->positive_preserving && l > 4) {
      f_1st[l - 5] = 0.5 * (vPlus[weno_scheme - 1] + vMinus[weno_scheme]);
    }
    if (weno_size == 6)
      fci[l] = WENO5_new(vPlus, vMinus, eps_here);
    else
      fci[l] = WENO7_new(vPlus, vMinus, eps_here);
  }
}

__device__ void hybrid_ud_part(const real *pv, const real *rhoE, int i_shared, const DParameter *param,
  const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci, real *f_1st) {
  const int n_var = param->n_var;

  int weno_scheme = 3, weno_size = 6;
  if (param->inviscid_scheme == 71 || param->inviscid_scheme == 72) {
    weno_scheme = 4;
    weno_size = 8;
  }

  for (int l = 0; l < n_var; ++l) {
    constexpr int max_weno_size = 8;

    real vPlus[max_weno_size] = {}, vMinus[max_weno_size] = {};
    for (int m = 0; m < weno_size; ++m) {
      const int is = i_shared + m - weno_scheme + 1;
      const auto *bv = &pv[is * n_var];
      const real Uk = uk[is];
      const real lambda = abs(Uk) + cGradK[is];

      real f, lam;
      switch (l) {
        case 0:
          f = Uk * bv[0];
          lam = lambda * bv[0];
          break;
        case 1:
          f = Uk * bv[1] * bv[0] + bv[4] * metric[is * 3];
          lam = lambda * bv[1] * bv[0];
          break;
        case 2:
          f = Uk * bv[2] * bv[0] + bv[4] * metric[is * 3 + 1];
          lam = lambda * bv[2] * bv[0];
          break;
        case 3:
          f = Uk * bv[3] * bv[0] + bv[4] * metric[is * 3 + 2];
          lam = lambda * bv[3] * bv[0];
          break;
        case 4:
          f = Uk * (rhoE[is] + bv[4]);
          lam = lambda * rhoE[is];
          break;
        default:
          f = Uk * bv[l] * bv[0];
          lam = lambda * bv[l] * bv[0];
      }
      vPlus[m] = 0.5 * jac[is] * (f + lam);
      vMinus[m] = 0.5 * jac[is] * (f - lam);
    }

    if (param->positive_preserving && l > 4) {
      f_1st[l - 5] = 0.5 * (vPlus[weno_scheme - 1] + vMinus[weno_scheme]);
    }

    if (weno_size == 6) {
      const real fP = 2 * vPlus[0] - 13 * vPlus[1] + 47 * vPlus[2] + 27 * vPlus[3] - 3 * vPlus[4];
      const real fM = -3 * vMinus[1] + 27 * vMinus[2] + 47 * vMinus[3] - 13 * vMinus[4] + 2 * vMinus[5];
      fci[l] = (fP + fM) / 60;
    } else {
      const real fP = -3 * vPlus[0] + 25 * vPlus[1] - 101 * vPlus[2] + 319 * vPlus[3] + 214 * vPlus[4] -
                      38 * vPlus[5] + 4 * vPlus[6];
      const real fM = 4 * vMinus[1] - 38 * vMinus[2] + 214 * vMinus[3] + 319 * vMinus[4] - 101 * vMinus[5] +
                      25 * vMinus[6] - 3 * vMinus[7];
      fci[l] = (fP + fM) / 420;
    }
  }
}

__device__ void
positive_preserving_limiter_new(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
  real dt, int idx_in_mesh, int max_extent, const real *pv, const real *jac) {
  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

  const int ns = n_var - 5;
  const auto svL = &pv[i_shared * n_var + 5];
  const auto svR = &pv[(i_shared + 1) * n_var + 5];
  const auto rhoL = pv[i_shared * n_var], rhoR = pv[(i_shared + 1) * n_var];
  real *fc_yq_i = &fc[tid * n_var + 5];

  for (int l = 0; l < ns; ++l) {
    real theta_p = 1.0, theta_m = 1.0;
    if (idx_in_mesh > -1) {
      const real up = 0.5 * alpha * svL[l] * rhoL * jac[i_shared] - dt * fc_yq_i[l];
      if (up < 0) {
        const real up_lf = 0.5 * alpha * svL[l] * rhoL * jac[i_shared] - dt * f_1st[tid * ns + l];
        if (abs(up - up_lf) > 1e-20) {
          theta_p = (0 - up_lf) / (up - up_lf);
          if (theta_p > 1)
            theta_p = 1.0;
          else if (theta_p < 0)
            theta_p = 0;
        }
      }
    }

    if (idx_in_mesh < max_extent - 1) {
      const real um = 0.5 * alpha * svR[l] * rhoR * jac[i_shared + 1] + dt * fc_yq_i[l];
      if (um < 0) {
        const real um_lf = 0.5 * alpha * svR[l] * rhoR * jac[i_shared + 1] + dt * f_1st[tid * ns + l];
        if (abs(um - um_lf) > 1e-20) {
          theta_m = (0 - um_lf) / (um - um_lf);
          if (theta_m > 1)
            theta_m = 1.0;
          else if (theta_m < 0)
            theta_m = 0;
        }
      }
    }

    fc_yq_i[l] = min(theta_p, theta_m) * (fc_yq_i[l] - f_1st[tid * ns + l]) + f_1st[tid * ns + l];
  }
}

template<MixtureModel mix_model>
__global__ void compute_convective_term_hybrid_ud_weno_x(DZone *zone, DParameter *param) {
  const int i = static_cast<int>((blockDim.x - 1) * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->mx;
  if (i >= max_extent) return;

  const int block_dim = static_cast<int>(blockDim.x);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;
  const auto n_var{param->n_var};
  const auto n_scalar{param->n_scalar};

  extern __shared__ real s[];
  real *metric = s;
  real *jac = &metric[n_point * 3];
  // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
  real *pv = &jac[n_point];
  real *cGradK = &pv[n_point * n_var];
  real *rhoE = &cGradK[n_point];
  real *uk = &rhoE[n_point];
  real *fc = &uk[n_point];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int tid = static_cast<int>(threadIdx.x);

  bool if_shock = false;
  for (int ii = -ngg + 1; ii <= ngg; ++ii) {
    if (zone->shock_sensor(i + ii, j, k) > param->sensor_threshold) {
      if_shock = true;
      break;
    }
  }

  const int i_shared = tid - 1 + ngg;
  metric[i_shared * 3] = zone->metric(i, j, k, 0);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k, 1);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k, 2);
  jac[i_shared] = zone->jac(i, j, k);
  for (auto l = 0; l < 5; ++l) {
    pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
  }
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
  }
  uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
                 metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
                 metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
  rhoE[i_shared] = zone->cv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cGradK[i_shared] = zone->acoustic_speed(i, j, k);
  else
    cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
                           metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
                           metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);

  // ghost cells
  if (tid < ngg - 1) {
    const int gi = i - (ngg - 1);

    metric[tid * 3] = zone->metric(gi, j, k, 0);
    metric[tid * 3 + 1] = zone->metric(gi, j, k, 1);
    metric[tid * 3 + 2] = zone->metric(gi, j, k, 2);
    jac[tid] = zone->jac(gi, j, k);
    for (auto l = 0; l < 5; ++l) {
      pv[tid * n_var + l] = zone->bv(gi, j, k, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[tid * n_var + 5 + l] = zone->sv(gi, j, k, l);
    }
    uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
              metric[tid * 3 + 1] * pv[tid * n_var + 2] +
              metric[tid * 3 + 2] * pv[tid * n_var + 3];
    rhoE[tid] = zone->cv(gi, j, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[tid] = zone->acoustic_speed(gi, j, k);
    else
      cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
    cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
                        metric[tid * 3 + 1] * metric[tid * 3 + 1] +
                        metric[tid * 3 + 2] * metric[tid * 3 + 2]);
  }
  if (tid > block_dim - ngg - 1 || i > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    const int gi = i + ngg;
    metric[iSh * 3] = zone->metric(gi, j, k, 0);
    metric[iSh * 3 + 1] = zone->metric(gi, j, k, 1);
    metric[iSh * 3 + 2] = zone->metric(gi, j, k, 2);
    jac[iSh] = zone->jac(gi, j, k);
    for (auto l = 0; l < 5; ++l) {
      pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
    }
    uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
              metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
              metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
    rhoE[iSh] = zone->cv(gi, j, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[iSh] = zone->acoustic_speed(gi, j, k);
    else
      cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
    cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                        metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                        metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
  }
  if (i == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      const int gi = i - (ngg - 1 - m - 1);

      metric[iSh * 3] = zone->metric(gi, j, k, 0);
      metric[iSh * 3 + 1] = zone->metric(gi, j, k, 1);
      metric[iSh * 3 + 2] = zone->metric(gi, j, k, 2);
      jac[iSh] = zone->jac(gi, j, k);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(gi, j, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(gi, j, k);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      const int gi = i + (m + 1);

      metric[iSh * 3] = zone->metric(gi, j, k, 0);
      metric[iSh * 3 + 1] = zone->metric(gi, j, k, 1);
      metric[iSh * 3 + 2] = zone->metric(gi, j, k, 2);
      jac[iSh] = zone->jac(gi, j, k);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(gi, j, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(gi, j, k);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
  }
  __syncthreads();

  if (if_shock) {
    //The inviscid term at this point is calculated by WENO scheme.
    hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var],
                                &f_1st[tid * (n_var - 5)]);
    // hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var],
    // &f_1st[tid * (n_var - 5)]);
  } else {
    //The inviscid term at this point is calculated by ep scheme.
    hybrid_ud_part(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var], &f_1st[tid * (n_var - 5)]);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_new(f_1st, n_var, tid, fc, param, i_shared, dt, i, max_extent, pv, jac);
  }
  __syncthreads();

  if (tid > 0 && i >= zone->iMin && i <= zone->iMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void compute_convective_term_hybrid_ud_weno_y(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->my;
  if (j >= max_extent) return;

  const int block_dim = static_cast<int>(blockDim.y);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;
  const auto n_var{param->n_var};
  const auto n_scalar{param->n_scalar};

  extern __shared__ real s[];
  real *metric = s;
  real *jac = &metric[n_point * 3];
  // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
  real *pv = &jac[n_point];
  real *cGradK = &pv[n_point * n_var];
  real *rhoE = &cGradK[n_point];
  real *uk = &rhoE[n_point];
  real *fc = &uk[n_point];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int tid = static_cast<int>(threadIdx.y);

  bool if_shock = false;
  for (int ii = -ngg + 1; ii <= ngg; ++ii) {
    if (zone->shock_sensor(i, j + ii, k) > param->sensor_threshold) {
      if_shock = true;
      break;
    }
  }

  const int i_shared = tid - 1 + ngg;
  metric[i_shared * 3] = zone->metric(i, j, k, 3);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k, 4);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k, 5);
  jac[i_shared] = zone->jac(i, j, k);
  for (auto l = 0; l < 5; ++l) {
    pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
  }
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
  }
  uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
                 metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
                 metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
  rhoE[i_shared] = zone->cv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cGradK[i_shared] = zone->acoustic_speed(i, j, k);
  else
    cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
                           metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
                           metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);

  // ghost cells
  if (tid < ngg - 1) {
    const int gj = j - (ngg - 1);

    metric[tid * 3] = zone->metric(i, gj, k, 3);
    metric[tid * 3 + 1] = zone->metric(i, gj, k, 4);
    metric[tid * 3 + 2] = zone->metric(i, gj, k, 5);
    jac[tid] = zone->jac(i, gj, k);
    for (auto l = 0; l < 5; ++l) {
      pv[tid * n_var + l] = zone->bv(i, gj, k, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[tid * n_var + 5 + l] = zone->sv(i, gj, k, l);
    }
    uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
              metric[tid * 3 + 1] * pv[tid * n_var + 2] +
              metric[tid * 3 + 2] * pv[tid * n_var + 3];
    rhoE[tid] = zone->cv(i, gj, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[tid] = zone->acoustic_speed(i, gj, k);
    else
      cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
    cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
                        metric[tid * 3 + 1] * metric[tid * 3 + 1] +
                        metric[tid * 3 + 2] * metric[tid * 3 + 2]);
  }
  if (tid > block_dim - ngg - 1 || j > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    const int gj = j + ngg;
    metric[iSh * 3] = zone->metric(i, gj, k, 3);
    metric[iSh * 3 + 1] = zone->metric(i, gj, k, 4);
    metric[iSh * 3 + 2] = zone->metric(i, gj, k, 5);
    jac[iSh] = zone->jac(i, gj, k);
    for (auto l = 0; l < 5; ++l) {
      pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
    }
    uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
              metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
              metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
    rhoE[iSh] = zone->cv(i, gj, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[iSh] = zone->acoustic_speed(i, gj, k);
    else
      cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
    cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                        metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                        metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
  }
  if (j == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      const int gj = j - (ngg - 1 - m - 1);

      metric[iSh * 3] = zone->metric(i, gj, k, 3);
      metric[iSh * 3 + 1] = zone->metric(i, gj, k, 4);
      metric[iSh * 3 + 2] = zone->metric(i, gj, k, 5);
      jac[iSh] = zone->jac(i, gj, k);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(i, gj, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(i, gj, k);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      const int gj = j + (m + 1);

      metric[iSh * 3] = zone->metric(i, gj, k, 3);
      metric[iSh * 3 + 1] = zone->metric(i, gj, k, 4);
      metric[iSh * 3 + 2] = zone->metric(i, gj, k, 5);
      jac[iSh] = zone->jac(i, gj, k);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(i, gj, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(i, gj, k);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
  }
  __syncthreads();

  if (if_shock) { //The inviscid term at this point is calculated by WENO scheme.
    hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var],
                                &f_1st[tid * (n_var - 5)]);
    // hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var],
    // &f_1st[tid * (n_var - 5)]);
  } else { //The inviscid term at this point is calculated by ep scheme.
    hybrid_ud_part(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var], &f_1st[tid * (n_var - 5)]);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_new(f_1st, n_var, tid, fc, param, i_shared, dt, j, max_extent, pv, jac);
  }
  __syncthreads();

  if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void compute_convective_term_hybrid_ud_weno_z(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
  const int max_extent = zone->mz;
  if (k >= max_extent) return;

  const int block_dim = static_cast<int>(blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;
  const auto n_var{param->n_var};
  const auto n_scalar{param->n_scalar};

  extern __shared__ real s[];
  real *metric = s;
  real *jac = &metric[n_point * 3];
  // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
  real *pv = &jac[n_point];
  real *cGradK = &pv[n_point * n_var];
  real *rhoE = &cGradK[n_point];
  real *uk = &rhoE[n_point];
  real *fc = &uk[n_point];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  bool if_shock = false;
  for (int ii = -ngg + 1; ii <= ngg; ++ii) {
    if (zone->shock_sensor(i, j, k + ii) > param->sensor_threshold) {
      if_shock = true;
      break;
    }
  }
  // WARNING: This is used for testing WENO scheme.
  //if_shock = false;

  const int tid = static_cast<int>(threadIdx.z);
  const int i_shared = tid - 1 + ngg;
  metric[i_shared * 3] = zone->metric(i, j, k, 6);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k, 7);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k, 8);
  jac[i_shared] = zone->jac(i, j, k);
  for (auto l = 0; l < 5; ++l) {
    pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
  }
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
  }
  uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
                 metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
                 metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
  rhoE[i_shared] = zone->cv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cGradK[i_shared] = zone->acoustic_speed(i, j, k);
  else
    cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
                           metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
                           metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);

  // ghost cells
  if (tid < ngg - 1) {
    const int gk = k - (ngg - 1);

    metric[tid * 3] = zone->metric(i, j, gk, 6);
    metric[tid * 3 + 1] = zone->metric(i, j, gk, 7);
    metric[tid * 3 + 2] = zone->metric(i, j, gk, 8);
    jac[tid] = zone->jac(i, j, gk);
    for (auto l = 0; l < 5; ++l) {
      pv[tid * n_var + l] = zone->bv(i, j, gk, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[tid * n_var + 5 + l] = zone->sv(i, j, gk, l);
    }
    uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
              metric[tid * 3 + 1] * pv[tid * n_var + 2] +
              metric[tid * 3 + 2] * pv[tid * n_var + 3];
    rhoE[tid] = zone->cv(i, j, gk, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[tid] = zone->acoustic_speed(i, j, gk);
    else
      cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
    cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
                        metric[tid * 3 + 1] * metric[tid * 3 + 1] +
                        metric[tid * 3 + 2] * metric[tid * 3 + 2]);
  }
  if (tid > block_dim - ngg - 1 || k > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    const int gk = k + ngg;

    metric[iSh * 3] = zone->metric(i, j, gk, 6);
    metric[iSh * 3 + 1] = zone->metric(i, j, gk, 7);
    metric[iSh * 3 + 2] = zone->metric(i, j, gk, 8);
    jac[iSh] = zone->jac(i, j, gk);
    for (auto l = 0; l < 5; ++l) {
      pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
    }
    for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
      pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
    }
    uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
              metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
              metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
    rhoE[iSh] = zone->cv(i, j, gk, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK[iSh] = zone->acoustic_speed(i, j, gk);
    else
      cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
    cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                        metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                        metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
  }
  if (k == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      const int gk = k - (ngg - 1 - m - 1);

      metric[iSh * 3] = zone->metric(i, j, gk, 6);
      metric[iSh * 3 + 1] = zone->metric(i, j, gk, 7);
      metric[iSh * 3 + 2] = zone->metric(i, j, gk, 8);
      jac[iSh] = zone->jac(i, j, gk);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(i, j, gk, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(i, j, gk);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      const int gk = k + (m + 1);

      metric[iSh * 3] = zone->metric(i, j, gk, 6);
      metric[iSh * 3 + 1] = zone->metric(i, j, gk, 7);
      metric[iSh * 3 + 2] = zone->metric(i, j, gk, 8);
      jac[iSh] = zone->jac(i, j, gk);
      for (auto l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
        pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(i, j, gk, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK[iSh] = zone->acoustic_speed(i, j, gk);
      else
        cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
      cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
                          metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
                          metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
    }
  }
  __syncthreads();

  if (if_shock) { //The inviscid term at this point is calculated by WENO scheme.
    hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var],
                                &f_1st[tid * (n_var - 5)]);
    // hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var], &f_1st[tid * (n_var - 5)]);
  } else { //The inviscid term at this point is calculated by ep scheme.
    hybrid_ud_part(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var], &f_1st[tid * (n_var - 5)]);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter_new(f_1st, n_var, tid, fc, param, i_shared, dt, k, max_extent, pv, jac);
  }
  __syncthreads();

  if (tid > 0 && k >= zone->kMin && k <= zone->kMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model> void compute_convective_term_hybrid_ud_weno(const Block &block, DZone *zone,
  DParameter *param, int n_var, const Parameter &parameter) {
  // The implementation of classic WENO.
  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                              // fc
                     + n_computation_per_block * n_var              // pv[5]+sv[n_scalar]
                     + n_computation_per_block * 7) * sizeof(real); // metric[3]+jacobian+rhoE+Uk+speed_of_sound
  if (parameter.get_bool("positive_preserving")) {
    shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
  }

  dim3 TPB(block_dim, 1, 1);
  dim3 BPG((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
  compute_convective_term_hybrid_ud_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);

  TPB = dim3(1, block_dim, 1);
  BPG = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
  compute_convective_term_hybrid_ud_weno_y<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);

  if (extent[2] > 1) {
    TPB = dim3(1, 1, 64);
    BPG = dim3(extent[0], extent[1], (extent[2] - 1) / (64 - 1) + 1);
    compute_convective_term_hybrid_ud_weno_z<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
  }
}

template void compute_convective_term_hybrid_ud_weno<MixtureModel::Air>(const Block &block, DZone *zone,
  DParameter *param, int n_var, const Parameter &parameter);

template void compute_convective_term_hybrid_ud_weno<MixtureModel::Mixture>(const Block &block, DZone *zone,
  DParameter *param, int n_var, const Parameter &parameter);
}

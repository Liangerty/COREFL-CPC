#include "FiniteRateChem.cuh"
#include "Field.h"
#include "Thermo.cuh"
#include "Constants.h"

namespace cfd {
__device__ void forward_reaction_rate_1(real t, real *kf, const real *concentration) {
  // According to chemistry/2004-Li-IntJ.Chem.Kinet.inp
  constexpr real iR_c = 1.0 / R_c;
  const real iT = 1.0 / t;
  const real iRcT = iR_c * iT;
  kf[0] = 3.55e+15 * pow(t, -0.41) * exp(-1.66e+4 * iRcT);
  kf[1] = 5.08e+4 * pow(t, 2.67) * exp(-6290 * iRcT);
  kf[2] = 2.16e+8 * pow(t, 1.51) * exp(-3430 * iRcT);
  kf[3] = 2.97e+6 * pow(t, 2.02) * exp(-13400 * iRcT);
  kf[4] = 4.58e+19 * pow(t, -1.40) * exp(-1.0438e+5 * iRcT);
  const real cc = concentration[0] * 2.5 + concentration[1] + concentration[2] +
                  concentration[3] + concentration[4] + concentration[5] +
                  concentration[6] + concentration[7] * 12 + concentration[8];
  kf[4] *= cc; // H2 + M = H + H + M
  kf[5] = 6.16e+15 * sqrt(iT) * cc;
  kf[6] = 4.71e+18 * iT * cc;
  kf[7] = 3.80e+22 * iT * iT * cc;
  real kf_high = 1.48e+12 * pow(t, 0.6);
  real kf_low = 6.37e+20 * pow(t, -1.72) * exp(-5.2e+2 * iRcT);
  if (kf_high < 1e-25 && kf_low < 1e-25) {
    // If both kf_high and kf_low are too small, set kf to zero
    kf[8] = 0;
  } else {
    const real cc_here = concentration[0] * 2 + concentration[1] + concentration[2] * 0.78 +
                         concentration[3] + concentration[4] + concentration[5] +
                         concentration[6] + concentration[7] * 11 + concentration[8];
    const real reduced_pressure = kf_low * cc_here / kf_high;
    constexpr real f_cent = 0.8;
    const real logFc = log10(f_cent);
    const real c = -0.4 - 0.67 * logFc;
    const real n = 0.75 - 1.27 * logFc;
    const real logPr = log10(reduced_pressure);
    const real tempo = (logPr + c) / (n - 0.14 * (logPr + c));
    const real p = logFc / (1.0 + tempo * tempo);
    kf[8] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow(10, p);
  }
  kf[9] = 1.66e+13 * exp(-820 * iRcT);
  kf[10] = 7.08e+13 * exp(-300 * iRcT);
  kf[11] = 3.25e+13;                                                    // O + H2 = H + OH
  kf[12] = 2.89e+13 * exp(500 * iRcT);                                  // HO2 + OH = H2O + O2
  kf[13] = 4.20e+14 * exp(-11980 * iRcT) + 1.30e+11 * exp(1630 * iRcT); // HO2 + HO2 = H2O2 + O2
  kf_high = 2.95e+14 * exp(-48400 * iRcT);
  kf_low = 1.20e+17 * exp(-45500 * iRcT);
  if (kf_high < 1e-25 && kf_low < 1e-25) {
    // If both kf_high and kf_low are too small, set kf to zero
    kf[14] = 0;
  } else {
    const real reduced_pressure = kf_low * cc / kf_high;
    constexpr real f_cent = 0.5;
    const real logFc = log10(f_cent);
    const real c = -0.4 - 0.67 * logFc;
    const real n = 0.75 - 1.27 * logFc;
    const real logPr = log10(reduced_pressure);
    const real tempo = (logPr + c) / (n - 0.14 * (logPr + c));
    const real p = logFc / (1.0 + tempo * tempo);
    kf[14] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow(10, p);
  }
  kf[15] = 2.41e+13 * exp(-3970 * iRcT);        // H2O2 + H = H2O + OH
  kf[16] = 4.82e+13 * exp(-7950 * iRcT);        // H2O2 + H = HO2 + H2
  kf[17] = 9.55e+6 * t * t * exp(-3970 * iRcT); // H2O2 + O = OH + HO2
  kf[18] = 1e+12 + 5.8e+14 * exp(-9560 * iRcT); // H2O2 + OH = HO2 + H2O
}

__device__ void
backward_reaction_rate_1(real t, const real *kf, real *kb, const DParameter *param) {
  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);

  constexpr real temp_p = p_atm / R_u * 1e-3; // Convert the unit to mol*K/cm3
  const real temp_t = temp_p / t;             // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  kb[0] = kf[0] * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
  kb[1] = kf[1] * exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
  kb[2] = kf[2] * exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
  kb[3] = kf[3] * exp(-gibbs_rt[3] - gibbs_rt[7] + gibbs_rt[4] + gibbs_rt[4]);
  kb[4] = kf[4] * iTemp_t * exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
  kb[5] = kf[5] * temp_t * exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
  kb[6] = kf[6] * temp_t * exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
  kb[7] = kf[7] * temp_t * exp(-gibbs_rt[1] - gibbs_rt[4] + gibbs_rt[7]);
  kb[8] = kf[8] * temp_t * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
  kb[9] = kf[9] * exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
  kb[10] = kf[10] * exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
  kb[11] = kf[11] * exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
  kb[12] = kf[12] * exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
  kb[13] = kf[13] * exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
  kb[14] = kf[14] * iTemp_t * exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
  kb[15] = kf[15] * exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
  kb[16] = kf[16] * exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
  kb[17] = kf[17] * exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
  kb[18] = kf[18] * exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
}

__device__ void rate_of_progress_1(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2) {
  q1[0] = kf[0] * c[2] * c[1];
  q2[0] = kb[0] * c[3] * c[4];
  q[0] = q1[0] - q2[0];
  q1[1] = kf[1] * c[0] * c[3];
  q2[1] = kb[1] * c[1] * c[4];
  q[1] = q1[1] - q2[1];
  q1[2] = kf[2] * c[0] * c[4];
  q2[2] = kb[2] * c[7] * c[1];
  q[2] = q1[2] - q2[2];
  q1[3] = kf[3] * c[3] * c[7];
  q2[3] = kb[3] * c[4] * c[4];
  q[3] = q1[3] - q2[3];
  q1[4] = kf[4] * c[0];
  q2[4] = kb[4] * c[1] * c[1];
  q[4] = q1[4] - q2[4];
  q1[5] = kf[5] * c[3] * c[3];
  q2[5] = kb[5] * c[2];
  q[5] = q1[5] - q2[5];
  q1[6] = kf[6] * c[3] * c[1];
  q2[6] = kb[6] * c[4];
  q[6] = q1[6] - q2[6];
  q1[7] = kf[7] * c[1] * c[4];
  q2[7] = kb[7] * c[7];
  q[7] = q1[7] - q2[7];
  q1[8] = kf[8] * c[1] * c[2];
  q2[8] = kb[8] * c[5];
  q[8] = q1[8] - q2[8];
  q1[9] = kf[9] * c[5] * c[1];
  q2[9] = kb[9] * c[0] * c[2];
  q[9] = q1[9] - q2[9];
  q1[10] = kf[10] * c[5] * c[1];
  q2[10] = kb[10] * c[4] * c[4];
  q[10] = q1[10] - q2[10];
  q1[11] = kf[11] * c[5] * c[3];
  q2[11] = kb[11] * c[2] * c[4];
  q[11] = q1[11] - q2[11];
  q1[12] = kf[12] * c[5] * c[4];
  q2[12] = kb[12] * c[7] * c[2];
  q[12] = q1[12] - q2[12];
  q1[13] = kf[13] * c[5] * c[5];
  q2[13] = kb[13] * c[6] * c[2];
  q[13] = q1[13] - q2[13];
  q1[14] = kf[14] * c[6];
  q2[14] = kb[14] * c[4] * c[4];
  q[14] = q1[14] - q2[14];
  q1[15] = kf[15] * c[6] * c[1];
  q2[15] = kb[15] * c[7] * c[4];
  q[15] = q1[15] - q2[15];
  q1[16] = kf[16] * c[6] * c[1];
  q2[16] = kb[16] * c[5] * c[0];
  q[16] = q1[16] - q2[16];
  q1[17] = kf[17] * c[6] * c[3];
  q2[17] = kb[17] * c[4] * c[5];
  q[17] = q1[17] - q2[17];
  q1[18] = kf[18] * c[6] * c[4];
  q2[18] = kb[18] * c[5] * c[7];
  q[18] = q1[18] - q2[18];
}

__device__ void chemical_source_1(const real *q1, const real *q2, real *d, real *omega) {
  real c = 0;
  c = (q2[1] + q2[2] + q2[4] + q1[9] + q1[16]) * 2016;
  d[0] = (q1[1] + q1[2] + q1[4] + q2[9] + q2[16]) * 2016;
  omega[0] = c - d[0]; // Unit is kg/(m3*s)
  c = (q2[0] + q2[6] + q2[7] + q2[8] + q2[9] + q2[10] + q2[15] + q2[16] + q1[1] + q1[2] + q1[4] * 2) * 1008;
  d[1] = (q1[0] + q1[6] + q1[7] + q1[8] + q1[9] + q1[10] + q1[15] + q1[16] + q2[1] + q2[2] + q2[4] * 2) * 1008;
  omega[1] = c - d[1]; // Unit is kg/(m3*s)
  c = (q2[0] + q2[8] + q1[5] + q1[9] + q1[11] + q1[12] + q1[13]) * 31998;
  d[2] = (q1[0] + q1[8] + q2[5] + q2[9] + q2[11] + q2[12] + q2[13]) * 31998;
  omega[2] = c - d[2]; // Unit is kg/(m3*s)
  c = (q2[1] + q2[3] + q2[5] * 2 + q2[6] + q2[11] + q2[17] + q1[0]) * 15999;
  d[3] = (q1[1] + q1[3] + q1[5] * 2 + q1[6] + q1[11] + q1[17] + q2[0]) * 15999;
  omega[3] = c - d[3]; // Unit is kg/(m3*s)
  c = (q2[2] + q2[7] + q2[12] + q2[18] + q1[0] + q1[1] + q1[3] * 2 + q1[6] + q1[10] * 2 + q1[11] + q1[14] * 2 + q1[15] +
       q1[17]) * 17007;
  d[4] = (q1[2] + q1[7] + q1[12] + q1[18] + q2[0] + q2[1] + q2[3] * 2 + q2[6] + q2[10] * 2 + q2[11] + q2[14] * 2
          + q2[15] + q2[17]) * 17007;
  omega[4] = c - d[4];
  c = (q2[9] + q2[10] + q2[11] + q2[12] + q2[13] * 2 + q1[8] + q1[16] + q1[17] + q1[18]) * 33006;
  d[5] = (q1[9] + q1[10] + q1[11] + q1[12] + q1[13] * 2 + q2[8] + q2[16] + q2[17] + q2[18]) * 33006;
  omega[5] = c - d[5]; // Unit is kg/(m3*s)
  c = (q2[14] + q2[15] + q2[16] + q2[17] + q2[18] + q1[13]) * 34014;
  d[6] = (q1[14] + q1[15] + q1[16] + q1[17] + q1[18] + q2[13]) * 34014;
  omega[6] = c - d[6]; // Unit is kg/(m3*s)
  c = (q2[3] + q1[2] + q1[7] + q1[12] + q1[15] + q1[18]) * 18015;
  d[7] = (q1[3] + q2[2] + q2[7] + q2[12] + q2[15] + q2[18]) * 18015;
  omega[7] = c - d[7]; // Unit is kg/(m3*s)
  d[8] = 0;
  omega[8] = 0;
}

__global__ void finite_rate_chemistry_1(DZone *zone, const DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &bv = zone->bv;
  const auto &sv = zone->sv;

  const int ns = param->n_spec;

  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const real density{bv(i, j, k, 0)};
  const auto imw = param->imw;
  for (int l = 0; l < ns; ++l) {
    c[l] = density * sv(i, j, k, l) * imw[l] * 1e-3;
  }

  // compute the forward reaction rate
  const real t{bv(i, j, k, 5)};
  real kf[MAX_REAC_NUMBER];
  forward_reaction_rate_1(t, kf, c);

  // compute the backward reaction rate
  real kb[MAX_REAC_NUMBER] = {};
  backward_reaction_rate_1(t, kf, kb, param);

  // compute the rate of progress
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  rate_of_progress_1(kf, kb, c, q, q1, q2);

  // compute the chemical source
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];
  chemical_source_1(q1, q2, omega_d, omega);

  real time_scale{1e+6};
  for (int l = 0; l < ns; ++l) {
    zone->dq(i, j, k, l + 5) += zone->jac(i, j, k) * omega[l];
    if (sv(i, j, k, l) > 1e-25 && abs(omega_d[l]) > 1e-25)
      time_scale = min(time_scale, abs(density * sv(i, j, k, l) / omega_d[l]));
  }
  zone->reaction_timeScale(i, j, k) = time_scale;

  // If implicit treat
  switch (param->chemSrcMethod) {
    case 0: // Explicit treatment
      break;
    case 1: // Exact point implicit
      compute_chem_src_jacobian(zone, i, j, k, param, q1, q2);
      break;
    case 2: // Diagonal approximation
      compute_chem_src_jacobian_diagonal(zone, i, j, k, param, omega_d);
      break;
    default: // Default, explicit treatment
      break;
  }
}

__global__ void finite_rate_chemistry(DZone *zone, const DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &bv = zone->bv;
  const auto &sv = zone->sv;

  const int ns = param->n_spec;

  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const real density{bv(i, j, k, 0)};
  const auto imw = param->imw;
  for (int l = 0; l < ns; ++l) {
    c[l] = density * sv(i, j, k, l) * imw[l] * 1e-3;
  }

  // compute the forward reaction rate
  const real t{bv(i, j, k, 5)};
  real kf[MAX_REAC_NUMBER];
  forward_reaction_rate(t, kf, c, param);

  // compute the backward reaction rate
  real kb[MAX_REAC_NUMBER] = {};
  backward_reaction_rate(t, kf, c, param, kb);

  // compute the rate of progress
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  rate_of_progress(kf, kb, c, q, q1, q2, param);

  // compute the chemical source
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];
  chemical_source(q1, q2, omega_d, omega, param);

  real time_scale{1e+6};
  for (int l = 0; l < ns; ++l) {
    zone->dq(i, j, k, l + 5) += zone->jac(i, j, k) * omega[l];
    if (sv(i, j, k, l) > 1e-25 && abs(omega_d[l]) > 1e-25)
      time_scale = min(time_scale, abs(density * sv(i, j, k, l) / omega_d[l]));
  }
  zone->reaction_timeScale(i, j, k) = time_scale;

  // If implicit treat
  switch (param->chemSrcMethod) {
    case 0: // Explicit treatment
      break;
    case 1: // Exact point implicit
      compute_chem_src_jacobian(zone, i, j, k, param, q1, q2);
      break;
    case 2: // Diagonal approximation
      compute_chem_src_jacobian_diagonal(zone, i, j, k, param, omega_d);
      break;
    default: // Default, explicit treatment
      break;
  }
}

__device__ void forward_reaction_rate(real t, real *kf, const real *concentration, const DParameter *param) {
  const auto A = param->A, b = param->b, Ea = param->Ea;
  const auto type = param->reac_type;
  const auto A2 = param->A2, b2 = param->b2, Ea2 = param->Ea2;
  const auto &third_body_coeff = param->third_body_coeff;
  const auto alpha = param->troe_alpha, t3 = param->troe_t3, t1 = param->troe_t1, t2 = param->troe_t2;
  for (int i = 0; i < param->n_reac; ++i) {
    kf[i] = arrhenius(t, A[i], b[i], Ea[i]);
    if (type[i] == 3) {
      // Duplicate reaction
      kf[i] += arrhenius(t, A2[i], b2[i], Ea2[i]);
    } else if (type[i] > 3) {
      real cc{0};
      for (int l = 0; l < param->n_spec; ++l) {
        cc += concentration[l] * third_body_coeff(i, l);
      }
      if (type[i] == 4) {
        // Third body reaction
        kf[i] *= cc;
      } else {
        const real kf_low = arrhenius(t, A2[i], b2[i], Ea2[i]);
        const real kf_high = kf[i];
        if (kf_high < 1e-25 && kf_low < 1e-25) {
          // If both kf_high and kf_low are too small, set kf to zero
          kf[i] = 0;
          continue;
        }
        const real reduced_pressure = kf_low * cc / kf_high;
        real F = 1.0;
        if (type[i] > 5) {
          // Troe form
          real f_cent = (1 - alpha[i]) * std::exp(-t / t3[i]) + alpha[i] * std::exp(-t / t1[i]);
          if (type[i] == 7) {
            f_cent += std::exp(-t2[i] / t);
          }
          const real logFc = std::log10(f_cent);
          const real c = -0.4 - 0.67 * logFc;
          const real n = 0.75 - 1.27 * logFc;
          const real logPr = std::log10(reduced_pressure);
          const real tempo = (logPr + c) / (n - 0.14 * (logPr + c));
          const real p = logFc / (1.0 + tempo * tempo);
          F = std::pow(10, p);
        }
        kf[i] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * F;
      }
    }
  }
}

__device__ real arrhenius(real t, real A, real b, real Ea) {
  return A * std::pow(t, b) * std::exp(-Ea / t);
}

__device__ void
backward_reaction_rate(real t, const real *kf, const real *concentration, const DParameter *param, real *kb) {
  int n_gibbs{param->n_reac};
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] == 0) {
      // Irreversible reaction
      kb[i] = 0;
      --n_gibbs;
    } else if (param->rev_type[i] == 1) {
      // REV reaction
      kb[i] = arrhenius(t, param->A2[i], param->b2[i], param->Ea2[i]);
      if (param->reac_type[i] == 4) {
        // Third body required
        real cc{0};
        for (int l = 0; l < param->n_spec; ++l) {
          cc += concentration[l] * param->third_body_coeff(i, l);
        }
        kb[i] *= cc;
      }
      --n_gibbs;
    }
  }
  if (n_gibbs < 1)
    return;

  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);
  constexpr real temp_p = p_atm / R_u * 1e-3; // Convert the unit to mol*K/cm3
  const real temp_t = temp_p / t;             // Unit is mol/cm3
  const auto &stoi_f = param->stoi_f, &stoi_b = param->stoi_b;
  const auto order = param->reac_order;
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] != 2 && param->reac_type[i] != 0) {
      real d_gibbs{0};
      for (int l = 0; l < param->n_spec; ++l) {
        d_gibbs += gibbs_rt[l] * (stoi_b(i, l) - stoi_f(i, l));
      }
      const real kc{std::pow(temp_t, order[i]) * std::exp(-d_gibbs)};
      kb[i] = kf[i] / kc;
    }
  }
}

__device__ void
rate_of_progress(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2, const DParameter *param) {
  const int n_spec{param->n_spec};
  const auto &stoi_f{param->stoi_f}, &stoi_b{param->stoi_b};
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] != 0) {
      q1[i] = 1.0;
      q2[i] = 1.0;
      for (int j = 0; j < n_spec; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
        q2[i] *= std::pow(c[j], stoi_b(i, j));
      }
      q1[i] *= kf[i];
      q2[i] *= kb[i];
      q[i] = q1[i] - q2[i];
    } else {
      q1[i] = 1.0;
      q2[i] = 0.0;
      for (int j = 0; j < n_spec; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
      }
      q1[i] *= kf[i];
      q[i] = q1[i];
    }
  }
}

__device__ void chemical_source(const real *q1, const real *q2, real *omega_d, real *omega, const DParameter *param) {
  const int n_spec{param->n_spec};
  const int n_reac{param->n_reac};
  const auto &stoi_f = param->stoi_f, &stoi_b{param->stoi_b};
  const auto imw = param->imw;
  for (int i = 0; i < n_spec; ++i) {
    real creation = 0;
    omega_d[i] = 0;
    for (int j = 0; j < n_reac; ++j) {
      creation += q2[j] * stoi_f(j, i) + q1[j] * stoi_b(j, i);
      omega_d[i] += q1[j] * stoi_f(j, i) + q2[j] * stoi_b(j, i);
    }
    creation *= 1e+3 / imw[i];        // Unit is kg/(m3*s)
    omega_d[i] *= 1e+3 / imw[i];      // Unit is kg/(m3*s)
    omega[i] = creation - omega_d[i]; // Unit is kg/(m3*s)
  }
}

__device__ void
compute_chem_src_jacobian(DZone *zone, int i, int j, int k, const DParameter *param, const real *q1, const real *q2) {
  const int n_spec{param->n_spec}, n_reac{param->n_reac};
  auto &sv = zone->sv;
  const auto &stoi_f = param->stoi_f, &stoi_b = param->stoi_b;
  const real density{zone->bv(i, j, k, 0)};
  auto &chem_jacobian = zone->chem_src_jac;
  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      real zz{0};
      if (sv(i, j, k, n) > 1e-25) {
        for (int r = 0; r < n_reac; ++r) {
          // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
          zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, n) * q1[r] - stoi_b(r, n) * q2[r]);
        }
        zz /= density * sv(i, j, k, n);
      }
      chem_jacobian(i, j, k, m * n_spec + n) = zz * 1e+3 / param->imw[m]; // //1e+3=1e-3(MW)*1e+6(cm->m)
    }
  }
}

__device__ void
compute_chem_src_jacobian_diagonal(DZone *zone, int i, int j, int k, const DParameter *param, const real *omega_d) {
  // The method described in 2015-Savard-JCP
  auto &chem_jacobian = zone->chem_src_jac;
  auto &sv = zone->sv;
  const real density{zone->bv(i, j, k, 0)};
  for (int l = 0; l < param->n_spec; ++l) {
    chem_jacobian(i, j, k, l) = 0;
    if (sv(i, j, k, l) > 1e-25) {
      chem_jacobian(i, j, k, l) = -omega_d[l] / (sv(i, j, k, l) * density);
    }
  }
}

__global__ void EPI(DZone *zone, int n_spec) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &chem_jac = zone->chem_src_jac;
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};

  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = 1.0 - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }
  solve_chem_system(lhs, zone, i, j, k, n_spec);
}

__device__ void EPI_for_dq0(DZone *zone, real diag, int i, int j, int k, int n_spec) {
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};
  auto &chem_jac = zone->chem_src_jac;

  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = diag - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }
  solve_chem_system(lhs, zone, i, j, k, n_spec);
}

__device__ void
EPI_for_dqk(DZone *zone, real diag, int i, int j, int k, const real *dq_total, int n_spec) {
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};

  auto &chem_jac = zone->chem_src_jac;
  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = diag - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }

  real rhs[MAX_SPEC_NUMBER] = {};
  for (int l = 0; l < n_spec; ++l) {
    rhs[l] = dq_total[5 + l] * dt;
  }
  solve_chem_system(lhs, rhs, n_spec);

  auto &dqk = zone->dqk;
  const auto &dq0 = zone->dq0;
  for (int l = 0; l < n_spec; ++l) {
    dqk(i, j, k, l + 5) = dq0(i, j, k, l + 5) + rhs[l];
  }
}

__device__ void solve_chem_system(real *lhs, DZone *zone, int i, int j, int k, int n_spec) {
  const int dim{n_spec};
  int ipiv[MAX_SPEC_NUMBER] = {};

  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    ipiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }

  auto &b = zone->dq;
  // Solve the linear system with LU matrix
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = b(i, j, k, 5 + t);
      b(i, j, k, 5 + t) = b(i, j, k, 5 + m);
      b(i, j, k, 5 + m) = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      b(i, j, k, 5 + m) -= lhs[m * dim + t] * b(i, j, k, 5 + t);
    }
  }
  b(i, j, k, 5 + dim - 1) /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      b(i, j, k, 5 + m) -= lhs[m * dim + t] * b(i, j, k, 5 + t);
    }
    b(i, j, k, 5 + m) /= lhs[m * dim + m];
  }
}

__device__ void solve_chem_system(real *lhs, real *rhs, int dim) {
  int ipiv[MAX_SPEC_NUMBER] = {};

  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    ipiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }

  // Solve the linear system with LU matrix
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = rhs[t];
      rhs[t] = rhs[m];
      rhs[m] = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
  }
  rhs[dim - 1] /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
    rhs[m] /= lhs[m * dim + m];
  }
}

__global__ void DA(DZone *zone, int n_spec) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt{zone->dt_local(i, j, k)};
  auto &chem_jac = zone->chem_src_jac;
  for (int l = 0; l < n_spec; ++l) {
    zone->dq(i, j, k, 5 + l) /= 1 - dt * chem_jac(i, j, k, l);
  }
}
} // cfd

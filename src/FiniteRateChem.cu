#include "FiniteRateChem.cuh"
#include "Field.h"
#include "Thermo.cuh"
#include "Constants.h"

namespace cfd {
__device__ __forceinline__ real log10_real(real x) {
  // log10(x) = ln(x) / ln(10)
  constexpr real INV_LN10 = 4.342944819032518e-01;
  return log(x) * INV_LN10;
}

__device__ __forceinline__ real pow10_real(real p) {
  // 10^p = exp(ln(10) * p)
  constexpr real kLn10 = 2.302585092994046; // ln(10) 2.302585092994046
  return exp(kLn10 * p);
}

__device__ void chemical_source_hardCoded1(real t, const real * __restrict__ c,
  const DParameter * __restrict__ param, real * __restrict__ omega_d, real * __restrict__ omega) {
  // According to chemistry/2004-Li-IntJ.Chem.Kinet.inp
  // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2

  // G/RT for Kc
  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);

  // thermodynamic scaling for kc and kb
  const real temp_t = p_atm / R_u * 1e-3 / t; // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  // Arrhenius parameters
  const real iT = 1.0 / t;
  const real iRcT = 1.0 / R_c * iT;
  const real logT = log(t);

  // Third body concentrations used here
  const real cc = c[0] * 2.5 + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] * 12 + c[8];

  // Production / destruction in mol/(cm3*s)
  real prod0{0}, prod1{0}, prod2{0}, prod3{0}, prod4{0}, prod5{0}, prod6{0}, prod7{0}, prod8{0};
  real dest0{0}, dest1{0}, dest2{0}, dest3{0}, dest4{0}, dest5{0}, dest6{0}, dest7{0}, dest8{0};

  // ----Reaction 0: H + O2 = O + OH
  {
    real kfb = 3.55e+15 * exp(-0.41 * logT - 1.66e+4 * iRcT);
    real qfb = kfb * c[1] * c[2];
    dest1 += qfb;
    dest2 += qfb;
    prod3 += qfb;
    prod4 += qfb;
    kfb = kfb * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
    qfb = kfb * c[3] * c[4];
    dest3 += qfb;
    dest4 += qfb;
    prod1 += qfb;
    prod2 += qfb;
  }
  // ----Reaction 1: H2 + O = H + OH
  {
    real kfb = 5.08e+4 * exp(2.67 * logT - 6290 * iRcT);
    real qfb = kfb * c[0] * c[3];
    dest0 += qfb;
    dest3 += qfb;
    prod1 += qfb;
    prod4 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
    qfb = kfb * c[1] * c[4];
    dest1 += qfb;
    dest4 += qfb;
    prod0 += qfb;
    prod3 += qfb;
  }
  // ----Reaction 2: H2 + OH = H2O + H
  {
    real kfb = 2.16e+8 * exp(1.51 * logT - 3430 * iRcT);
    real qfb = kfb * c[0] * c[4];
    dest0 += qfb;
    dest4 += qfb;
    prod7 += qfb;
    prod1 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
    qfb = kfb * c[7] * c[1];
    dest7 += qfb;
    dest1 += qfb;
    prod0 += qfb;
    prod4 += qfb;
  }
  // ----Reaction 3: O + H2O = OH + OH
  {
    real kf = 2.97e+6 * exp(2.02 * logT - 13400 * iRcT);
    real qf = kf * c[3] * c[7];
    dest3 += qf;
    dest7 += qf;
    prod4 += 2 * qf;
    kf *= exp(-gibbs_rt[3] - gibbs_rt[7] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    dest4 += 2 * qf;
    prod3 += qf;
    prod7 += qf;
  }
  // ----Reaction 4: H2 + M = H + H + M
  {
    real kf = 4.58e+19 * exp(-1.40 * logT - 1.0438e+5 * iRcT);
    kf *= cc;
    real qf = kf * c[0];
    dest0 += qf;
    prod1 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
    qf = kf * c[1] * c[1];
    dest1 += 2 * qf;
    prod0 += qf;
  }
  // ----Reaction 5: O + O + M = O2 + M
  {
    real kf = 6.16e+15 * sqrt(iT) * cc;
    real qf = kf * c[3] * c[3];
    dest3 += 2 * qf;
    prod2 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
    qf = kf * c[2];
    dest2 += qf;
    prod3 += 2 * qf;
  }
  // ----Reaction 6: O + H + M = OH + M
  {
    real kf = 4.71e+18 * iT * cc;
    real qf = kf * c[1] * c[3];
    dest3 += qf;
    dest1 += qf;
    prod4 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[4];
    dest4 += qf;
    prod3 += qf;
    prod1 += qf;
  }
  // ----Reaction 7: H + OH + M = H2O + M
  {
    real kf = 3.8e+22 * iT * iT * cc;
    real qf = kf * c[1] * c[4];
    dest1 += qf;
    dest4 += qf;
    prod7 += qf;
    kf *= temp_t * exp(-gibbs_rt[1] - gibbs_rt[4] + gibbs_rt[7]);
    qf = kf * c[7];
    dest7 += qf;
    prod1 += qf;
    prod4 += qf;
  }
  // ----Reaction 8: H + O2 (+M) = HO2 (+M)
  {
    const real kf_high = 1.48e+12 * exp(0.6 * logT);
    const real kf_low = 6.37e+20 * exp(-1.72 * logT - 5.2e+2 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real cc2 = c[0] * 2 + c[1] + c[2] * 0.78 + c[3] + c[4] + c[5] + c[6] + c[7] * 11 + c[8];
      const real reduced_pressure = kf_low * cc2 / kf_high;
      constexpr real logFc = -0.09691001300805639; // log10(0.8) = -9.691001300805639e-02
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[1] * c[2];
    dest1 += qf;
    dest2 += qf;
    prod5 += qf;
    kf *= temp_t * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
    qf = kf * c[5];
    dest5 += qf;
    prod1 += qf;
    prod2 += qf;
  }
  // ----Reaction 9: HO2 + H = H2 + O2
  {
    real kf = 1.66e+13 * exp(-820 * iRcT);
    real qf = kf * c[5] * c[1];
    dest5 += qf;
    dest1 += qf;
    prod0 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
    qf = kf * c[0] * c[2];
    dest0 += qf;
    dest2 += qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 10: HO2 + H = OH + OH
  {
    real kf = 7.08e+13 * exp(-300 * iRcT);
    real qf = kf * c[5] * c[1];
    dest5 += qf;
    dest1 += qf;
    prod4 += 2 * qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    dest4 += 2 * qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 11: HO2 + O = O2 + OH
  {
    real qf = 3.25e+13 * c[5] * c[3];
    dest5 += qf;
    dest3 += qf;
    prod2 += qf;
    prod4 += qf;
    const real kb = 3.25e+13 * exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
    qf = kb * c[2] * c[4];
    dest2 += qf;
    dest4 += qf;
    prod5 += qf;
    prod3 += qf;
  }
  // ----Reaction 12: HO2 + OH = H2O + O2
  {
    real kf = 2.89e+13 * exp(500 * iRcT);
    real qf = kf * c[5] * c[4];
    dest5 += qf;
    dest4 += qf;
    prod7 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
    qf = kf * c[7] * c[2];
    dest7 += qf;
    dest2 += qf;
    prod5 += qf;
    prod4 += qf;
  }
  // ----Reaction 13: HO2 + HO2 = H2O2 + O2
  {
    real kf = 4.20e+14 * exp(-11980 * iRcT) + 1.30e+11 * exp(1630 * iRcT);
    real qf = kf * c[5] * c[5];
    dest5 += 2 * qf;
    prod6 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
    qf = kf * c[6] * c[2];
    dest6 += qf;
    dest2 += qf;
    prod5 += 2 * qf;
  }
  // ----Reaction 14: H2O2 (+ M) = OH + OH (+ M)
  {
    const real kf_high = 2.95e+14 * exp(-48400 * iRcT);
    const real kf_low = 1.20e+17 * exp(-45500 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real reduced_pressure = kf_low * cc / kf_high;
      constexpr real logFc = -3.010299956639812e-01; // log10(0.5) = -3.010299956639812e-01
      const real cT = -0.4 - 0.67 * logFc;
      const real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[6];
    dest6 += qf;
    prod4 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    dest4 += 2 * qf;
    prod6 += qf;
  }
  // ----Reaction 15: H2O2 + H = H2O + OH
  {
    real kf = 2.41e+13 * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[1];
    dest6 += qf;
    dest1 += qf;
    prod7 += qf;
    prod4 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
    qf = kf * c[7] * c[4];
    dest7 += qf;
    dest4 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ---Reaction 16: H2O2 + H = HO2 + H2
  {
    real kf = 4.82e+13 * exp(-7950 * iRcT);
    real qf = kf * c[6] * c[1];
    dest6 += qf;
    dest1 += qf;
    prod5 += qf;
    prod0 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
    qf = kf * c[5] * c[0];
    dest5 += qf;
    dest0 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ----Reaction 17: H2O2 + O = OH + HO2
  {
    real kf = 9.55e+6 * t * t * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[3];
    dest6 += qf;
    dest3 += qf;
    prod4 += qf;
    prod5 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
    qf = kf * c[4] * c[5];
    dest4 += qf;
    dest5 += qf;
    prod6 += qf;
    prod3 += qf;
  }
  // ----Reaction 18: H2O2 + OH = HO2 + H2O
  {
    real kf = 1e+12 + 5.8e+14 * exp(-9560 * iRcT);
    real qf = kf * c[6] * c[4];
    dest6 += qf;
    dest4 += qf;
    prod5 += qf;
    prod7 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
    qf = kf * c[5] * c[7];
    dest5 += qf;
    dest7 += qf;
    prod6 += qf;
    prod4 += qf;
  }

  // Compute net production rates
  // Convert from mol/(cm3*s) to kg/(m3*s)
  omega_d[0] = dest0 * 2016;
  omega_d[1] = dest1 * 1008;
  omega_d[2] = dest2 * 31998;
  omega_d[3] = dest3 * 15999;
  omega_d[4] = dest4 * 17007;
  omega_d[5] = dest5 * 33006;
  omega_d[6] = dest6 * 34014;
  omega_d[7] = dest7 * 18015;
  omega_d[8] = dest8 * 28014;

  omega[0] = (prod0 - dest0) * 2016;
  omega[1] = (prod1 - dest1) * 1008;
  omega[2] = (prod2 - dest2) * 31998;
  omega[3] = (prod3 - dest3) * 15999;
  omega[4] = (prod4 - dest4) * 17007;
  omega[5] = (prod5 - dest5) * 33006;
  omega[6] = (prod6 - dest6) * 34014;
  omega[7] = (prod7 - dest7) * 18015;
  omega[8] = (prod8 - dest8) * 28014;
}

__device__ void chemical_source_hardCoded1(real t, const real * __restrict__ c,
  const DParameter * __restrict__ param, real * __restrict__ omega_d, real * __restrict__ omega, real * __restrict__ q1,
  real * __restrict__ q2) {
  // According to chemistry/2004-Li-IntJ.Chem.Kinet.inp
  // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2

  // G/RT for Kc
  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);

  // thermodynamic scaling for kc and kb
  const real temp_t = p_atm / R_u * 1e-3 / t; // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  // Arrhenius parameters
  const real iT = 1.0 / t;
  const real iRcT = 1.0 / R_c * iT;
  const real logT = log(t);

  // Third body concentrations used here
  const real cc = c[0] * 2.5 + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] * 12 + c[8];

  // Production / destruction in mol/(cm3*s)
  real prod0{0}, prod1{0}, prod2{0}, prod3{0}, prod4{0}, prod5{0}, prod6{0}, prod7{0}, prod8{0};
  real dest0{0}, dest1{0}, dest2{0}, dest3{0}, dest4{0}, dest5{0}, dest6{0}, dest7{0}, dest8{0};

  // ----Reaction 0: H + O2 = O + OH
  {
    real kfb = 3.55e+15 * exp(-0.41 * logT - 1.66e+4 * iRcT);
    real qfb = kfb * c[1] * c[2];
    q1[0] = qfb;
    dest1 += qfb;
    dest2 += qfb;
    prod3 += qfb;
    prod4 += qfb;
    kfb = kfb * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
    qfb = kfb * c[3] * c[4];
    q2[0] = qfb;
    dest3 += qfb;
    dest4 += qfb;
    prod1 += qfb;
    prod2 += qfb;
  }
  // ----Reaction 1: H2 + O = H + OH
  {
    real kfb = 5.08e+4 * exp(2.67 * logT - 6290 * iRcT);
    real qfb = kfb * c[0] * c[3];
    q1[1] = qfb;
    dest0 += qfb;
    dest3 += qfb;
    prod1 += qfb;
    prod4 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
    qfb = kfb * c[1] * c[4];
    q2[1] = qfb;
    dest1 += qfb;
    dest4 += qfb;
    prod0 += qfb;
    prod3 += qfb;
  }
  // ----Reaction 2: H2 + OH = H2O + H
  {
    real kfb = 2.16e+8 * exp(1.51 * logT - 3430 * iRcT);
    real qfb = kfb * c[0] * c[4];
    q1[2] = qfb;
    dest0 += qfb;
    dest4 += qfb;
    prod7 += qfb;
    prod1 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
    qfb = kfb * c[7] * c[1];
    q2[2] = qfb;
    dest7 += qfb;
    dest1 += qfb;
    prod0 += qfb;
    prod4 += qfb;
  }
  // ----Reaction 3: O + H2O = OH + OH
  {
    real kf = 2.97e+6 * exp(2.02 * logT - 13400 * iRcT);
    real qf = kf * c[3] * c[7];
    q1[3] = qf;
    dest3 += qf;
    dest7 += qf;
    prod4 += 2 * qf;
    kf *= exp(-gibbs_rt[3] - gibbs_rt[7] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[3] = qf;
    dest4 += 2 * qf;
    prod3 += qf;
    prod7 += qf;
  }
  // ----Reaction 4: H2 + M = H + H + M
  {
    real kf = 4.58e+19 * exp(-1.40 * logT - 1.0438e+5 * iRcT);
    kf *= cc;
    real qf = kf * c[0];
    q1[4] = qf;
    dest0 += qf;
    prod1 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
    qf = kf * c[1] * c[1];
    q2[4] = qf;
    dest1 += 2 * qf;
    prod0 += qf;
  }
  // ----Reaction 5: O + O + M = O2 + M
  {
    real kf = 6.16e+15 * sqrt(iT) * cc;
    real qf = kf * c[3] * c[3];
    q1[5] = qf;
    dest3 += 2 * qf;
    prod2 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
    qf = kf * c[2];
    q2[5] = qf;
    dest2 += qf;
    prod3 += 2 * qf;
  }
  // ----Reaction 6: O + H + M = OH + M
  {
    real kf = 4.71e+18 * iT * cc;
    real qf = kf * c[1] * c[3];
    q1[6] = qf;
    dest3 += qf;
    dest1 += qf;
    prod4 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[4];
    q2[6] = qf;
    dest4 += qf;
    prod3 += qf;
    prod1 += qf;
  }
  // ----Reaction 7: H + OH + M = H2O + M
  {
    real kf = 3.8e+22 * iT * iT * cc;
    real qf = kf * c[1] * c[4];
    q1[7] = qf;
    dest1 += qf;
    dest4 += qf;
    prod7 += qf;
    kf *= temp_t * exp(-gibbs_rt[1] - gibbs_rt[4] + gibbs_rt[7]);
    qf = kf * c[7];
    q2[7] = qf;
    dest7 += qf;
    prod1 += qf;
    prod4 += qf;
  }
  // ----Reaction 8: H + O2 (+M) = HO2 (+M)
  {
    const real kf_high = 1.48e+12 * exp(0.6 * logT);
    const real kf_low = 6.37e+20 * exp(-1.72 * logT - 5.2e+2 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real cc2 = c[0] * 2 + c[1] + c[2] * 0.78 + c[3] + c[4] + c[5] + c[6] + c[7] * 11 + c[8];
      const real reduced_pressure = kf_low * cc2 / kf_high;
      constexpr real logFc = -0.09691001300805639; // log10(0.8) = -9.691001300805639e-02
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[1] * c[2];
    q1[8] = qf;
    dest1 += qf;
    dest2 += qf;
    prod5 += qf;
    kf *= temp_t * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
    qf = kf * c[5];
    q2[8] = qf;
    dest5 += qf;
    prod1 += qf;
    prod2 += qf;
  }
  // ----Reaction 9: HO2 + H = H2 + O2
  {
    real kf = 1.66e+13 * exp(-820 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[9] = qf;
    dest5 += qf;
    dest1 += qf;
    prod0 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
    qf = kf * c[0] * c[2];
    q2[9] = qf;
    dest0 += qf;
    dest2 += qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 10: HO2 + H = OH + OH
  {
    real kf = 7.08e+13 * exp(-300 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[10] = qf;
    dest5 += qf;
    dest1 += qf;
    prod4 += 2 * qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[10] = qf;
    dest4 += 2 * qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 11: HO2 + O = O2 + OH
  {
    real qf = 3.25e+13 * c[5] * c[3];
    q1[11] = qf;
    dest5 += qf;
    dest3 += qf;
    prod2 += qf;
    prod4 += qf;
    const real kb = 3.25e+13 * exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
    qf = kb * c[2] * c[4];
    q2[11] = qf;
    dest2 += qf;
    dest4 += qf;
    prod5 += qf;
    prod3 += qf;
  }
  // ----Reaction 12: HO2 + OH = H2O + O2
  {
    real kf = 2.89e+13 * exp(500 * iRcT);
    real qf = kf * c[5] * c[4];
    q1[12] = qf;
    dest5 += qf;
    dest4 += qf;
    prod7 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
    qf = kf * c[7] * c[2];
    q2[12] = qf;
    dest7 += qf;
    dest2 += qf;
    prod5 += qf;
    prod4 += qf;
  }
  // ----Reaction 13: HO2 + HO2 = H2O2 + O2
  {
    real kf = 4.20e+14 * exp(-11980 * iRcT) + 1.30e+11 * exp(1630 * iRcT);
    real qf = kf * c[5] * c[5];
    q1[13] = qf;
    dest5 += 2 * qf;
    prod6 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
    qf = kf * c[6] * c[2];
    q2[13] = qf;
    dest6 += qf;
    dest2 += qf;
    prod5 += 2 * qf;
  }
  // ----Reaction 14: H2O2 (+ M) = OH + OH (+ M)
  {
    const real kf_high = 2.95e+14 * exp(-48400 * iRcT);
    const real kf_low = 1.20e+17 * exp(-45500 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real reduced_pressure = kf_low * cc / kf_high;
      constexpr real logFc = -3.010299956639812e-01; // log10(0.5) = -3.010299956639812e-01
      const real cT = -0.4 - 0.67 * logFc;
      const real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[6];
    q1[14] = qf;
    dest6 += qf;
    prod4 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[14] = qf;
    dest4 += 2 * qf;
    prod6 += qf;
  }
  // ----Reaction 15: H2O2 + H = H2O + OH
  {
    real kf = 2.41e+13 * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[15] = qf;
    dest6 += qf;
    dest1 += qf;
    prod7 += qf;
    prod4 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
    qf = kf * c[7] * c[4];
    q2[15] = qf;
    dest7 += qf;
    dest4 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ---Reaction 16: H2O2 + H = HO2 + H2
  {
    real kf = 4.82e+13 * exp(-7950 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[16] = qf;
    dest6 += qf;
    dest1 += qf;
    prod5 += qf;
    prod0 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
    qf = kf * c[5] * c[0];
    q2[16] = qf;
    dest5 += qf;
    dest0 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ----Reaction 17: H2O2 + O = OH + HO2
  {
    real kf = 9.55e+6 * t * t * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[3];
    q1[17] = qf;
    dest6 += qf;
    dest3 += qf;
    prod4 += qf;
    prod5 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
    qf = kf * c[4] * c[5];
    q2[17] = qf;
    dest4 += qf;
    dest5 += qf;
    prod6 += qf;
    prod3 += qf;
  }
  // ----Reaction 18: H2O2 + OH = HO2 + H2O
  {
    real kf = 1e+12 + 5.8e+14 * exp(-9560 * iRcT);
    real qf = kf * c[6] * c[4];
    q1[18] = qf;
    dest6 += qf;
    dest4 += qf;
    prod5 += qf;
    prod7 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
    qf = kf * c[5] * c[7];
    q2[18] = qf;
    dest5 += qf;
    dest7 += qf;
    prod6 += qf;
    prod4 += qf;
  }

  // Compute net production rates
  // Convert from mol/(cm3*s) to kg/(m3*s)
  omega_d[0] = dest0 * 2016;
  omega_d[1] = dest1 * 1008;
  omega_d[2] = dest2 * 31998;
  omega_d[3] = dest3 * 15999;
  omega_d[4] = dest4 * 17007;
  omega_d[5] = dest5 * 33006;
  omega_d[6] = dest6 * 34014;
  omega_d[7] = dest7 * 18015;
  omega_d[8] = dest8 * 28014;

  omega[0] = (prod0 - dest0) * 2016;
  omega[1] = (prod1 - dest1) * 1008;
  omega[2] = (prod2 - dest2) * 31998;
  omega[3] = (prod3 - dest3) * 15999;
  omega[4] = (prod4 - dest4) * 17007;
  omega[5] = (prod5 - dest5) * 33006;
  omega[6] = (prod6 - dest6) * 34014;
  omega[7] = (prod7 - dest7) * 18015;
  omega[8] = (prod8 - dest8) * 28014;
}

__device__ void chemical_source_hardCoded_burke_9s20r(real t, const real * __restrict__ c,
  const DParameter * __restrict__ param, real * __restrict__ omega_d, real * __restrict__ omega, real * __restrict__ q1,
  real * __restrict__ q2) {
  // According to chemistry/2011-Burke-Int.J.Chem-simp.inp
  // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2

  // G/RT for Kc
  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);

  // thermodynamic scaling for kc and kb
  const real temp_t = p_atm / R_u * 1e-3 / t; // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  // Arrhenius parameters
  const real iT = 1.0 / t;
  const real iRcT = 1.0 / R_c * iT;
  const real logT = log(t);

  // Third body concentrations used here
  const real cc = c[0] * 2.5 + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] * 12 + c[8];

  // Production / destruction in mol/(cm3*s)
  real prod0{0}, prod1{0}, prod2{0}, prod3{0}, prod4{0}, prod5{0}, prod6{0}, prod7{0}, prod8{0};
  real dest0{0}, dest1{0}, dest2{0}, dest3{0}, dest4{0}, dest5{0}, dest6{0}, dest7{0}, dest8{0};

  // ----Reaction 0: H + O2 = O + OH
  {
    real kfb = 1.04e+14 * exp(-1.5286e+4 * iRcT);
    real qfb = kfb * c[1] * c[2];
    q1[0] = qfb;
    dest1 += qfb;
    dest2 += qfb;
    prod3 += qfb;
    prod4 += qfb;
    kfb = kfb * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
    qfb = kfb * c[3] * c[4];
    q2[0] = qfb;
    dest3 += qfb;
    dest4 += qfb;
    prod1 += qfb;
    prod2 += qfb;
  }
  // ----Reaction 1: H2 + O = H + OH
  {
    real kfb = 3.818e+12 * exp(-7948 * iRcT) + 8.792e+14 * exp(-19170 * iRcT);
    real qfb = kfb * c[0] * c[3];
    q1[1] = qfb;
    dest0 += qfb;
    dest3 += qfb;
    prod1 += qfb;
    prod4 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
    qfb = kfb * c[1] * c[4];
    q2[1] = qfb;
    dest1 += qfb;
    dest4 += qfb;
    prod0 += qfb;
    prod3 += qfb;
  }
  // ----Reaction 2: H2 + OH = H2O + H
  {
    real kfb = 2.16e+8 * exp(1.51 * logT - 3430 * iRcT);
    real qfb = kfb * c[0] * c[4];
    q1[2] = qfb;
    dest0 += qfb;
    dest4 += qfb;
    prod7 += qfb;
    prod1 += qfb;
    kfb *= exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
    qfb = kfb * c[7] * c[1];
    q2[2] = qfb;
    dest7 += qfb;
    dest1 += qfb;
    prod0 += qfb;
    prod4 += qfb;
  }
  // ----Reaction 3: OH + OH = O + H2O
  {
    real kf = 3.34e+4 * exp(2.42 * logT + 1930 * iRcT);
    real qf = kf * c[4] * c[4];
    q1[3] = qf;
    dest4 += 2 * qf;
    prod3 += qf;
    prod7 += qf;
    kf *= exp(-gibbs_rt[4] - gibbs_rt[4] + gibbs_rt[3] + gibbs_rt[7]);
    qf = kf * c[3] * c[7];
    q2[3] = qf;
    dest3 += qf;
    dest7 += qf;
    prod4 += 2 * qf;
  }
  // ----Reaction 4: H2 + M = H + H + M
  {
    real kf = 4.577e+19 * exp(-1.40 * logT - 1.0438e+5 * iRcT);
    kf *= cc;
    real qf = kf * c[0];
    q1[4] = qf;
    dest0 += qf;
    prod1 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
    qf = kf * c[1] * c[1];
    q2[4] = qf;
    dest1 += 2 * qf;
    prod0 += qf;
  }
  // ----Reaction 5: O + O + M = O2 + M
  {
    real kf = 6.165e+15 * sqrt(iT) * cc;
    real qf = kf * c[3] * c[3];
    q1[5] = qf;
    dest3 += 2 * qf;
    prod2 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
    qf = kf * c[2];
    q2[5] = qf;
    dest2 += qf;
    prod3 += 2 * qf;
  }
  // ----Reaction 6: O + H + M = OH + M
  {
    real kf = 4.71e+18 * iT * cc;
    real qf = kf * c[1] * c[3];
    q1[6] = qf;
    dest3 += qf;
    dest1 += qf;
    prod4 += qf;
    kf *= temp_t * exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[4];
    q2[6] = qf;
    dest4 += qf;
    prod3 += qf;
    prod1 += qf;
  }
  // ----Reaction 7: H2O + M = H + OH + M
  {
    real kf = 6.064e+27 * exp(-3.322 * logT - 1.2079e+5 * iRcT)
              * (c[0] * 3 + c[1] + c[2] * 1.5 + c[3] + c[4] + c[5] + c[6] + c[8] * 2);
    real qf = kf * c[7];
    q1[7] = qf;
    dest7 += qf;
    prod1 += qf;
    prod4 += qf;
    kf *= iTemp_t * exp(-gibbs_rt[7] + gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[1] * c[4];
    q2[7] = qf;
    dest1 += qf;
    dest4 += qf;
    prod7 += qf;
  }
  // ----Reaction 8: H2O + H2O = H + OH + H2O
  {
    real kf = 1.006e+26 * exp(-2.44 * logT - 1.2018e+5 * iRcT);
    real qf = kf * c[7] * c[7];
    q1[8] = qf;
    dest7 += 2 * qf;
    prod1 += qf;
    prod4 += qf;
    prod7 += qf;
    kf *= iTemp_t * exp(-gibbs_rt[7] + gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[1] * c[4] * c[7];
    q2[8] = qf;
    dest1 += qf;
    dest4 += qf;
    dest7 += qf;
    prod7 += 2 * qf;
  }
  // ----Reaction 9: H + O2 (+M) = HO2 (+M)
  {
    const real kf_high = 4.65084e+12 * exp(0.44 * logT);
    const real kf_low = 6.366e+20 * exp(-1.72 * logT - 5.248e+2 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real cc2 = c[0] * 2 + c[1] + c[2] * 0.78 + c[3] + c[4] + c[5] + c[6] + c[7] * 14 + c[8];
      const real reduced_pressure = kf_low * cc2 / kf_high;
      constexpr real logFc = -0.3010299956639812; // log10(0.5) = -3.010299956639812e-01
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[1] * c[2];
    q1[9] = qf;
    dest1 += qf;
    dest2 += qf;
    prod5 += qf;
    kf *= temp_t * exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
    qf = kf * c[5];
    q2[9] = qf;
    dest5 += qf;
    prod1 += qf;
    prod2 += qf;
  }
  // ----Reaction 10: HO2 + H = H2 + O2
  {
    real kf = 2.75e+6 * exp(2.09 * logT + 1451 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[10] = qf;
    dest5 += qf;
    dest1 += qf;
    prod0 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
    qf = kf * c[0] * c[2];
    q2[10] = qf;
    dest0 += qf;
    dest2 += qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 11: HO2 + H = OH + OH
  {
    real kf = 7.079e+13 * exp(-295 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[11] = qf;
    dest5 += qf;
    dest1 += qf;
    prod4 += 2 * qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[11] = qf;
    dest4 += 2 * qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 12: HO2 + O = O2 + OH
  {
    real kf = 2.85e+10 * t * exp(723.93 * iRcT);
    real qf = kf * c[5] * c[3];
    q1[12] = qf;
    dest5 += qf;
    dest3 += qf;
    prod2 += qf;
    prod4 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
    qf = kf * c[2] * c[4];
    q2[12] = qf;
    dest2 += qf;
    dest4 += qf;
    prod5 += qf;
    prod3 += qf;
  }
  // ----Reaction 13: HO2 + OH = H2O + O2
  {
    real kf = 2.89e+13 * exp(497 * iRcT);
    real qf = kf * c[5] * c[4];
    q1[13] = qf;
    dest5 += qf;
    dest4 += qf;
    prod7 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
    qf = kf * c[7] * c[2];
    q2[13] = qf;
    dest7 += qf;
    dest2 += qf;
    prod5 += qf;
    prod4 += qf;
  }
  // ----Reaction 14: HO2 + HO2 = H2O2 + O2
  {
    real kf = 4.20e+14 * exp(-11982 * iRcT) + 1.30e+11 * exp(1629.3 * iRcT);
    real qf = kf * c[5] * c[5];
    q1[14] = qf;
    dest5 += 2 * qf;
    prod6 += qf;
    prod2 += qf;
    kf *= exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
    qf = kf * c[6] * c[2];
    q2[14] = qf;
    dest6 += qf;
    dest2 += qf;
    prod5 += 2 * qf;
  }
  // ----Reaction 15: H2O2 (+ M) = OH + OH (+ M)
  {
    const real kf_high = 2e+12 * exp(0.9 * logT - 48749 * iRcT);
    const real kf_low = 2.49e+24 * exp(-2.3 * logT - 48749 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2
      const real reduced_pressure = kf_low * (c[0] * 3.7 + c[1] + c[2] * 1.2 + c[3] + c[4] + c[5] + c[6] * 7.7
                                              + c[7] * 7.5 + c[8] * 1.5) / kf_high;
      constexpr real logFc = -3.665315444204135e-01; // log10(0.43) = -3.665315444204135e-01
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[6];
    q1[15] = qf;
    dest6 += qf;
    prod4 += 2 * qf;
    kf *= iTemp_t * exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[15] = qf;
    dest4 += 2 * qf;
    prod6 += qf;
  }
  // ----Reaction 16: H2O2 + H = H2O + OH
  {
    real kf = 2.41e+13 * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[16] = qf;
    dest6 += qf;
    dest1 += qf;
    prod7 += qf;
    prod4 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
    qf = kf * c[7] * c[4];
    q2[16] = qf;
    dest7 += qf;
    dest4 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ---Reaction 17: H2O2 + H = HO2 + H2
  {
    real kf = 4.82e+13 * exp(-7950 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[17] = qf;
    dest6 += qf;
    dest1 += qf;
    prod5 += qf;
    prod0 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
    qf = kf * c[5] * c[0];
    q2[17] = qf;
    dest5 += qf;
    dest0 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ----Reaction 18: H2O2 + O = OH + HO2
  {
    real kf = 9.55e+6 * t * t * exp(-3970 * iRcT);
    real qf = kf * c[6] * c[3];
    q1[18] = qf;
    dest6 += qf;
    dest3 += qf;
    prod4 += qf;
    prod5 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
    qf = kf * c[4] * c[5];
    q2[18] = qf;
    dest4 += qf;
    dest5 += qf;
    prod6 += qf;
    prod3 += qf;
  }
  // ----Reaction 19: H2O2 + OH = HO2 + H2O
  {
    real kf = 1.74e+12 * exp(-318 * iRcT) + 7.59e+13 * exp(-7270 * iRcT);
    real qf = kf * c[6] * c[4];
    q1[19] = qf;
    dest6 += qf;
    dest4 += qf;
    prod5 += qf;
    prod7 += qf;
    kf *= exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
    qf = kf * c[5] * c[7];
    q2[19] = qf;
    dest5 += qf;
    dest7 += qf;
    prod6 += qf;
    prod4 += qf;
  }

  // Compute net production rates
  // Convert from mol/(cm3*s) to kg/(m3*s)
  omega_d[0] = dest0 * 2016;
  omega_d[1] = dest1 * 1008;
  omega_d[2] = dest2 * 31998;
  omega_d[3] = dest3 * 15999;
  omega_d[4] = dest4 * 17007;
  omega_d[5] = dest5 * 33006;
  omega_d[6] = dest6 * 34014;
  omega_d[7] = dest7 * 18015;
  omega_d[8] = dest8 * 28014;

  omega[0] = (prod0 - dest0) * 2016;
  omega[1] = (prod1 - dest1) * 1008;
  omega[2] = (prod2 - dest2) * 31998;
  omega[3] = (prod3 - dest3) * 15999;
  omega[4] = (prod4 - dest4) * 17007;
  omega[5] = (prod5 - dest5) * 33006;
  omega[6] = (prod6 - dest6) * 34014;
  omega[7] = (prod7 - dest7) * 18015;
  omega[8] = (prod8 - dest8) * 28014;
}

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

__device__ void backward_reaction_rate_1(real t, const real *kf, real *kb, const DParameter *param) {
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
  real kb[MAX_REAC_NUMBER];
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];

  const int hardCodedMech = param->hardCodedMech;
  if (hardCodedMech == 1) {
    // compute the forward reaction rate
    forward_reaction_rate_1(t, kf, c);

    backward_reaction_rate_1(t, kf, kb, param);

    // compute the rate of progress
    rate_of_progress_1(kf, kb, c, q, q1, q2);

    // compute the chemical source
    chemical_source_1(q1, q2, omega_d, omega);
  } else {
    // generic mechanisms
    forward_reaction_rate(t, kf, c, param);

    backward_reaction_rate(t, kf, c, param, kb);

    // compute the rate of progress
    rate_of_progress(kf, kb, c, q, q1, q2, param);

    // compute the chemical source
    chemical_source(q1, q2, omega_d, omega, param);
  }

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

__device__ void backward_reaction_rate(real t, const real *kf, const real *concentration, const DParameter *param,
  real *kb) {
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

__device__ void rate_of_progress(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2,
  const DParameter *param) {
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

__device__ void compute_chem_src_jacobian(DZone *zone, int i, int j, int k, const DParameter *param, const real *q1,
  const real *q2) {
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

__device__ void compute_chem_src_jacobian_diagonal(DZone *zone, int i, int j, int k, const DParameter *param,
  const real *omega_d) {
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

__host__ __device__ void solve_chem_system(real *lhs, real *rhs, int dim) {
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

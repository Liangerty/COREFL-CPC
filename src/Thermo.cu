#include "Thermo.cuh"
#include "DParameter.cuh"
#include "Constants.h"
#include "ChemData.h"

#ifdef HighTempMultiPart
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        enthalpy[i] =
            coeff(0, 0, i) * tt + 0.5 * coeff(1, 0, i) * tt2 + coeff(2, 0, i) * tt3 / 3 + 0.25 * coeff(3, 0, i) * tt4 +
            0.2 * coeff(4, 0, i) * tt5 + coeff(5, 0, i);
        const real cp =
            coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] =
            coeff(0, j, i) * tt + 0.5 * coeff(1, j, i) * tt2 + coeff(2, j, i) * tt3 / 3 + 0.25 * coeff(3, j, i) * tt4 +
            0.2 * coeff(4, j, i) * tt5 + coeff(5, j, i);
        const real cp =
            coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real hMin = coeff(0, j - 1, i) * Tmin + 0.5 * coeff(1, j - 1, i) * Tmin * Tmin
                            + coeff(2, j - 1, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j - 1, i) * Tmin * Tmin *
                            Tmin * Tmin
                            + 0.2 * coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j - 1, i);
                real hMax = coeff(0, j, i) * Tmax + 0.5 * coeff(1, j, i) * Tmax * Tmax
                            + coeff(2, j, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j, i) * Tmax * Tmax * Tmax *
                            Tmax
                            + 0.2 * coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real hMin = coeff(0, j, i) * Tmin + 0.5 * coeff(1, j, i) * Tmin * Tmin
                            + coeff(2, j, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j, i) * Tmin * Tmin * Tmin *
                            Tmin
                            + 0.2 * coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j, i);
                real hMax = coeff(0, j + 1, i) * Tmax + 0.5 * coeff(1, j + 1, i) * Tmax * Tmax
                            + coeff(2, j + 1, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j + 1, i) * Tmax * Tmax *
                            Tmax * Tmax
                            + 0.2 * coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j + 1, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              enthalpy[i] =
                  coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4
                  +
                  0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
            }
            break;
          }
        }
      }
      enthalpy[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, lnT{log(t)};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        const real cp = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                        + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        const real cp = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                        + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            break;
          }
        }
      }
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        enthalpy[i] =
            coeff(0, 0, i) * tt + 0.5 * coeff(1, 0, i) * tt2 + coeff(2, 0, i) * tt3 / 3 + 0.25 * coeff(3, 0, i) * tt4 +
            0.2 * coeff(4, 0, i) * tt5 + coeff(5, 0, i);
        cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) *
                tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] =
            coeff(0, j, i) * tt + 0.5 * coeff(1, j, i) * tt2 + coeff(2, j, i) * tt3 / 3 + 0.25 * coeff(3, j, i) * tt4 +
            0.2 * coeff(4, j, i) * tt5 + coeff(5, j, i);
        cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) *
                tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real cpMin = coeff(0, j - 1, i) + coeff(1, j - 1, i) * Tmin + coeff(2, j - 1, i) * Tmin * Tmin
                             + coeff(3, j - 1, i) * Tmin * Tmin * Tmin + coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j, i) + coeff(1, j, i) * Tmax + coeff(2, j, i) * Tmax * Tmax
                             + coeff(3, j, i) * Tmax * Tmax * Tmax + coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);

                real hMin = coeff(0, j - 1, i) * Tmin + 0.5 * coeff(1, j - 1, i) * Tmin * Tmin
                            + coeff(2, j - 1, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j - 1, i) * Tmin * Tmin *
                            Tmin
                            * Tmin
                            + 0.2 * coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j - 1, i);
                real hMax = coeff(0, j, i) * Tmax + 0.5 * coeff(1, j, i) * Tmax * Tmax
                            + coeff(2, j, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j, i) * Tmax * Tmax * Tmax *
                            Tmax
                            + 0.2 * coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real cpMin = coeff(0, j, i) + coeff(1, j, i) * Tmin + coeff(2, j, i) * Tmin * Tmin
                             + coeff(3, j, i) * Tmin * Tmin * Tmin + coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j + 1, i) + coeff(1, j + 1, i) * Tmax + coeff(2, j + 1, i) * Tmax * Tmax
                             + coeff(3, j + 1, i) * Tmax * Tmax * Tmax + coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);

                real hMin = coeff(0, j, i) * Tmin + 0.5 * coeff(1, j, i) * Tmin * Tmin
                            + coeff(2, j, i) * Tmin * Tmin * Tmin / 3 + 0.25 * coeff(3, j, i) * Tmin * Tmin * Tmin *
                            Tmin
                            + 0.2 * coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin * Tmin + coeff(5, j, i);
                real hMax = coeff(0, j + 1, i) * Tmax + 0.5 * coeff(1, j + 1, i) * Tmax * Tmax
                            + coeff(2, j + 1, i) * Tmax * Tmax * Tmax / 3 + 0.25 * coeff(3, j + 1, i) * Tmax * Tmax *
                            Tmax
                            * Tmax
                            + 0.2 * coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax * Tmax + coeff(5, j + 1, i);
                enthalpy[i] = hMin + (t - Tmin) * (hMax - hMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              enthalpy[i] =
                  coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4
                  +
                  0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
              cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) *
                      t4;
            }
            break;
          }
        }
      }
      cp[i] *= R_u * param->imw[i];
      enthalpy[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, lnT{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_cp(real t, real *cp, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) *
                tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) *
                tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            bool com = false;
            if (j > 0) {
              if (param->temperature_cuts(i, j) + 200 > t) {
                real Tmin = param->temperature_cuts(i, j) - 200, Tmax = param->temperature_cuts(i, j) + 200;
                real cpMin = coeff(0, j - 1, i) + coeff(1, j - 1, i) * Tmin + coeff(2, j - 1, i) * Tmin * Tmin
                             + coeff(3, j - 1, i) * Tmin * Tmin * Tmin + coeff(4, j - 1, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j, i) + coeff(1, j, i) * Tmax + coeff(2, j, i) * Tmax * Tmax
                             + coeff(3, j, i) * Tmax * Tmax * Tmax + coeff(4, j, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);
                com = true;
              }
            } else if (j < param->n_temperature_range[i] - 1) {
              if (param->temperature_cuts(i, j + 1) - 200 < t) {
                real Tmin = param->temperature_cuts(i, j + 1) - 200, Tmax = param->temperature_cuts(i, j + 1) + 200;
                real cpMin = coeff(0, j, i) + coeff(1, j, i) * Tmin + coeff(2, j, i) * Tmin * Tmin
                             + coeff(3, j, i) * Tmin * Tmin * Tmin + coeff(4, j, i) * Tmin * Tmin * Tmin * Tmin;
                real cpMax = coeff(0, j + 1, i) + coeff(1, j + 1, i) * Tmax + coeff(2, j + 1, i) * Tmax * Tmax
                             + coeff(3, j + 1, i) * Tmax * Tmax * Tmax + coeff(4, j + 1, i) * Tmax * Tmax * Tmax * Tmax;
                cp[i] = cpMin + (t - Tmin) * (cpMax - cpMin) / (Tmax - Tmin);
                com = true;
              }
            }
            if (!com) {
              cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) *
                      t4;
            }
            break;
          }
        }
      }
      cp[i] *= R_u * param->imw[i];
    }
  } else {
    const real it{1.0 / t}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{log(t)};
  auto &coeff = param->therm_poly_coeff;
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        gibbs_rt[i] = coeff(0, 0, i) * (1.0 - log_tt) - 0.5 * coeff(1, 0, i) * tt - coeff(2, 0, i) * tt2 / 6.0 -
                      coeff(3, 0, i) * tt3 / 12.0 - coeff(4, 0, i) * tt4 * 0.05 + coeff(5, 0, i) * tt_inv -
                      coeff(6, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = coeff(0, j, i) * (1.0 - log_tt) - 0.5 * coeff(1, j, i) * tt - coeff(2, j, i) * tt2 / 6.0 -
                      coeff(3, j, i) * tt3 / 12.0 - coeff(4, j, i) * tt4 * 0.05 + coeff(5, j, i) * tt_inv -
                      coeff(6, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = coeff(0, j, i) * (1.0 - log_t) - 0.5 * coeff(1, j, i) * t - coeff(2, j, i) * t2 / 6.0 -
                          coeff(3, j, i) * t3 / 12.0 - coeff(4, j, i) * t4 * 0.05 + coeff(5, j, i) * t_inv -
                          coeff(6, j, i);
            break;
          }
        }
      }
    }
  } else {
    const real it{1.0 / t}, lnt{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        gibbs_rt[i] = -0.5 * coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt * (lntt - 1) +
                      coeff(2, 0, i) * (1.0 - lntt) - 0.5 * coeff(3, 0, i) * tt - coeff(4, 0, i) * tt2 / 6.0 -
                      coeff(5, 0, i) * tt3 / 12.0 - coeff(6, 0, i) * tt4 * 0.05 + coeff(7, 0, i) * itt -
                      coeff(8, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = -0.5 * coeff(0, j, i) * itt2 + coeff(1, j, i) * itt * (lntt - 1) +
                      coeff(2, j, i) * (1.0 - lntt) - 0.5 * coeff(3, j, i) * tt - coeff(4, j, i) * tt2 / 6.0 -
                      coeff(5, j, i) * tt3 / 12.0 - coeff(6, j, i) * tt4 * 0.05 + coeff(7, j, i) * itt -
                      coeff(8, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = -0.5 * coeff(0, j, i) * it2 + coeff(1, j, i) * it * (lnt - 1) +
                          coeff(2, j, i) * (1.0 - lnt) - 0.5 * coeff(3, j, i) * t - coeff(4, j, i) * t2 / 6.0 -
                          coeff(5, j, i) * t3 / 12.0 - coeff(6, j, i) * t4 * 0.05 + coeff(7, j, i) * it -
                          coeff(8, j, i);
            break;
          }
        }
      }
    }
  }
}
#else
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->low_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                      0.2 * coeff(i, 4) * t5 + coeff(i, 5);
      }
      enthalpy[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnT{log(t)};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        const real cp = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                        + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        const real cp = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                        + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            break;
          }
        }
      }
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const double tt = param->t_low[i];
        const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->low_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->t_high[i]) {
        const double tt = param->t_high[i];
        const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
        auto &coeff = param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                      0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                      0.2 * coeff(i, 4) * t5 + coeff(i, 5);
        cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
      }
      enthalpy[i] *= param->gas_const[i];
      cp[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnT{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        enthalpy[i] = -coeff(0, 0, i) * itt + coeff(1, 0, i) * log(tt) + coeff(2, 0, i) * tt
                      + 0.5 * coeff(3, 0, i) * tt2 + coeff(4, 0, i) * tt3 / 3 + 0.25 * coeff(5, 0, i) * tt4
                      + 0.2 * coeff(6, 0, i) * tt5 + coeff(7, 0, i);
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        enthalpy[i] = -coeff(0, j, i) * itt + coeff(1, j, i) * log(tt) + coeff(2, j, i) * tt
                      + 0.5 * coeff(3, j, i) * tt2 + coeff(4, j, i) * tt3 / 3 + 0.25 * coeff(5, j, i) * tt4
                      + 0.2 * coeff(6, j, i) * tt5 + coeff(7, j, i);
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
        enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            enthalpy[i] = -coeff(0, j, i) * it + coeff(1, j, i) * lnT + coeff(2, j, i) * t
                          + 0.5 * coeff(3, j, i) * t2 + coeff(4, j, i) * t3 / 3 + 0.25 * coeff(5, j, i) * t4
                          + 0.2 * coeff(6, j, i) * t5 + coeff(7, j, i);
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
      enthalpy[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_cp(real t, real *cp, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  if (param->nasa_7_or_9 == 7) {
    for (auto i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        auto &coeff = param->low_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
        auto &coeff = param->high_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      } else {
        auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
      }
      cp[i] *= param->gas_const[i];
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        cp[i] = coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt + coeff(2, 0, i)
                + coeff(3, 0, i) * tt + coeff(4, 0, i) * tt2 + coeff(5, 0, i) * tt3 + coeff(6, 0, i) * tt4;
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt,
            itt2 = 1.0 / tt2, itt = 1.0 / tt;
        const auto j = param->n_temperature_range[i] - 1;
        cp[i] = coeff(0, j, i) * itt2 + coeff(1, j, i) * itt + coeff(2, j, i)
                + coeff(3, j, i) * tt + coeff(4, j, i) * tt2 + coeff(5, j, i) * tt3 + coeff(6, j, i) * tt4;
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            cp[i] = coeff(0, j, i) * it2 + coeff(1, j, i) * it + coeff(2, j, i)
                    + coeff(3, j, i) * t + coeff(4, j, i) * t2 + coeff(5, j, i) * t3 + coeff(6, j, i) * t4;
            break;
          }
        }
      }
      cp[i] *= param->gas_const[i];
    }
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
  if (param->nasa_7_or_9 == 7) {
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->t_low[i]) {
        const real tt = param->t_low[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto &coeff = param->low_temp_coeff;
        gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                      coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
      } else if (t > param->t_high[i]) {
        const real tt = param->t_high[i];
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
        const auto &coeff = param->high_temp_coeff;
        gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                      coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
      } else {
        const auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
        gibbs_rt[i] =
            coeff(i, 0) * (1.0 - log_t) - 0.5 * coeff(i, 1) * t - coeff(i, 2) * t2 / 6.0 - coeff(i, 3) * t3 / 12.0 -
            coeff(i, 4) * t4 * 0.05 + coeff(i, 5) * t_inv - coeff(i, 6);
      }
    }
  } else {
    auto &coeff = param->therm_poly_coeff;
    const real it{1.0 / t}, lnt{log(t)}, it2{it * it};
    for (int i = 0; i < param->n_spec; ++i) {
      if (t < param->temperature_cuts(i, 0)) {
        const real tt = param->temperature_cuts(i, 0);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        gibbs_rt[i] = -0.5 * coeff(0, 0, i) * itt2 + coeff(1, 0, i) * itt * (lntt - 1) +
                      coeff(2, 0, i) * (1.0 - lntt) - 0.5 * coeff(3, 0, i) * tt - coeff(4, 0, i) * tt2 / 6.0 -
                      coeff(5, 0, i) * tt3 / 12.0 - coeff(6, 0, i) * tt4 * 0.05 + coeff(7, 0, i) * itt -
                      coeff(8, 0, i);
      } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
        const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
        const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, itt = 1 / tt, lntt = log(tt), itt2 = itt * itt;
        const auto j = param->n_temperature_range[i] - 1;
        gibbs_rt[i] = -0.5 * coeff(0, j, i) * itt2 + coeff(1, j, i) * itt * (lntt - 1) +
                      coeff(2, j, i) * (1.0 - lntt) - 0.5 * coeff(3, j, i) * tt - coeff(4, j, i) * tt2 / 6.0 -
                      coeff(5, j, i) * tt3 / 12.0 - coeff(6, j, i) * tt4 * 0.05 + coeff(7, j, i) * itt -
                      coeff(8, j, i);
      } else {
        for (int j = 0; j < param->n_temperature_range[i]; ++j) {
          if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
            gibbs_rt[i] = -0.5 * coeff(0, j, i) * it2 + coeff(1, j, i) * it * (lnt - 1) +
                          coeff(2, j, i) * (1.0 - lnt) - 0.5 * coeff(3, j, i) * t - coeff(4, j, i) * t2 / 6.0 -
                          coeff(5, j, i) * t3 / 12.0 - coeff(6, j, i) * t4 * 0.05 + coeff(7, j, i) * it -
                          coeff(8, j, i);
            break;
          }
        }
      }
    }
  }
}

#endif

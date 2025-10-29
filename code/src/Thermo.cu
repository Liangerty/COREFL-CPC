#include "Thermo.cuh"
#include "DParameter.cuh"
#include "Constants.h"
#include "ChemData.h"

#ifdef HighTempMultiPart
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const cfd::DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
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
          enthalpy[i] =
              coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4 +
              0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
          break;
        }
      }
    }
    enthalpy[i] *= cfd::R_u / param->mw[i];
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  auto &coeff = param->therm_poly_coeff;
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->temperature_cuts(i, 0)) {
      const real tt = param->temperature_cuts(i, 0);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      enthalpy[i] =
          coeff(0, 0, i) * tt + 0.5 * coeff(1, 0, i) * tt2 + coeff(2, 0, i) * tt3 / 3 + 0.25 * coeff(3, 0, i) * tt4 +
          0.2 * coeff(4, 0, i) * tt5 + coeff(5, 0, i);
      cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) * tt4;
      enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
    } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
      const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      const auto j = param->n_temperature_range[i] - 1;
      enthalpy[i] =
          coeff(0, j, i) * tt + 0.5 * coeff(1, j, i) * tt2 + coeff(2, j, i) * tt3 / 3 + 0.25 * coeff(3, j, i) * tt4 +
          0.2 * coeff(4, j, i) * tt5 + coeff(5, j, i);
      cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) * tt4;
      enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
    } else {
      for (int j = 0; j < param->n_temperature_range[i]; ++j) {
        if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
          enthalpy[i] =
              coeff(0, j, i) * t + 0.5 * coeff(1, j, i) * t2 + coeff(2, j, i) * t3 / 3 + 0.25 * coeff(3, j, i) * t4 +
              0.2 * coeff(4, j, i) * t5 + coeff(5, j, i);
          cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) * t4;
          break;
        }
      }
    }
    cp[i] *= R_u / param->mw[i];
    enthalpy[i] *= R_u / param->mw[i];
  }
}

__device__ void cfd::compute_cp(real t, real *cp, cfd::DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  auto &coeff = param->therm_poly_coeff;
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->temperature_cuts(i, 0)) {
      const real tt = param->temperature_cuts(i, 0);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      cp[i] = coeff(0, 0, i) + coeff(1, 0, i) * tt + coeff(2, 0, i) * tt2 + coeff(3, 0, i) * tt3 + coeff(4, 0, i) * tt4;
    } else if (t > param->temperature_cuts(i, param->n_temperature_range[i])) {
      const real tt = param->temperature_cuts(i, param->n_temperature_range[i]);
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      const auto j = param->n_temperature_range[i] - 1;
      cp[i] = coeff(0, j, i) + coeff(1, j, i) * tt + coeff(2, j, i) * tt2 + coeff(3, j, i) * tt3 + coeff(4, j, i) * tt4;
    } else {
      for (int j = 0; j < param->n_temperature_range[i]; ++j) {
        if (param->temperature_cuts(i, j) <= t && t <= param->temperature_cuts(i, j + 1)) {
          cp[i] = coeff(0, j, i) + coeff(1, j, i) * t + coeff(2, j, i) * t2 + coeff(3, j, i) * t3 + coeff(4, j, i) * t4;
          break;
        }
      }
    }
    cp[i] *= R_u / param->mw[i];
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const cfd::DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
  auto &coeff = param->therm_poly_coeff;
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
}
#else
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, const DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const real tt = param->t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = param->low_temp_coeff;
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
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const double tt = param->t_low[i];
      const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = param->low_temp_coeff;
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
}

__device__ void cfd::compute_cp(real t, real *cp, DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  for (auto i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const real tt = param->t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      auto &coeff = param->low_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
    } else {
      auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    cp[i] *= param->gas_const[i];
  }
}

__device__ void cfd::compute_gibbs_div_rt(real t, const DParameter *param, real *gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const real tt = param->t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
      const auto &coeff = param->low_temp_coeff;
      gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                    coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
    } else {
      const auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
      gibbs_rt[i] =
          coeff(i, 0) * (1.0 - log_t) - 0.5 * coeff(i, 1) * t - coeff(i, 2) * t2 / 6.0 - coeff(i, 3) * t3 / 12.0 -
          coeff(i, 4) * t4 * 0.05 + coeff(i, 5) * t_inv - coeff(i, 6);
    }
  }
}

// __device__ void cfd::compute_gibbs_div_rt_1(real t, real *gibbs_rt) {
//   const real tt = t;
//   if (t < 200) {
//     t = 200;
//   } else if (t > 3000) {
//     t = 3000;
//   }
//   const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t_v{1 - log(t)};
//   if (t < 1000) {
//     // H2
//     gibbs_rt[0] = 2.34433112 * log_t_v - 0.5 * 7.98052075E-03 * t + 1.94781510E-05 * t2 / 6.0 -
//                   2.01572094E-08 * t3 / 12.0 + 7.37611761E-12 * t4 * 0.05 - 9.17935173E+02 * t_inv - 6.83010238E-01;
//     // H
//     gibbs_rt[1] = 2.50000000 * log_t_v - 0.5 * 7.05332819E-13 * t + 1.99591964E-15 * t2 / 6.0 -
//                   2.30081632E-18 * t3 / 12.0 + 9.27732332E-22 * t4 * 0.05 + 2.54736599E+04 * t_inv + 4.46682853E-01;
//     // O2
//     gibbs_rt[2] = 3.78245636 * log_t_v + 0.5 * 2.99673416E-03 * t - 9.84730201E-06 * t2 / 6.0 +
//                   9.68129509E-09 * t3 / 12.0 - 3.24372837E-12 * t4 * 0.05 - 1.06394356E+03 * t_inv - 3.65767573;
//     // O
//     gibbs_rt[3] = 3.16826710 * log_t_v + 0.5 * 3.27931884E-03 * t - 6.64306396E-06 * t2 / 6.0 +
//                   6.12806624E-09 * t3 / 12.0 - 2.11265971E-12 * t4 * 0.05 + 2.91222592E+04 * t_inv - 2.05193346;
//     // OH
//     gibbs_rt[4] = 3.99201543 * log_t_v + 0.5 * 2.40131752E-03 * t - 4.61793841E-06 * t2 / 6.0 +
//                   3.88113333E-09 * t3 / 12.0 - 1.36411470E-12 * t4 * 0.05 + 3.61508056E+03 * t_inv + 1.03925458E-01;
//     // HO2
//     gibbs_rt[5] = 4.30179801 * log_t_v + 0.5 * 4.74912051E-03 * t - 2.11582891E-05 * t2 / 6.0 +
//                   2.42763894E-08 * t3 / 12.0 - 9.29225124E-12 * t4 * 0.05 + 2.94808040E+02 * t_inv - 3.71666245;
//     // H2O2
//     gibbs_rt[6] = 4.27611269 * log_t_v + 0.5 * 5.42822417E-04 * t - 1.67335701E-05 * t2 / 6.0 +
//                   2.15770813E-08 * t3 / 12.0 - 8.62454363E-12 * t4 * 0.05 - 1.77025821E+04 * t_inv - 3.43505074;
//     // H2O
//     gibbs_rt[7] = 4.19864056 * log_t_v + 0.5 * 2.03643410E-03 * t - 6.52040211E-06 * t2 / 6.0 +
//                   5.48797062E-09 * t3 / 12.0 - 1.77197817E-12 * t4 * 0.05 - 3.02937267E+04 * t_inv + 0.849032208;
//   } else {
//     // H2
//     gibbs_rt[0] = 3.3372792 * log_t_v + 0.5 * 4.94024731E-05 * t - 4.99456778E-07 * t2 / 6.0 +
//                   1.79566394E-10 * t3 / 12.0 - 2.00255376E-14 * t4 * 0.05 - 9.50158922E+02 * t_inv + 3.20502331;
//     // H
//     gibbs_rt[1] = 2.50000000 * log_t_v - 0.5 * 2.30842973E-11 * t - 1.61561948E-14 * t2 / 6.0 +
//                   4.73515235E-18 * t3 / 12.0 - 4.98197357E-22 * t4 * 0.05 + 2.54736599E+04 * t_inv + 4.46682914E-01;
//     // O2
//     gibbs_rt[2] = 3.28253784 * log_t_v - 0.5 * 1.48308754E-03 * t + 7.57966669E-07 * t2 / 6.0 -
//                   2.09470555E-10 * t3 / 12.0 + 2.16717794E-14 * t4 * 0.05 - 1.08845772E+03 * t_inv - 5.45323129;
//     // O
//     gibbs_rt[3] = 2.56942078 * log_t_v + 0.5 * 8.59741137E-05 * t - 4.19484589E-08 * t2 / 6.0 +
//                   1.00177799E-11 * t3 / 12.0 - 1.22833691E-15 * t4 * 0.05 + 2.92175791E+04 * t_inv - 4.78433864;
//     // OH
//     gibbs_rt[4] = 3.09288767 * log_t_v - 0.5 * 5.48429716E-04 * t - 1.26505228E-07 * t2 / 6.0 +
//                   8.79461556E-11 * t3 / 12.0 - 1.17412376E-14 * t4 * 0.05 + 3.85865700E+03 * t_inv - 4.4766961;
//     // HO2
//     gibbs_rt[5] = 4.01721090 * log_t_v - 0.5 * 2.23982013E-03 * t + 6.33658150E-07 * t2 / 6.0 -
//                   1.14246370E-10 * t3 / 12.0 + 1.07908535E-14 * t4 * 0.05 + 1.11856713E+02 * t_inv - 3.78510215;
//     // H2O2
//     gibbs_rt[6] = 4.16500285 * log_t_v - 0.5 * 4.90831694E-03 * t + 1.90139225E-06 * t2 / 6.0 -
//                   3.71185986E-10 * t3 / 12.0 + 2.87908305E-14 * t4 * 0.05 - 1.78617877E+04 * t_inv - 2.91615662;
//     // H2O
//     gibbs_rt[7] = 3.03399249 * log_t_v - 0.5 * 2.17691804E-03 * t + 1.64072518E-07 * t2 / 6.0 +
//                   9.70419870E-11 * t3 / 12.0 - 1.68200992E-14 * t4 * 0.05 - 3.00042971E+04 * t_inv - 4.96677010;
//   }
//   if (tt > 3000 && tt <= 5000) {
//     // N2 is still in high-temperature part
//     const real tt2{tt * tt}, tt3{t2 * tt}, tt4{t3 * tt}, tt_inv{1 / tt}, log_tt_v{1 - log(tt)};
//     gibbs_rt[8] = 2.92664 * log_tt_v - 0.5 * 0.14879768E-02 * tt + 0.05684760E-05 * tt2 / 6.0 -
//                   0.10097038E-09 * tt3 / 12.0 + 0.06753351E-13 * tt4 * 0.05 - 0.09227977E+04 * tt_inv - 5.98052;
//   } else if (t < 1000) {
//     // N2
//     gibbs_rt[8] = 3.298677 * log_t_v - 0.5 * 0.14082404E-02 * t + 0.03963222E-04 * t2 / 6.0 -
//                   0.05641515E-07 * t3 / 12.0 + 0.02444854E-10 * t4 * 0.05 - 0.10208999E+04 * t_inv - 3.950372;
//   } else {
//     // N2
//     gibbs_rt[8] = 2.92664 * log_t_v - 0.5 * 0.14879768E-02 * t + 0.05684760E-05 * t2 / 6.0 -
//                   0.10097038E-09 * t3 / 12.0 + 0.06753351E-13 * t4 * 0.05 - 0.09227977E+04 * t_inv - 5.980528;
//   }
// }
//
// __device__ void cfd::compute_enthalpy_and_cp_1(real t, real *enthalpy, real *cp, const DParameter *param) {
//   const real tt = t;
//   if (t < 200) {
//     t = 200;
//     constexpr real t2{200 * 200}, t3{t2 * 200}, t4{t3 * 200}, t5{t4 * 200};
//     // H2
//     enthalpy[0] = 2.34433112 * t + 0.5 * 7.98052075E-03 * t2 - 1.94781510E-05 * t3 / 3 + 0.25 * 2.01572094E-08 * t4 -
//                   0.2 * 7.37611761E-12 * t5 - 9.17935173E+02;
//     cp[0] = 2.34433112 + 7.98052075E-03 * t - 1.94781510E-05 * t2 + 2.01572094E-08 * t3 - 7.37611761E-12 * t4;
//     enthalpy[0] += cp[0] * (t - 200);
//     // H
//     enthalpy[1] = 2.5 * t + 0.5 * 7.05332819E-13 * t2 - 1.99591964E-15 * t3 / 3 + 0.25 * 2.30081632E-18 * t4 -
//                   0.2 * 9.27732332E-22 * t5 + 2.54736599E+04;
//     cp[1] = 2.5 + 7.05332819E-13 * t - 1.99591964E-15 * t2 + 2.30081632E-18 * t3 - 9.27732332E-22 * t4;
//     enthalpy[1] += cp[1] * (t - 200);
//     // O2
//     enthalpy[2] = 3.78245636 * t - 0.5 * 2.99673416E-03 * t2 + 9.84730201E-06 * t3 / 3 - 0.25 * 9.68129509E-09 * t4 +
//                   0.2 * 3.24372837E-12 * t5 - 1.06394356E+03;
//     cp[2] = 3.78245636 - 2.99673416E-03 * t + 9.84730201E-06 * t2 - 9.68129509E-09 * t3 + 3.24372837E-12 * t4;
//     enthalpy[2] += cp[2] * (t - 200);
//     // O
//     enthalpy[3] = 3.16826710 * t - 0.5 * 3.27931884E-03 * t2 + 6.64306396E-06 * t3 / 3 - 0.25 * 6.12806624E-09 * t4 +
//                   0.2 * 2.11265971E-12 * t5 + 2.91222592E+04;
//     cp[3] = 3.16826710 - 3.27931884E-03 * t + 6.64306396E-06 * t2 - 6.12806624E-09 * t3 + 2.11265971E-12 * t4;
//     enthalpy[3] += cp[3] * (t - 200);
//     // OH
//     enthalpy[4] = 3.99201543 * t - 0.5 * 2.40131752E-03 * t2 + 4.61793841E-06 * t3 / 3 - 0.25 * 3.88113333E-09 * t4 +
//                   0.2 * 1.36411470E-12 * t5 + 3.61508056E+03;
//     cp[4] = 3.99201543 - 2.40131752E-03 * t + 4.61793841E-06 * t2 - 3.88113333E-09 * t3 + 1.36411470E-12 * t4;
//     enthalpy[4] += cp[4] * (t - 200);
//     // HO2
//     enthalpy[5] = 4.30179801 * t - 0.5 * 4.74912051E-03 * t2 + 2.11582891E-05 * t3 / 3 -
//                   0.25 * 2.42763894E-08 * t4 + 0.2 * 9.29225124E-12 * t5 + 2.94808040E+02;
//     cp[5] = 4.30179801 - 4.74912051E-03 * t + 2.11582891E-05 * t2 - 2.42763894E-08 * t3 + 9.29225124E-12 * t4;
//     enthalpy[5] += cp[5] * (t - 200);
//     // H2O2
//     enthalpy[6] = 4.27611269 * t - 0.5 * 5.42822417E-04 * t2 + 1.67335701E-05 * t3 / 3 -
//                   0.25 * 2.15770813E-08 * t4 + 0.2 * 8.62454363E-12 * t5 - 1.77025821E+04;
//     cp[6] = 4.27611269 - 5.42822417E-04 * t + 1.67335701E-05 * t2 - 2.15770813E-08 * t3 + 8.62454363E-12 * t4;
//     enthalpy[6] += cp[6] * (t - 200);
//     // H2O
//     enthalpy[7] = 4.19864056 * t - 0.5 * 2.03643410E-03 * t2 + 6.52040211E-06 * t3 / 3 -
//                   0.25 * 5.48797062E-09 * t4 + 0.2 * 1.77197817E-12 * t5 - 3.02937267E+04;
//     cp[7] = 4.19864056 - 2.03643410E-03 * t + 6.52040211E-06 * t2 - 5.48797062E-09 * t3 + 1.77197817E-12 * t4;
//     enthalpy[7] += cp[7] * (t - 200);
//     // N2
//     enthalpy[8] = 3.298677 * 300 + 0.5 * 0.14082404E-02 * 90000 - 0.03963222E-04 * 2.7e+7 / 3 +
//                   0.25 * 0.05641515E-07 * 8.1e+9 - 0.2 * 0.02444854E-10 * 2.43e+12 - 1.02089999E+03;
//     cp[8] = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//             8.1e+9;
//     enthalpy[8] += cp[8] * (t - 300);
//   } else if (t > 3000) {
//     t = 3000;
//     constexpr real t2{9e+6}, t3{2.7e+10}, t4{8.1e+13}, t5{2.43e+17};
//     // H2
//     enthalpy[0] = 3.33727920 * t - 0.5 * 4.94024731E-05 * t2 + 4.99456778E-07 * t3 / 3 -
//                   0.25 * 1.79566394E-10 * t4 + 0.2 * 2.00255376E-14 * t5 - 9.50158922E+02;
//     cp[0] = 3.33727920 - 4.94024731E-05 * t + 4.99456778E-07 * t2 - 1.79566394E-10 * t3 + 2.00255376E-14 * t4;
//     enthalpy[0] += cp[0] * (t - 3000); // Do a linear interpolation for enthalpy
//     // H
//     enthalpy[1] = 2.50000001 * t - 0.5 * 2.30842973E-11 * t2 + 1.61561948E-14 * t3 / 3 -
//                   0.25 * 4.73515235E-18 * t4 + 0.2 * 4.98197357E-22 * t5 + 2.54736599E+04;
//     cp[1] = 2.50000001 - 2.30842973E-11 * t + 1.61561948E-14 * t2 - 4.73515235E-18 * t3 + 4.98197357E-22 * t4;
//     enthalpy[1] += cp[1] * (t - 3000);
//     // O2
//     enthalpy[2] = 3.28253784 * t + 0.5 * 1.48308754E-03 * t2 - 7.57966669E-07 * t3 / 3 +
//                   0.25 * 2.09470555E-10 * t4 - 0.2 * 2.16717794E-14 * t5 - 1.08845772E+03;
//     cp[2] = 3.28253784 + 1.48308754E-03 * t - 7.57966669E-07 * t2 + 2.09470555E-10 * t3 - 2.16717794E-14 * t4;
//     enthalpy[2] += cp[2] * (t - 3000);
//     // O
//     enthalpy[3] = 2.56942078 * t - 0.5 * 8.59741137E-05 * t2 + 4.19484589E-08 * t3 / 3 -
//                   0.25 * 1.00177799E-11 * t4 + 0.2 * 1.22833691E-15 * t5 + 2.92175791E+04;
//     cp[3] = 2.56942078 - 8.59741137E-05 * t + 4.19484589E-08 * t2 - 1.00177799E-11 * t3 + 1.22833691E-15 * t4;
//     enthalpy[3] += cp[3] * (t - 3000);
//     // OH
//     enthalpy[4] = 3.09288767 * t + 0.5 * 5.48429716E-04 * t2 + 1.26505228E-07 * t3 / 3 -
//                   0.25 * 8.79461556E-11 * t4 + 0.2 * 1.17412376E-14 * t5 + 3.85865700E+03;
//     cp[4] = 3.09288767 + 5.48429716E-04 * t + 1.26505228E-07 * t2 - 8.79461556E-11 * t3 + 1.17412376E-14 * t4;
//     enthalpy[4] += cp[4] * (t - 3000);
//     // HO2
//     enthalpy[5] = 4.01721090 * t + 0.5 * 2.23982013E-03 * t2 - 6.33658150E-07 * t3 / 3 +
//                   0.25 * 1.14246370E-10 * t4 - 0.2 * 1.07908535E-14 * t5 + 1.11856713E+02;
//     cp[5] = 4.01721090 + 2.23982013E-03 * t - 6.33658150E-07 * t2 + 1.14246370E-10 * t3 - 1.07908535E-14 * t4;
//     enthalpy[5] += cp[5] * (t - 3000);
//     // H2O2
//     enthalpy[6] = 4.16500285 * t + 0.5 * 4.90831694E-03 * t2 - 1.90139225E-06 * t3 / 3 +
//                   0.25 * 3.71185986E-10 * t4 - 0.2 * 2.87908305E-14 * t5 - 1.78617877E+04;
//     cp[6] = 4.16500285 + 4.90831694E-03 * t - 1.90139225E-06 * t2 + 3.71185986E-10 * t3 - 2.87908305E-14 * t4;
//     enthalpy[6] += cp[6] * (t - 3000);
//     // H2O
//     enthalpy[7] = 3.03399249 * t + 0.5 * 2.17691804E-03 * t2 - 1.64072518E-07 * t3 / 3 -
//                   0.25 * 9.70419870E-11 * t4 + 0.2 * 1.68200992E-14 * t5 - 3.00042971E+04;
//     cp[7] = 3.03399249 + 2.17691804E-03 * t - 1.64072518E-07 * t2 - 9.70419870E-11 * t3 + 1.68200992E-14 * t4;
//     enthalpy[7] += cp[7] * (t - 3000);
//     if (tt < 5000) {
//       // N2
//       const real tt2{tt * tt}, tt3{t2 * tt}, tt4{t3 * tt}, tt5{tt4 * tt};
//       enthalpy[8] = 2.92664 * tt + 0.5 * 0.14879768E-02 * tt2 - 0.05684760E-05 * tt3 / 3 +
//                     0.25 * 0.10097038E-09 * tt4 - 0.2 * 0.06753351E-13 * tt5 - 0.09227977E+04;
//       cp[8] = 2.92664 + 0.14879768E-02 * tt - 0.05684760E-05 * tt2 + 0.10097038E-09 * tt3 - 0.06753351E-13 * tt4;
//     } else {
//       enthalpy[8] = 2.92664 * 5000 + 0.5 * 0.14879768E-02 * 2.5e+7 - 0.05684760E-05 * 1.25e+11 / 3 +
//                     0.25 * 0.10097038E-09 * 6.25e+14 - 0.2 * 0.06753351E-13 * 3.125e+16 - 0.09227977E+04;
//       cp[8] = 2.92664 + 0.14879768E-02 * 5000 - 0.05684760E-05 * 2.5e+7 + 0.10097038E-09 * 1.25e+11 - 0.06753351E-13 *
//               6.25e+14;
//       enthalpy[8] += cp[8] * (t - 5000);
//     }
//   } else {
//     const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
//     if (t < 1000) {
//       // H2
//       enthalpy[0] = 2.34433112 * t + 0.5 * 7.98052075E-03 * t2 - 1.94781510E-05 * t3 / 3 + 0.25 * 2.01572094E-08 * t4 -
//                     0.2 * 7.37611761E-12 * t5 - 9.17935173E+02;
//       cp[0] = 2.34433112 + 7.98052075E-03 * t - 1.94781510E-05 * t2 + 2.01572094E-08 * t3 - 7.37611761E-12 * t4;
//       // H
//       enthalpy[1] = 2.5 * t + 0.5 * 7.05332819E-13 * t2 - 1.99591964E-15 * t3 / 3 + 0.25 * 2.30081632E-18 * t4 -
//                     0.2 * 9.27732332E-22 * t5 + 2.54736599E+04;
//       cp[1] = 2.5 + 7.05332819E-13 * t - 1.99591964E-15 * t2 + 2.30081632E-18 * t3 - 9.27732332E-22 * t4;
//       // O2
//       enthalpy[2] = 3.78245636 * t - 0.5 * 2.99673416E-03 * t2 + 9.84730201E-06 * t3 / 3 - 0.25 * 9.68129509E-09 * t4 +
//                     0.2 * 3.24372837E-12 * t5 - 1.06394356E+03;
//       cp[2] = 3.78245636 - 2.99673416E-03 * t + 9.84730201E-06 * t2 - 9.68129509E-09 * t3 + 3.24372837E-12 * t4;
//       // O
//       enthalpy[3] = 3.16826710 * t - 0.5 * 3.27931884E-03 * t2 + 6.64306396E-06 * t3 / 3 - 0.25 * 6.12806624E-09 * t4 +
//                     0.2 * 2.11265971E-12 * t5 + 2.91222592E+04;
//       cp[3] = 3.16826710 - 3.27931884E-03 * t + 6.64306396E-06 * t2 - 6.12806624E-09 * t3 + 2.11265971E-12 * t4;
//       // OH
//       enthalpy[4] = 3.99201543 * t - 0.5 * 2.40131752E-03 * t2 + 4.61793841E-06 * t3 / 3 - 0.25 * 3.88113333E-09 * t4 +
//                     0.2 * 1.36411470E-12 * t5 + 3.61508056E+03;
//       cp[4] = 3.99201543 - 2.40131752E-03 * t + 4.61793841E-06 * t2 - 3.88113333E-09 * t3 + 1.36411470E-12 * t4;
//       // HO2
//       enthalpy[5] = 4.30179801 * t - 0.5 * 4.74912051E-03 * t2 + 2.11582891E-05 * t3 / 3 -
//                     0.25 * 2.42763894E-08 * t4 + 0.2 * 9.29225124E-12 * t5 + 2.94808040E+02;
//       cp[5] = 4.30179801 - 4.74912051E-03 * t + 2.11582891E-05 * t2 - 2.42763894E-08 * t3 + 9.29225124E-12 * t4;
//       // H2O2
//       enthalpy[6] = 4.27611269 * t - 0.5 * 5.42822417E-04 * t2 + 1.67335701E-05 * t3 / 3 -
//                     0.25 * 2.15770813E-08 * t4 + 0.2 * 8.62454363E-12 * t5 - 1.77025821E+04;
//       cp[6] = 4.27611269 - 5.42822417E-04 * t + 1.67335701E-05 * t2 - 2.15770813E-08 * t3 + 8.62454363E-12 * t4;
//       // H2O
//       enthalpy[7] = 4.19864056 * t - 0.5 * 2.03643410E-03 * t2 + 6.52040211E-06 * t3 / 3 -
//                     0.25 * 5.48797062E-09 * t4 + 0.2 * 1.77197817E-12 * t5 - 3.02937267E+04;
//       cp[7] = 4.19864056 - 2.03643410E-03 * t + 6.52040211E-06 * t2 - 5.48797062E-09 * t3 + 1.77197817E-12 * t4;
//       if (t < 300) {
//         // N2
//         enthalpy[8] = 3.298677 * 300 + 0.5 * 0.14082404E-02 * 90000 - 0.03963222E-04 * 2.7e+7 / 3 +
//                       0.25 * 0.05641515E-07 * 8.1e+9 - 0.2 * 0.02444854E-10 * 2.43e+12 - 1.02089999E+03;
//         cp[8] = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//                 8.1e+9;
//         enthalpy[8] += cp[8] * (t - 300);
//       } else {
//         // N2
//         enthalpy[8] = 3.298677 * t + 0.5 * 0.14082404E-02 * t2 - 0.03963222E-04 * t3 / 3 +
//                       0.25 * 0.05641515E-07 * t4 - 0.2 * 0.02444854E-10 * t5 - 1.02089999E+03;
//         cp[8] = 3.298677 + 0.14082404E-02 * t - 0.03963222E-04 * t2 + 0.05641515E-07 * t3 - 0.02444854E-10 * t4;
//       }
//     } else {
//       // H2
//       enthalpy[0] = 3.33727920 * t - 0.5 * 4.94024731E-05 * t2 + 4.99456778E-07 * t3 / 3 -
//                     0.25 * 1.79566394E-10 * t4 + 0.2 * 2.00255376E-14 * t5 - 9.50158922E+02;
//       cp[0] = 3.33727920 - 4.94024731E-05 * t + 4.99456778E-07 * t2 - 1.79566394E-10 * t3 + 2.00255376E-14 * t4;
//       // H
//       enthalpy[1] = 2.50000001 * t - 0.5 * 2.30842973E-11 * t2 + 1.61561948E-14 * t3 / 3 -
//                     0.25 * 4.73515235E-18 * t4 + 0.2 * 4.98197357E-22 * t5 + 2.54736599E+04;
//       cp[1] = 2.50000001 - 2.30842973E-11 * t + 1.61561948E-14 * t2 - 4.73515235E-18 * t3 + 4.98197357E-22 * t4;
//       // O2
//       enthalpy[2] = 3.28253784 * t + 0.5 * 1.48308754E-03 * t2 - 7.57966669E-07 * t3 / 3 +
//                     0.25 * 2.09470555E-10 * t4 - 0.2 * 2.16717794E-14 * t5 - 1.08845772E+03;
//       cp[2] = 3.28253784 + 1.48308754E-03 * t - 7.57966669E-07 * t2 + 2.09470555E-10 * t3 - 2.16717794E-14 * t4;
//       // O
//       enthalpy[3] = 2.56942078 * t - 0.5 * 8.59741137E-05 * t2 + 4.19484589E-08 * t3 / 3 -
//                     0.25 * 1.00177799E-11 * t4 + 0.2 * 1.22833691E-15 * t5 + 2.92175791E+04;
//       cp[3] = 2.56942078 - 8.59741137E-05 * t + 4.19484589E-08 * t2 - 1.00177799E-11 * t3 + 1.22833691E-15 * t4;
//       // OH
//       enthalpy[4] = 3.09288767 * t + 0.5 * 5.48429716E-04 * t2 + 1.26505228E-07 * t3 / 3 -
//                     0.25 * 8.79461556E-11 * t4 + 0.2 * 1.17412376E-14 * t5 + 3.85865700E+03;
//       cp[4] = 3.09288767 + 5.48429716E-04 * t + 1.26505228E-07 * t2 - 8.79461556E-11 * t3 + 1.17412376E-14 * t4;
//       // HO2
//       enthalpy[5] = 4.01721090 * t + 0.5 * 2.23982013E-03 * t2 - 6.33658150E-07 * t3 / 3 +
//                     0.25 * 1.14246370E-10 * t4 - 0.2 * 1.07908535E-14 * t5 + 1.11856713E+02;
//       cp[5] = 4.01721090 + 2.23982013E-03 * t - 6.33658150E-07 * t2 + 1.14246370E-10 * t3 - 1.07908535E-14 * t4;
//       // H2O2
//       enthalpy[6] = 4.16500285 * t + 0.5 * 4.90831694E-03 * t2 - 1.90139225E-06 * t3 / 3 +
//                     0.25 * 3.71185986E-10 * t4 - 0.2 * 2.87908305E-14 * t5 - 1.78617877E+04;
//       cp[6] = 4.16500285 + 4.90831694E-03 * t - 1.90139225E-06 * t2 + 3.71185986E-10 * t3 - 2.87908305E-14 * t4;
//       // H2O
//       enthalpy[7] = 3.03399249 * t + 0.5 * 2.17691804E-03 * t2 - 1.64072518E-07 * t3 / 3 -
//                     0.25 * 9.70419870E-11 * t4 + 0.2 * 1.68200992E-14 * t5 - 3.00042971E+04;
//       cp[7] = 3.03399249 + 2.17691804E-03 * t - 1.64072518E-07 * t2 - 9.70419870E-11 * t3 + 1.68200992E-14 * t4;
//       // N2
//       enthalpy[8] = 2.92664 * t + 0.5 * 0.14879768E-02 * t2 - 0.05684760E-05 * t3 / 3 +
//                     0.25 * 0.10097038E-09 * t4 - 0.2 * 0.06753351E-13 * t5 - 0.09227977E+04;
//       cp[8] = 2.92664 + 0.14879768E-02 * t - 0.05684760E-05 * t2 + 0.10097038E-09 * t3 - 0.06753351E-13 * t4;
//     }
//   }
//   for (int i = 0; i < param->n_spec; ++i) {
//     enthalpy[i] *= param->gas_const[i];
//     cp[i] *= param->gas_const[i];
//   }
// }
//
// __device__ void cfd::compute_cp_1(real t, real *cp, DParameter *param) {
//   const real tt = t;
//   if (t < 200) {
//     t = 200;
//     constexpr real t2{200 * 200}, t3{t2 * 200}, t4{t3 * 200};
//     // H2
//     cp[0] = 2.34433112 + 7.98052075E-03 * t - 1.94781510E-05 * t2 + 2.01572094E-08 * t3 - 7.37611761E-12 * t4;
//     // H
//     cp[1] = 2.5 + 7.05332819E-13 * t - 1.99591964E-15 * t2 + 2.30081632E-18 * t3 - 9.27732332E-22 * t4;
//     // O2
//     cp[2] = 3.78245636 - 2.99673416E-03 * t + 9.84730201E-06 * t2 - 9.68129509E-09 * t3 + 3.24372837E-12 * t4;
//     // O
//     cp[3] = 3.16826710 - 3.27931884E-03 * t + 6.64306396E-06 * t2 - 6.12806624E-09 * t3 + 2.11265971E-12 * t4;
//     // OH
//     cp[4] = 3.99201543 - 2.40131752E-03 * t + 4.61793841E-06 * t2 - 3.88113333E-09 * t3 + 1.36411470E-12 * t4;
//     // HO2
//     cp[5] = 4.30179801 - 4.74912051E-03 * t + 2.11582891E-05 * t2 - 2.42763894E-08 * t3 + 9.29225124E-12 * t4;
//     // H2O2
//     cp[6] = 4.27611269 - 5.42822417E-04 * t + 1.67335701E-05 * t2 - 2.15770813E-08 * t3 + 8.62454363E-12 * t4;
//     // H2O
//     cp[7] = 4.19864056 - 2.03643410E-03 * t + 6.52040211E-06 * t2 - 5.48797062E-09 * t3 + 1.77197817E-12 * t4;
//     // N2
//     cp[8] = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//             8.1e+9;
//   } else if (t > 3000) {
//     t = 3000;
//     constexpr real t2{9e+6}, t3{2.7e+10}, t4{8.1e+13};
//     // H2
//     cp[0] = 3.33727920 - 4.94024731E-05 * t + 4.99456778E-07 * t2 - 1.79566394E-10 * t3 + 2.00255376E-14 * t4;
//     // H
//     cp[1] = 2.50000001 - 2.30842973E-11 * t + 1.61561948E-14 * t2 - 4.73515235E-18 * t3 + 4.98197357E-22 * t4;
//     // O2
//     cp[2] = 3.28253784 + 1.48308754E-03 * t - 7.57966669E-07 * t2 + 2.09470555E-10 * t3 - 2.16717794E-14 * t4;
//     // O
//     cp[3] = 2.56942078 - 8.59741137E-05 * t + 4.19484589E-08 * t2 - 1.00177799E-11 * t3 + 1.22833691E-15 * t4;
//     // OH
//     cp[4] = 3.09288767 + 5.48429716E-04 * t + 1.26505228E-07 * t2 - 8.79461556E-11 * t3 + 1.17412376E-14 * t4;
//     // HO2
//     cp[5] = 4.01721090 + 2.23982013E-03 * t - 6.33658150E-07 * t2 + 1.14246370E-10 * t3 - 1.07908535E-14 * t4;
//     // H2O2
//     cp[6] = 4.16500285 + 4.90831694E-03 * t - 1.90139225E-06 * t2 + 3.71185986E-10 * t3 - 2.87908305E-14 * t4;
//     // H2O
//     cp[7] = 3.03399249 + 2.17691804E-03 * t - 1.64072518E-07 * t2 - 9.70419870E-11 * t3 + 1.68200992E-14 * t4;
//     if (tt < 5000) {
//       // N2
//       const real tt2{tt * tt}, tt3{t2 * tt}, tt4{t3 * tt};
//       cp[8] = 2.92664 + 0.14879768E-02 * tt - 0.05684760E-05 * tt2 + 0.10097038E-09 * tt3 - 0.06753351E-13 * tt4;
//     } else {
//       cp[8] = 2.92664 + 0.14879768E-02 * 5000 - 0.05684760E-05 * 2.5e+7 + 0.10097038E-09 * 1.25e+11 - 0.06753351E-13 *
//               6.25e+14;
//     }
//   } else {
//     const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
//     if (t < 1000) {
//       // H2
//       cp[0] = 2.34433112 + 7.98052075E-03 * t - 1.94781510E-05 * t2 + 2.01572094E-08 * t3 - 7.37611761E-12 * t4;
//       // H
//       cp[1] = 2.5 + 7.05332819E-13 * t - 1.99591964E-15 * t2 + 2.30081632E-18 * t3 - 9.27732332E-22 * t4;
//       // O2
//       cp[2] = 3.78245636 - 2.99673416E-03 * t + 9.84730201E-06 * t2 - 9.68129509E-09 * t3 + 3.24372837E-12 * t4;
//       // O
//       cp[3] = 3.16826710 - 3.27931884E-03 * t + 6.64306396E-06 * t2 - 6.12806624E-09 * t3 + 2.11265971E-12 * t4;
//       // OH
//       cp[4] = 3.99201543 - 2.40131752E-03 * t + 4.61793841E-06 * t2 - 3.88113333E-09 * t3 + 1.36411470E-12 * t4;
//       // HO2
//       cp[5] = 4.30179801 - 4.74912051E-03 * t + 2.11582891E-05 * t2 - 2.42763894E-08 * t3 + 9.29225124E-12 * t4;
//       // H2O2
//       cp[6] = 4.27611269 - 5.42822417E-04 * t + 1.67335701E-05 * t2 - 2.15770813E-08 * t3 + 8.62454363E-12 * t4;
//       // H2O
//       cp[7] = 4.19864056 - 2.03643410E-03 * t + 6.52040211E-06 * t2 - 5.48797062E-09 * t3 + 1.77197817E-12 * t4;
//       if (t < 300) {
//         // N2
//         cp[8] = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//                 8.1e+9;
//       } else {
//         // N2
//         cp[8] = 3.298677 + 0.14082404E-02 * t - 0.03963222E-04 * t2 + 0.05641515E-07 * t3 - 0.02444854E-10 * t4;
//       }
//     } else {
//       // H2
//       cp[0] = 3.33727920 - 4.94024731E-05 * t + 4.99456778E-07 * t2 - 1.79566394E-10 * t3 + 2.00255376E-14 * t4;
//       // H
//       cp[1] = 2.50000001 - 2.30842973E-11 * t + 1.61561948E-14 * t2 - 4.73515235E-18 * t3 + 4.98197357E-22 * t4;
//       // O2
//       cp[2] = 3.28253784 + 1.48308754E-03 * t - 7.57966669E-07 * t2 + 2.09470555E-10 * t3 - 2.16717794E-14 * t4;
//       // O
//       cp[3] = 2.56942078 - 8.59741137E-05 * t + 4.19484589E-08 * t2 - 1.00177799E-11 * t3 + 1.22833691E-15 * t4;
//       // OH
//       cp[4] = 3.09288767 + 5.48429716E-04 * t + 1.26505228E-07 * t2 - 8.79461556E-11 * t3 + 1.17412376E-14 * t4;
//       // HO2
//       cp[5] = 4.01721090 + 2.23982013E-03 * t - 6.33658150E-07 * t2 + 1.14246370E-10 * t3 - 1.07908535E-14 * t4;
//       // H2O2
//       cp[6] = 4.16500285 + 4.90831694E-03 * t - 1.90139225E-06 * t2 + 3.71185986E-10 * t3 - 2.87908305E-14 * t4;
//       // H2O
//       cp[7] = 3.03399249 + 2.17691804E-03 * t - 1.64072518E-07 * t2 - 9.70419870E-11 * t3 + 1.68200992E-14 * t4;
//       // N2
//       cp[8] = 2.92664 + 0.14879768E-02 * t - 0.05684760E-05 * t2 + 0.10097038E-09 * t3 - 0.06753351E-13 * t4;
//     }
//   }
//   for (int i = 0; i < param->n_spec; ++i) {
//     cp[i] *= param->gas_const[i];
//   }
// }
//
// __device__ void cfd::compute_enthalpy_1(real t, real *enthalpy, const DParameter *param) {
//   const real tt = t;
//   if (t < 200) {
//     t = 200;
//     constexpr real t2{200 * 200}, t3{t2 * 200}, t4{t3 * 200}, t5{t4 * 200};
//     // H2
//     enthalpy[0] = 2.34433112 * t + 0.5 * 7.98052075E-03 * t2 - 1.94781510E-05 * t3 / 3 + 0.25 * 2.01572094E-08 * t4 -
//                   0.2 * 7.37611761E-12 * t5 - 9.17935173E+02;
//     real cp = 2.34433112 + 7.98052075E-03 * t - 1.94781510E-05 * t2 + 2.01572094E-08 * t3 - 7.37611761E-12 * t4;
//     enthalpy[0] += cp * (t - 200);
//     // H
//     enthalpy[1] = 2.5 * t + 0.5 * 7.05332819E-13 * t2 - 1.99591964E-15 * t3 / 3 + 0.25 * 2.30081632E-18 * t4 -
//                   0.2 * 9.27732332E-22 * t5 + 2.54736599E+04;
//     cp = 2.5 + 7.05332819E-13 * t - 1.99591964E-15 * t2 + 2.30081632E-18 * t3 - 9.27732332E-22 * t4;
//     enthalpy[1] += cp * (t - 200);
//     // O2
//     enthalpy[2] = 3.78245636 * t - 0.5 * 2.99673416E-03 * t2 + 9.84730201E-06 * t3 / 3 - 0.25 * 9.68129509E-09 * t4 +
//                   0.2 * 3.24372837E-12 * t5 - 1.06394356E+03;
//     cp = 3.78245636 - 2.99673416E-03 * t + 9.84730201E-06 * t2 - 9.68129509E-09 * t3 + 3.24372837E-12 * t4;
//     enthalpy[2] += cp * (t - 200);
//     // O
//     enthalpy[3] = 3.16826710 * t - 0.5 * 3.27931884E-03 * t2 + 6.64306396E-06 * t3 / 3 - 0.25 * 6.12806624E-09 * t4 +
//                   0.2 * 2.11265971E-12 * t5 + 2.91222592E+04;
//     cp = 3.16826710 - 3.27931884E-03 * t + 6.64306396E-06 * t2 - 6.12806624E-09 * t3 + 2.11265971E-12 * t4;
//     enthalpy[3] += cp * (t - 200);
//     // OH
//     enthalpy[4] = 3.99201543 * t - 0.5 * 2.40131752E-03 * t2 + 4.61793841E-06 * t3 / 3 - 0.25 * 3.88113333E-09 * t4 +
//                   0.2 * 1.36411470E-12 * t5 + 3.61508056E+03;
//     cp = 3.99201543 - 2.40131752E-03 * t + 4.61793841E-06 * t2 - 3.88113333E-09 * t3 + 1.36411470E-12 * t4;
//     enthalpy[4] += cp * (t - 200);
//     // HO2
//     enthalpy[5] = 4.30179801 * t - 0.5 * 4.74912051E-03 * t2 + 2.11582891E-05 * t3 / 3 -
//                   0.25 * 2.42763894E-08 * t4 + 0.2 * 9.29225124E-12 * t5 + 2.94808040E+02;
//     cp = 4.30179801 - 4.74912051E-03 * t + 2.11582891E-05 * t2 - 2.42763894E-08 * t3 + 9.29225124E-12 * t4;
//     enthalpy[5] += cp * (t - 200);
//     // H2O2
//     enthalpy[6] = 4.27611269 * t - 0.5 * 5.42822417E-04 * t2 + 1.67335701E-05 * t3 / 3 -
//                   0.25 * 2.15770813E-08 * t4 + 0.2 * 8.62454363E-12 * t5 - 1.77025821E+04;
//     cp = 4.27611269 - 5.42822417E-04 * t + 1.67335701E-05 * t2 - 2.15770813E-08 * t3 + 8.62454363E-12 * t4;
//     enthalpy[6] += cp * (t - 200);
//     // H2O
//     enthalpy[7] = 4.19864056 * t - 0.5 * 2.03643410E-03 * t2 + 6.52040211E-06 * t3 / 3 -
//                   0.25 * 5.48797062E-09 * t4 + 0.2 * 1.77197817E-12 * t5 - 3.02937267E+04;
//     cp = 4.19864056 - 2.03643410E-03 * t + 6.52040211E-06 * t2 - 5.48797062E-09 * t3 + 1.77197817E-12 * t4;
//     enthalpy[7] += cp * (t - 200);
//     // N2
//     enthalpy[8] = 3.298677 * 300 + 0.5 * 0.14082404E-02 * 90000 - 0.03963222E-04 * 2.7e+7 / 3 +
//                   0.25 * 0.05641515E-07 * 8.1e+9 - 0.2 * 0.02444854E-10 * 2.43e+12 - 1.02089999E+03;
//     cp = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//          8.1e+9;
//     enthalpy[8] += cp * (t - 300);
//   } else if (t > 3000) {
//     t = 3000;
//     constexpr real t2{9e+6}, t3{2.7e+10}, t4{8.1e+13}, t5{2.43e+17};
//     // H2
//     enthalpy[0] = 3.33727920 * t - 0.5 * 4.94024731E-05 * t2 + 4.99456778E-07 * t3 / 3 -
//                   0.25 * 1.79566394E-10 * t4 + 0.2 * 2.00255376E-14 * t5 - 9.50158922E+02;
//     real cp = 3.33727920 - 4.94024731E-05 * t + 4.99456778E-07 * t2 - 1.79566394E-10 * t3 + 2.00255376E-14 * t4;
//     enthalpy[0] += cp * (t - 3000); // Do a linear interpolation for enthalpy
//     // H
//     enthalpy[1] = 2.50000001 * t - 0.5 * 2.30842973E-11 * t2 + 1.61561948E-14 * t3 / 3 -
//                   0.25 * 4.73515235E-18 * t4 + 0.2 * 4.98197357E-22 * t5 + 2.54736599E+04;
//     cp = 2.50000001 - 2.30842973E-11 * t + 1.61561948E-14 * t2 - 4.73515235E-18 * t3 + 4.98197357E-22 * t4;
//     enthalpy[1] += cp * (t - 3000);
//     // O2
//     enthalpy[2] = 3.28253784 * t + 0.5 * 1.48308754E-03 * t2 - 7.57966669E-07 * t3 / 3 +
//                   0.25 * 2.09470555E-10 * t4 - 0.2 * 2.16717794E-14 * t5 - 1.08845772E+03;
//     cp = 3.28253784 + 1.48308754E-03 * t - 7.57966669E-07 * t2 + 2.09470555E-10 * t3 - 2.16717794E-14 * t4;
//     enthalpy[2] += cp * (t - 3000);
//     // O
//     enthalpy[3] = 2.56942078 * t - 0.5 * 8.59741137E-05 * t2 + 4.19484589E-08 * t3 / 3 -
//                   0.25 * 1.00177799E-11 * t4 + 0.2 * 1.22833691E-15 * t5 + 2.92175791E+04;
//     cp = 2.56942078 - 8.59741137E-05 * t + 4.19484589E-08 * t2 - 1.00177799E-11 * t3 + 1.22833691E-15 * t4;
//     enthalpy[3] += cp * (t - 3000);
//     // OH
//     enthalpy[4] = 3.09288767 * t + 0.5 * 5.48429716E-04 * t2 + 1.26505228E-07 * t3 / 3 -
//                   0.25 * 8.79461556E-11 * t4 + 0.2 * 1.17412376E-14 * t5 + 3.85865700E+03;
//     cp = 3.09288767 + 5.48429716E-04 * t + 1.26505228E-07 * t2 - 8.79461556E-11 * t3 + 1.17412376E-14 * t4;
//     enthalpy[4] += cp * (t - 3000);
//     // HO2
//     enthalpy[5] = 4.01721090 * t + 0.5 * 2.23982013E-03 * t2 - 6.33658150E-07 * t3 / 3 +
//                   0.25 * 1.14246370E-10 * t4 - 0.2 * 1.07908535E-14 * t5 + 1.11856713E+02;
//     cp = 4.01721090 + 2.23982013E-03 * t - 6.33658150E-07 * t2 + 1.14246370E-10 * t3 - 1.07908535E-14 * t4;
//     enthalpy[5] += cp * (t - 3000);
//     // H2O2
//     enthalpy[6] = 4.16500285 * t + 0.5 * 4.90831694E-03 * t2 - 1.90139225E-06 * t3 / 3 +
//                   0.25 * 3.71185986E-10 * t4 - 0.2 * 2.87908305E-14 * t5 - 1.78617877E+04;
//     cp = 4.16500285 + 4.90831694E-03 * t - 1.90139225E-06 * t2 + 3.71185986E-10 * t3 - 2.87908305E-14 * t4;
//     enthalpy[6] += cp * (t - 3000);
//     // H2O
//     enthalpy[7] = 3.03399249 * t + 0.5 * 2.17691804E-03 * t2 - 1.64072518E-07 * t3 / 3 -
//                   0.25 * 9.70419870E-11 * t4 + 0.2 * 1.68200992E-14 * t5 - 3.00042971E+04;
//     cp = 3.03399249 + 2.17691804E-03 * t - 1.64072518E-07 * t2 - 9.70419870E-11 * t3 + 1.68200992E-14 * t4;
//     enthalpy[7] += cp * (t - 3000);
//     if (tt < 5000) {
//       // N2
//       const real tt2{tt * tt}, tt3{t2 * tt}, tt4{t3 * tt}, tt5{tt4 * tt};
//       enthalpy[8] = 2.92664 * tt + 0.5 * 0.14879768E-02 * tt2 - 0.05684760E-05 * tt3 / 3 +
//                     0.25 * 0.10097038E-09 * tt4 - 0.2 * 0.06753351E-13 * tt5 - 0.09227977E+04;
//     } else {
//       enthalpy[8] = 2.92664 * 5000 + 0.5 * 0.14879768E-02 * 2.5e+7 - 0.05684760E-05 * 1.25e+11 / 3 +
//                     0.25 * 0.10097038E-09 * 6.25e+14 - 0.2 * 0.06753351E-13 * 3.125e+16 - 0.09227977E+04;
//       cp = 2.92664 + 0.14879768E-02 * 5000 - 0.05684760E-05 * 2.5e+7 + 0.10097038E-09 * 1.25e+11 - 0.06753351E-13 *
//            6.25e+14;
//       enthalpy[8] += cp * (t - 5000);
//     }
//   } else {
//     const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
//     if (t < 1000) {
//       // H2
//       enthalpy[0] = 2.34433112 * t + 0.5 * 7.98052075E-03 * t2 - 1.94781510E-05 * t3 / 3 + 0.25 * 2.01572094E-08 * t4 -
//                     0.2 * 7.37611761E-12 * t5 - 9.17935173E+02;
//       // H
//       enthalpy[1] = 2.5 * t + 0.5 * 7.05332819E-13 * t2 - 1.99591964E-15 * t3 / 3 + 0.25 * 2.30081632E-18 * t4 -
//                     0.2 * 9.27732332E-22 * t5 + 2.54736599E+04;
//       // O2
//       enthalpy[2] = 3.78245636 * t - 0.5 * 2.99673416E-03 * t2 + 9.84730201E-06 * t3 / 3 - 0.25 * 9.68129509E-09 * t4 +
//                     0.2 * 3.24372837E-12 * t5 - 1.06394356E+03;
//       // O
//       enthalpy[3] = 3.16826710 * t - 0.5 * 3.27931884E-03 * t2 + 6.64306396E-06 * t3 / 3 - 0.25 * 6.12806624E-09 * t4 +
//                     0.2 * 2.11265971E-12 * t5 + 2.91222592E+04;
//       // OH
//       enthalpy[4] = 3.99201543 * t - 0.5 * 2.40131752E-03 * t2 + 4.61793841E-06 * t3 / 3 - 0.25 * 3.88113333E-09 * t4 +
//                     0.2 * 1.36411470E-12 * t5 + 3.61508056E+03;
//       // HO2
//       enthalpy[5] = 4.30179801 * t - 0.5 * 4.74912051E-03 * t2 + 2.11582891E-05 * t3 / 3 -
//                     0.25 * 2.42763894E-08 * t4 + 0.2 * 9.29225124E-12 * t5 + 2.94808040E+02;
//       // H2O2
//       enthalpy[6] = 4.27611269 * t - 0.5 * 5.42822417E-04 * t2 + 1.67335701E-05 * t3 / 3 -
//                     0.25 * 2.15770813E-08 * t4 + 0.2 * 8.62454363E-12 * t5 - 1.77025821E+04;
//       // H2O
//       enthalpy[7] = 4.19864056 * t - 0.5 * 2.03643410E-03 * t2 + 6.52040211E-06 * t3 / 3 -
//                     0.25 * 5.48797062E-09 * t4 + 0.2 * 1.77197817E-12 * t5 - 3.02937267E+04;
//       if (t < 300) {
//         // N2
//         enthalpy[8] = 3.298677 * 300 + 0.5 * 0.14082404E-02 * 90000 - 0.03963222E-04 * 2.7e+7 / 3 +
//                       0.25 * 0.05641515E-07 * 8.1e+9 - 0.2 * 0.02444854E-10 * 2.43e+12 - 1.02089999E+03;
//         real cp = 3.298677 + 0.14082404E-02 * 300 - 0.03963222E-04 * 90000 + 0.05641515E-07 * 2.7e+7 - 0.02444854E-10 *
//                   8.1e+9;
//         enthalpy[8] += cp * (t - 300);
//       } else {
//         // N2
//         enthalpy[8] = 3.298677 * t + 0.5 * 0.14082404E-02 * t2 - 0.03963222E-04 * t3 / 3 +
//                       0.25 * 0.05641515E-07 * t4 - 0.2 * 0.02444854E-10 * t5 - 1.02089999E+03;
//       }
//     } else {
//       // H2
//       enthalpy[0] = 3.33727920 * t - 0.5 * 4.94024731E-05 * t2 + 4.99456778E-07 * t3 / 3 -
//                     0.25 * 1.79566394E-10 * t4 + 0.2 * 2.00255376E-14 * t5 - 9.50158922E+02;
//       // H
//       enthalpy[1] = 2.50000001 * t - 0.5 * 2.30842973E-11 * t2 + 1.61561948E-14 * t3 / 3 -
//                     0.25 * 4.73515235E-18 * t4 + 0.2 * 4.98197357E-22 * t5 + 2.54736599E+04;
//       // O2
//       enthalpy[2] = 3.28253784 * t + 0.5 * 1.48308754E-03 * t2 - 7.57966669E-07 * t3 / 3 +
//                     0.25 * 2.09470555E-10 * t4 - 0.2 * 2.16717794E-14 * t5 - 1.08845772E+03;
//       // O
//       enthalpy[3] = 2.56942078 * t - 0.5 * 8.59741137E-05 * t2 + 4.19484589E-08 * t3 / 3 -
//                     0.25 * 1.00177799E-11 * t4 + 0.2 * 1.22833691E-15 * t5 + 2.92175791E+04;
//       // OH
//       enthalpy[4] = 3.09288767 * t + 0.5 * 5.48429716E-04 * t2 + 1.26505228E-07 * t3 / 3 -
//                     0.25 * 8.79461556E-11 * t4 + 0.2 * 1.17412376E-14 * t5 + 3.85865700E+03;
//       // HO2
//       enthalpy[5] = 4.01721090 * t + 0.5 * 2.23982013E-03 * t2 - 6.33658150E-07 * t3 / 3 +
//                     0.25 * 1.14246370E-10 * t4 - 0.2 * 1.07908535E-14 * t5 + 1.11856713E+02;
//       // H2O2
//       enthalpy[6] = 4.16500285 * t + 0.5 * 4.90831694E-03 * t2 - 1.90139225E-06 * t3 / 3 +
//                     0.25 * 3.71185986E-10 * t4 - 0.2 * 2.87908305E-14 * t5 - 1.78617877E+04;
//       // H2O
//       enthalpy[7] = 3.03399249 * t + 0.5 * 2.17691804E-03 * t2 - 1.64072518E-07 * t3 / 3 -
//                     0.25 * 9.70419870E-11 * t4 + 0.2 * 1.68200992E-14 * t5 - 3.00042971E+04;
//       // N2
//       enthalpy[8] = 2.92664 * t + 0.5 * 0.14879768E-02 * t2 - 0.05684760E-05 * t3 / 3 +
//                     0.25 * 0.10097038E-09 * t4 - 0.2 * 0.06753351E-13 * t5 - 0.09227977E+04;
//     }
//   }
//   for (int i = 0; i < param->n_spec; ++i) {
//     enthalpy[i] *= param->gas_const[i];
//   }
// }

#endif

#include "MixingLayer.cuh"
#include "Parameter.h"
#include "ChemData.h"
#include "Constants.h"
#include "Parallel.h"
#include "Transport.cuh"

namespace cfd {
void get_mixing_layer_info(Parameter &parameter, const Species &species, std::vector<real> &var_info) {
  // If we initialize a mixing layer problem, we need to know the following parameters:
  // 1. The convective Mach number Ma_c
  // 2. The vortex thickness at the entrance delta_omega
  // 3. The info of the fuel stream: The mass fraction of species, the Mach number, and temperature
  // 4. The info of the oxidizer stream: The mass fraction of species, the temperature. The Mach number should be computed with the Ma_c.
  const int n_spec{parameter.get_int("n_spec")};
  const real ma_c{parameter.get_real("ma_c")};
  auto &upper = parameter.get_struct("upper_stream");
  auto &lower = parameter.get_struct("lower_stream");

  real ma_upper{-1}, ma_lower{-1};
  if (upper.find("mach") != upper.cend()) {
    ma_upper = std::get<real>(upper.at("mach"));
  }
  if (lower.find("mach") != lower.cend()) {
    ma_lower = std::get<real>(lower.at("mach"));
  }
  const real vel_ratio{parameter.get_real("velocity_ratio")};

  const real T_upper{std::get<real>(upper.at("temperature"))}, T_lower{std::get<real>(lower.at("temperature"))};
  const real p_upper{std::get<real>(upper.at("pressure"))}, p_lower{std::get<real>(lower.at("pressure"))};
  real rho_upper, rho_lower;
  real c_upper, c_lower;
  real mix_frac_upper{-1}, mix_frac_lower{-1};
  real yk_upper[MAX_SPEC_NUMBER] = {}, yk_lower[MAX_SPEC_NUMBER] = {};
  real mu_upper{0}, mu_lower{0};
  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (const auto &[name, idx]: species.spec_list) {
      if (upper.find(name) != upper.cend()) {
        yk_upper[idx] = std::get<real>(upper.at(name));
      }
      if (lower.find(name) != lower.cend()) {
        yk_lower[idx] = std::get<real>(lower.at(name));
      }
    }
    mix_frac_upper = std::get<real>(upper.at("mixture_fraction"));
    mix_frac_lower = std::get<real>(lower.at("mixture_fraction"));
    real cpi_upper[MAX_SPEC_NUMBER], cpi_lower[MAX_SPEC_NUMBER];
    species.compute_cp(T_upper, cpi_upper);
    species.compute_cp(T_lower, cpi_lower);
    real mw_inv_upper{0}, mw_inv_lower{0};
    real cp_upper{0}, cp_lower{0};
    for (int i = 0; i < n_spec; ++i) {
      mw_inv_upper += yk_upper[i] / species.mw[i];
      mw_inv_lower += yk_lower[i] / species.mw[i];
      cp_upper += cpi_upper[i] * yk_upper[i];
      cp_lower += cpi_lower[i] * yk_lower[i];
    }
    const real gamma_upper{cp_upper / (cp_upper - R_u * mw_inv_upper)};
    const real gamma_lower{cp_lower / (cp_lower - R_u * mw_inv_lower)};
    c_upper = std::sqrt(gamma_upper * R_u * mw_inv_upper * T_upper);
    c_lower = std::sqrt(gamma_lower * R_u * mw_inv_lower * T_lower);
    rho_upper = p_upper / (R_u * mw_inv_upper * T_upper);
    rho_lower = p_lower / (R_u * mw_inv_lower * T_lower);
    mu_upper = compute_viscosity(T_upper, 1 / mw_inv_upper, yk_upper, species);
    mu_lower = compute_viscosity(T_lower, 1 / mw_inv_lower, yk_lower, species);
  } else {
    c_upper = std::sqrt(gamma_air * R_u / mw_air * T_upper);
    c_lower = std::sqrt(gamma_air * R_u / mw_air * T_lower);
    rho_upper = p_upper * mw_air / (R_u * T_upper);
    rho_lower = p_lower * mw_air / (R_u * T_lower);
    mu_upper = Sutherland(T_upper);
    mu_lower = Sutherland(T_lower);
  }
  if (vel_ratio > 0) {
    real u_slow, u_fast;
    if (vel_ratio > 1) {
      u_slow = ma_c * (c_upper + c_lower) / (vel_ratio - 1);
      u_fast = vel_ratio * u_slow;
    } else {
      u_fast = ma_c * (c_upper + c_lower) / (1 - vel_ratio);
      u_slow = u_fast * vel_ratio;
    }
    if (parameter.get_bool("upper_faster")) {
      ma_upper = u_fast / c_upper;
      ma_lower = u_slow / c_lower;
    } else {
      ma_upper = u_slow / c_upper;
      ma_lower = u_fast / c_lower;
    }
  } else if (ma_upper < 0 && ma_lower < 0) {
    printf("At least one of the mach number and the convective Mach number should be given.\n");
    MpiParallel::exit();
  } else if (ma_upper > 0 && ma_lower > 0) {
    const real convective_mach = abs(ma_upper * c_upper - ma_lower * c_lower) / (c_upper + c_lower);
    if (abs(ma_c - convective_mach) > 1e-3) {
      printf(
        "The convective mach number with given streams = %e, which is not consistent with the given convective mach number %e.\n",
        convective_mach, ma_c);
      printf(
        "The computation is continued with the given streams, where the given convective mach number is ignored.\n");
    }
  } else if (ma_upper > 0) {
    if (parameter.get_bool("upper_faster")) {
      ma_lower = (ma_upper - ma_c) * c_upper / c_lower - ma_c;
    } else {
      ma_lower = (ma_c + ma_upper) * c_upper / c_lower + ma_c;
    }
  } else {
    if (parameter.get_bool("upper_faster")) {
      ma_upper = (ma_c + ma_lower) * c_lower / c_upper + ma_c;
    } else {
      ma_upper = (ma_lower - ma_c) * c_lower / c_upper - ma_c;
    }
  }
  const real u_upper{ma_upper * c_upper}, u_lower{ma_lower * c_lower};

  // 7 : rho(0, 7+ns), u(1, 8+ns), v(2, 9+ns), w(3, 10+ns), p(4, 11+ns), T(5, 12+ns), mix_frac(6+ns, 13+2*ns);
  // n_spec : yk(6:6+ns-1, 13+ns:13+2*ns-1)
  // 2 : tke(13+2*ns+1, 13+2*ns+3), omega(13+2*ns+2, 13+2*ns+4)
  var_info.resize((7 + n_spec + 2) * 2, 0);
  var_info[0] = rho_upper;
  var_info[1] = u_upper;
  var_info[2] = 0;
  var_info[3] = 0;
  var_info[4] = p_upper;
  var_info[5] = T_upper;
  for (int i = 0; i < n_spec; ++i) {
    var_info[6 + i] = yk_upper[i];
  }
  var_info[6 + n_spec] = mix_frac_upper;

  var_info[7 + n_spec] = rho_lower;
  var_info[8 + n_spec] = u_lower;
  var_info[9 + n_spec] = 0;
  var_info[10 + n_spec] = 0;
  var_info[11 + n_spec] = p_lower;
  var_info[12 + n_spec] = T_lower;
  for (int i = 0; i < n_spec; ++i) {
    var_info[13 + n_spec + i] = yk_lower[i];
  }
  var_info[13 + 2 * n_spec] = mix_frac_lower;

  int counter = 13 + 2 * n_spec;

  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    if (parameter.get_int("RANS_model") == 2) {
      // SST model
      real turb_intensity{0.01};
      real mutMul{1};
      if (upper.find("turb_viscosity_ratio") != upper.cend()) {
        mutMul = std::get<real>(upper.at("turb_viscosity_ratio"));
      }
      if (upper.find("turbulence_intensity") != upper.cend()) {
        turb_intensity = std::get<real>(upper.at("turbulence_intensity"));
      }
      const real tke_upper = 1.5 * turb_intensity * turb_intensity * u_upper * u_upper;
      const real omega_upper = rho_upper * tke_upper / (mutMul * mu_upper);

      turb_intensity = 0.01;
      mutMul = 1;
      if (lower.find("turb_viscosity_ratio") != lower.cend()) {
        mutMul = std::get<real>(lower.at("turb_viscosity_ratio"));
      }
      if (lower.find("turbulence_intensity") != lower.cend()) {
        turb_intensity = std::get<real>(lower.at("turbulence_intensity"));
      }
      const real tke_lower = 1.5 * turb_intensity * turb_intensity * u_lower * u_lower;
      const real omega_lower = rho_lower * tke_lower / (mutMul * mu_lower);

      var_info[counter + 1] = tke_upper;
      var_info[counter + 2] = omega_upper;
      var_info[counter + 3] = tke_lower;
      var_info[counter + 4] = omega_lower;
      // counter += 4;
    }
  }

  if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
    var_info.resize(14 + 2 * n_spec + 4 + 2 * n_ps);
    counter = 14 + 2 * n_spec + 4;
    for (int i = 1; i <= n_ps; ++i) {
      if (upper.find("ps" + std::to_string(i)) != upper.cend()) {
        var_info[counter + 2 * (i - 1)] = std::get<real>(upper.at("ps" + std::to_string(i)));
      }
      if (lower.find("ps" + std::to_string(i)) != lower.cend()) {
        var_info[counter + 2 * (i - 1) + 1] = std::get<real>(lower.at("ps" + std::to_string(i)));
      }
    }
  }
}
}

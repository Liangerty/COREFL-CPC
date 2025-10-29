#include "Transport.cuh"
#include "DParameter.cuh"
#include "Field.h"
#include "Constants.h"

__host__ __device__
real cfd::Sutherland(real temperature) {
  return 1.7894e-5 * pow(temperature / 288.16, 1.5) * (288.16 + 110) / (temperature + 110);
}

real cfd::compute_viscosity(real temperature, real mw_total, real const *Y, const Species &spec) {
  // This method can only be used on CPU, while for GPU the allocation may be performed in every step
  std::vector<real> x(spec.n_spec);
  std::vector<real> vis_spec(spec.n_spec);
  gxl::MatrixDyn<real> partition_fun(spec.n_spec, spec.n_spec);
  for (int i = 0; i < spec.n_spec; ++i) {
    x[i] = Y[i] * mw_total / spec.mw[i];
    const real t_dl{temperature * spec.LJ_potent_inv[i]}; // dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis_spec[i] = spec.vis_coeff[i] * std::sqrt(temperature) / collision_integral;
  }
  for (int i = 0; i < spec.n_spec; ++i) {
    for (int j = 0; j < spec.n_spec; ++j) {
      if (i == j) {
        partition_fun(i, j) = 1.0;
      } else {
        const real numerator{1 + std::sqrt(vis_spec[i] / vis_spec[j]) * spec.WjDivWi_to_One4th(i, j)};
        partition_fun(i, j) = numerator * numerator * spec.sqrt_WiDivWjPl1Mul8(i, j);
      }
    }
  }
  real viscosity{0};
  for (int i = 0; i < spec.n_spec; ++i) {
    real vis_temp{0};
    for (int j = 0; j < spec.n_spec; ++j) {
      vis_temp += partition_fun(i, j) * x[j];
    }
    viscosity += vis_spec[i] * x[i] / vis_temp;
  }
  return viscosity;
}

__device__ void cfd::compute_transport_property(int i, int j, int k, real temperature, real mw_total, real *cp,
  DParameter *param, DZone *zone) {
  const auto n_spec{param->n_spec};

  __shared__ real s[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER * 2 + MAX_SPEC_NUMBER];
  real *WjDivWi_to_One4th = s;
  real *sqrt_WiDivWjPl1Mul8 = WjDivWi_to_One4th + n_spec * n_spec;
  real *mw = &sqrt_WiDivWjPl1Mul8[n_spec * n_spec];
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Load the parameters into shared memory
    for (int m = 0; m < n_spec; ++m) {
      for (int n = 0; n < n_spec; ++n) {
        WjDivWi_to_One4th[m * n_spec + n] = param->WjDivWi_to_One4th(m, n);
        sqrt_WiDivWjPl1Mul8[m * n_spec + n] = param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      mw[m] = 1.0 / param->imw[m];
    }
  }
  __syncthreads();

  real X[MAX_SPEC_NUMBER], vis_denom[MAX_SPEC_NUMBER];
  // To save registers, vis_denom is first used as viscosity of species, then the denominator of the diffusivity computations
  real temp = sqrt(temperature); // Square root of temperature
  for (int l = 0; l < n_spec; ++l) {
    X[l] = zone->sv(i, j, k, l) * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    vis_denom[l] =
        param->vis_coeff[l] * temp / (1.147 * pow(t_dl, -0.145) + pow(t_dl + 0.5, -2)); // temp = sqrt(temperature)
  }
  // The binary-diffusion coefficients including self-diffusion coefficients are computed.
  const real t_3over2_over_p{temperature * temp}; // temp = sqrt(temperature)
  // The p is not divided here, because the computation of (rho * D_ij) contains a multiplication of p, these two operations cancel each other.

  real viscosity = 0;
  real conductivity = 0;
  for (int m = 0; m < n_spec; ++m) {
    // compute the thermal conductivity
    real R = param->gas_const[m];
    real lambda = 15 * 0.25 * vis_denom[m] * R;
    if (param->geometry[m] == 1) {
      // Linear geometry
      const real t_red{temperature * param->kb_over_eps_jk(m, m)};
      // compute ZRot
      real tRedInv = 1 / t_red;
      real FT = 1 + 0.5 * sqrt(tRedInv * pi) * pi + (2 + 0.25 * pi * pi) * tRedInv +
                sqrt(pi * tRedInv) * pi * tRedInv;
      const real rhoD =
          param->binary_diffusivity_coeff(m, m) * t_3over2_over_p / (compute_Omega_D(t_red) * R * temperature);
      const real ADivPiB =
          (2.5 * vis_denom[m] - rhoD) /
          ((pi * param->ZRotF298[m] / FT * vis_denom[m] + 10.0 / 3 * vis_denom[m] + 2.0 * rhoD));
      lambda += -5 * ADivPiB * vis_denom[m] * R + rhoD * cp[m] + rhoD * (2 * ADivPiB - 2.5) * R;
    } else if (param->geometry[m] == 2) {
      // Non-linear geometry
      const real t_red{temperature * param->kb_over_eps_jk(m, m)};
      // compute ZRot
      real tRedInv = 1 / t_red;
      real FT = 1 + 0.5 * sqrt(tRedInv * pi) * pi + (2 + 0.25 * pi * pi) * tRedInv +
                sqrt(pi * tRedInv) * pi * tRedInv;
      const real rhoD =
          param->binary_diffusivity_coeff(m, m) * t_3over2_over_p / (compute_Omega_D(t_red) * R * temperature);
      const real ADivPiB =
          (2.5 * vis_denom[m] - rhoD) / ((pi * param->ZRotF298[m] / FT * vis_denom[m] + 5 * vis_denom[m] + 2.0 * rhoD));
      lambda += -7.5 * ADivPiB * vis_denom[m] * R + rhoD * cp[m] + rhoD * (3 * ADivPiB - 2.5) * R;
    }

    temp = 0;
    R = sqrt(vis_denom[m]); // R is not used anymore, so we can reuse it.
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        // convert the next line into a fma operation
        const real numerator = 1.0 + rsqrt(vis_denom[n]) * (R * WjDivWi_to_One4th[m * n_spec + n]);
        partition_func = numerator * numerator * sqrt_WiDivWjPl1Mul8[m * n_spec + n];
      }
      //      vis_temp += partition_func * X[n];
      temp += partition_func * X[n];
    }
    //    const real cond_temp = 1.065 * vis_temp - 0.065 * X[m];
    viscosity += vis_denom[m] * X[m] / temp;
    //    viscosity += vis_denom[m] * X[m] / vis_temp;
    //    conductivity += lambda * X[m] / (1.065 * vis_temp - 0.065 * X[m]);
    conductivity += lambda * X[m] / (1.065 * temp - 0.065 * X[m]);
  }
  zone->mul(i, j, k) = viscosity;
  zone->thermal_conductivity(i, j, k) = conductivity;

  // The diffusivity is computed by mixture-averaged method.
  constexpr real eps{1e-12};
  // cp is reused for the numerator, while vis_denom is reused for the denominator of diffusivity
  memset(cp, 0, sizeof(real) * MAX_SPEC_NUMBER);
  memset(vis_denom, 0, sizeof(real) * MAX_SPEC_NUMBER);
  // The viscosity of species are not used here, so we can reuse this array.
  for (int l = 0; l < n_spec; ++l) {
    temp = (X[l] + eps) * mw[l];
    for (int n = 0; n < n_spec; ++n) {
      if (l != n) {
        cp[n] += temp;
      }

      if (n > l) {
        temp = compute_Omega_D(temperature * param->kb_over_eps_jk(l, n)) /
               (param->binary_diffusivity_coeff(l, n) * t_3over2_over_p); // The inverse of D_ln*p
        vis_denom[l] += (X[n] + eps) * temp;
        vis_denom[n] += (X[l] + eps) * temp;
      }
    }
  }
  for (int l = 0; l < n_spec; ++l) {
    zone->rho_D(i, j, k, l) = cp[l] / (vis_denom[l] * temperature * R_u);
  }
}

__device__ real
cfd::compute_viscosity(int i, int j, int k, real temperature, real mw_total, DParameter *param, const DZone *zone) {
  const auto n_spec{param->n_spec};
  const real *imw = param->imw;
  const auto &yk = zone->sv;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = yk(i, j, k, l) * mw_total * imw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    viscosity += vis[m] * X[m] / vis_temp;
  }
  return viscosity;
}

__device__ real cfd::compute_Omega_D(real t_red) {
  return 1.0548 * std::pow(t_red, -0.15504) + pow(t_red + 0.55909, -2.1705);
}

__device__ real cfd::compute_viscosity(real temperature, real mw_total, const real *Y, DParameter *param) {
  const auto n_spec{param->n_spec};
  const real *imw = param->imw;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = Y[l] * mw_total * imw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    viscosity += vis[m] * X[m] / vis_temp;
  }
  return viscosity;
}

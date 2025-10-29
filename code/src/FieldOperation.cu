#include "FieldOperation.cuh"

__device__ void
cfd::compute_temperature_and_pressure(int i, int j, int k, const DParameter *param, DZone *zone, real total_energy) {
  const int n_spec = param->n_spec;
  auto &Y = zone->sv;
  auto &bv = zone->bv;

  real gas_const{0};
  for (int l = 0; l < n_spec; ++l) {
    gas_const += Y(i, j, k, l) * param->gas_const[l];
  }
  const real e =
      total_energy / bv(i, j, k, 0) - 0.5 * (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
                                             bv(i, j, k, 3) * bv(i, j, k, 3));

  real err{1}, t{bv(i, j, k, 5)};
  constexpr int max_iter{1000};
  constexpr real eps{1e-3};
  int iter = 0;

  real h_i[MAX_SPEC_NUMBER], cp_i[MAX_SPEC_NUMBER];
  while (err > eps && iter++ < max_iter) {
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cp_tot{0}, h{0};
    for (int l = 0; l < n_spec; ++l) {
      cp_tot += cp_i[l] * Y(i, j, k, l);
      h += h_i[l] * Y(i, j, k, l);
    }
    const real e_t = h - gas_const * t;
    const real cv = cp_tot - gas_const;
    const real t1 = t - (e_t - e) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  bv(i, j, k, 5) = t;
  bv(i, j, k, 4) = bv(i, j, k, 0) * t * gas_const;
  if (t < 0 || bv(i, j, k, 4) < 0) {
    printf("(%d,%d,%d) in p[%d]: duvwpt=(%f,%f,%f,%f,%f,%f)"
           "\n", i, j, k, param->myid,
           bv(i, j, k, 0), bv(i, j, k, 1), bv(i, j, k, 2), bv(i, j, k, 3), bv(i, j, k, 4), bv(i, j, k, 5)
    );
  }
}

__global__ void cfd::eliminate_k_gradient(DZone *zone, const DParameter *param) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  if (i >= mx + ngg || j >= my + ngg) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;
  const int n_scalar = param->n_scalar;

  for (int k = 1; k <= ngg; ++k) {
    for (int l = 0; l < 6; ++l) {
      bv(i, j, k, l) = bv(i, j, 0, l);
      bv(i, j, -k, l) = bv(i, j, 0, l);
    }
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv(i, j, 0, l);
      sv(i, j, -k, l) = sv(i, j, 0, l);
    }
    auto &cv = zone->cv;
    for (int l = 0; l < param->n_var; ++l) {
      cv(i, j, k, l) = cv(i, j, 0, l);
      cv(i, j, -k, l) = cv(i, j, 0, l);
    }
  }
}

__device__ real ducros_sensor(const cfd::DZone *zone, real eps, int i, int j, int k) {
  const auto &bv = zone->bv;
  const auto &metric = zone->metric;
  const auto xi_x = metric(i, j, k, 0), xi_y = metric(i, j, k, 1), xi_z = metric(i, j, k, 2);
  const auto eta_x = metric(i, j, k, 3), eta_y = metric(i, j, k, 4), eta_z = metric(i, j, k, 5);
  const auto zeta_x = metric(i, j, k, 6), zeta_y = metric(i, j, k, 7), zeta_z = metric(i, j, k, 8);

  constexpr real oneD12 = 1.0 / 12.0, twoD3 = 2.0 / 3.0;
  const real dud1 = oneD12 * bv(i - 2, j, k, 1) - twoD3 * bv(i - 1, j, k, 1) +
                    twoD3 * bv(i + 1, j, k, 1) - oneD12 * bv(i + 2, j, k, 1);
  const real dud2 = oneD12 * bv(i, j - 2, k, 1) - twoD3 * bv(i, j - 1, k, 1) +
                    twoD3 * bv(i, j + 1, k, 1) - oneD12 * bv(i, j + 2, k, 1);
  const real dud3 = oneD12 * bv(i, j, k - 2, 1) - twoD3 * bv(i, j, k - 1, 1) +
                    twoD3 * bv(i, j, k + 1, 1) - oneD12 * bv(i, j, k + 2, 1);
  const real dvd1 = oneD12 * bv(i - 2, j, k, 2) - twoD3 * bv(i - 1, j, k, 2) +
                    twoD3 * bv(i + 1, j, k, 2) - oneD12 * bv(i + 2, j, k, 2);
  const real dvd2 = oneD12 * bv(i, j - 2, k, 2) - twoD3 * bv(i, j - 1, k, 2) +
                    twoD3 * bv(i, j + 1, k, 2) - oneD12 * bv(i, j + 2, k, 2);
  const real dvd3 = oneD12 * bv(i, j, k - 2, 2) - twoD3 * bv(i, j, k - 1, 2) +
                    twoD3 * bv(i, j, k + 1, 2) - oneD12 * bv(i, j, k + 2, 2);
  const real dwd1 = oneD12 * bv(i - 2, j, k, 3) - twoD3 * bv(i - 1, j, k, 3) +
                    twoD3 * bv(i + 1, j, k, 3) - oneD12 * bv(i + 2, j, k, 3);
  const real dwd2 = oneD12 * bv(i, j - 2, k, 3) - twoD3 * bv(i, j - 1, k, 3) +
                    twoD3 * bv(i, j + 1, k, 3) - oneD12 * bv(i, j + 2, k, 3);
  const real dwd3 = oneD12 * bv(i, j, k - 2, 3) - twoD3 * bv(i, j, k - 1, 3) +
                    twoD3 * bv(i, j, k + 1, 3) - oneD12 * bv(i, j, k + 2, 3);
  const real duDx = dud1 * xi_x + dud2 * eta_x + dud3 * zeta_x;
  const real dvDy = dvd1 * xi_y + dvd2 * eta_y + dvd3 * zeta_y;
  const real dwDz = dwd1 * xi_z + dwd2 * eta_z + dwd3 * zeta_z;
  real divV2 = duDx + dvDy + dwDz;
  if (divV2 > 0) {
    return 0;
  } else {
    const real duDy = dud1 * xi_y + dud2 * eta_y + dud3 * zeta_y;
    const real duDz = dud1 * xi_z + dud2 * eta_z + dud3 * zeta_z;
    const real dvDx = dvd1 * xi_x + dvd2 * eta_x + dvd3 * zeta_x;
    const real dvDz = dvd1 * xi_z + dvd2 * eta_z + dvd3 * zeta_z;
    const real dwDx = dwd1 * xi_x + dwd2 * eta_x + dwd3 * zeta_x;
    const real dwDy = dwd1 * xi_y + dwd2 * eta_y + dwd3 * zeta_y;

    divV2 *= divV2;
    const real velocity_curl_2 = (dwDy - dvDz) * (dwDy - dvDz) + (duDz - dwDx) * (duDz - dwDx) +
                                 (dvDx - duDy) * (dvDx - duDy);
    // const real delta0 = 0.001;
    // const real epsilon = param->v_ref * param->v_ref / (delta0 * delta0);
    return divV2 / (divV2 + velocity_curl_2 + eps);
  }
}

__device__ real jameson_sensor(const cfd::DZone *zone, int i, int j, int k) {
  // According to (Dang,2022,PoF)
  const auto &bv = zone->bv;

  const real twoP = bv(i, j, k, 4) * 2;
  const real pI = bv(i + 1, j, k, 4) + bv(i - 1, j, k, 4);
  const real pJ = bv(i, j + 1, k, 4) + bv(i, j - 1, k, 4);
  const real pK = bv(i, j, k + 1, 4) + bv(i, j, k - 1, 4);

  const real phiI = abs(pI - twoP) / (pI + twoP);
  const real phiJ = abs(pJ - twoP) / (pJ + twoP);
  const real phiK = abs(pK - twoP) / (pK + twoP);
  return phiI + phiJ + phiK;
}

__device__ real density_pressure_jump_sensor(const cfd::DZone *zone, real eps, int i, int j, int k) {
  // According to (Martinez Ferrer et, al., 2014, Computers & Fluids)
  const auto &bv = zone->bv;

  // The original sensor is: abs(\rho_{i+1} - \rho_i)/\rho_i < 0.05 && abs(p_{i+1} - p_i)/p_i < 0.05
  // As the original method seems to rely on the cartesian coordinate, and the sensor is different on 3 directions.
  // Here, we modify it to be the max of the three directions, and max between density and pressure jump.
  real sensor{0};
  const real rho = bv(i, j, k, 0), rhoI = 1.0 / rho;
  const real p = bv(i, j, k, 4), pI = 1.0 / p;
  sensor = max(sensor, abs(bv(i + 1, j, k, 0) - rho) * rhoI);
  sensor = max(sensor, abs(bv(i, j + 1, k, 0) - rho) * rhoI);
  sensor = max(sensor, abs(bv(i, j, k + 1, 0) - rho) * rhoI);
  sensor = max(sensor, abs(bv(i + 1, j, k, 4) - p) * pI);
  sensor = max(sensor, abs(bv(i, j + 1, k, 4) - p) * pI);
  sensor = max(sensor, abs(bv(i, j, k + 1, 4) - p) * pI);

  return sensor;
}

__global__ void cfd::compute_shock_sensor(DZone *zone, const DParameter *param) {
  // Calculate the shock sensor, and save the results to the 3-D Array 'shock_sensor'.
  // The closer is the shock_sensor to 1, the stronger is the shock.
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  real sensor{1};
  if (param->shock_sensor == 1) {
    // modified Jameson sensor.
    sensor = jameson_sensor(zone, i, j, k);
  } else if (param->shock_sensor == 2) {
    // density and pressure jump sensor
    sensor = density_pressure_jump_sensor(zone, param->sensor_eps, i, j, k);
  } else {
    // modified Ducros sensor is used by default
    sensor = ducros_sensor(zone, param->sensor_eps, i, j, k);
  }
  zone->shock_sensor(i, j, k) = sensor;
}

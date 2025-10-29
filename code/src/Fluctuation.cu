#include "Fluctuation.cuh"
#include "Constants.h"

void cfd::compute_fluctuation(const Block &block, DZone *zone, DParameter *param, int form, Parameter &parameter,
  std::uniform_real_distribution<real> &dist, std::mt19937 &gen) {
  const int problem_type = parameter.get_int("problem_type");
  if (problem_type == 1 && form == 1) {
    // The fluctuation of Ferrer.
    const real eps1 = dist(gen);
    const real eps2 = dist(gen);
    const auto mx = block.mx, my = block.my, mz = block.mz;
    const int dim{mz == 1 ? 2 : 3};

    dim3 tpb = {32, 8, 2};
    if (dim == 2)
      tpb = {32, 16, 1};
    dim3 BPG{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    ferrer_fluctuation<<<BPG, tpb>>>(zone, param, eps1, eps2);
  }
}

__global__ void cfd::ferrer_fluctuation(DZone *zone, DParameter *param, real eps1, real eps2) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const real alphaUc = param->fluctuation_intensity * param->convective_velocity;
  const real oneDivO2 = -1.0 / (0.0625 * param->delta_omega0 * param->delta_omega0);
  const real Lz = abs(zone->z(i, j, zone->mz - 1) - zone->z(i, j, 0));
  const real LzCoeff = 2 * pi * param->N_spanwise_wave / Lz;
  const real xx0 = (zone->x(i, j, k) - param->x0_fluc) * (zone->x(i, j, k) - param->x0_fluc);
  const real yy0 = (zone->y(i, j, k) - param->y0_fluc) * (zone->y(i, j, k) - param->y0_fluc);
  const real zz0 = (zone->z(i, j, k) - param->z0_fluc);
  const real vp = eps1 * alphaUc * exp(oneDivO2 * (xx0 + yy0)) * cos(LzCoeff * zz0 + eps2 * pi);

  zone->fluc_val(i, j, k, 0) = vp;
}

__global__ void cfd::update_values_with_fluctuations(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  auto &bv = zone->bv;
  auto &cv = zone->cv;
  const auto &fluc_val = zone->fluc_val;
  if (param->problem_type == 1 && param->fluctuation_form == 1) {
    // only v is updated
    const real v0 = bv(i, j, k, 2);
    real v1 = v0;
    v1 += fluc_val(i, j, k, 0);
    bv(i, j, k, 2) = v1;
    cv(i, j, k, 2) = cv(i, j, k, 0) * v1;
    cv(i, j, k, 4) += cv(i, j, k, 0) * 0.5 * (v1 * v1 - v0 * v0); // update the kinetic energy
  }
}

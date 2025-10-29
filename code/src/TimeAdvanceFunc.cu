#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include <mpi.h>
#include "gxl_lib/MyAtomic.cuh"

__global__ void cfd::store_last_step(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  zone->bv_last(i, j, k, 0) = bv(i, j, k, 0);
  zone->bv_last(i, j, k, 1) = sqrt(
    bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
  zone->bv_last(i, j, k, 2) = bv(i, j, k, 4);
  zone->bv_last(i, j, k, 3) = bv(i, j, k, 5);
}

__global__ void cfd::local_time_step_without_reaction(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &metric{zone->metric};
  const auto &bv = zone->bv;
  const int dim{zone->mz == 1 ? 2 : 3};

  const real grad_xi =   norm3d(metric(i, j, k, 0), metric(i, j, k, 1), metric(i, j, k, 2));
  const real grad_eta =  norm3d(metric(i, j, k, 3), metric(i, j, k, 4), metric(i, j, k, 5));
  const real grad_zeta = norm3d(metric(i, j, k, 6), metric(i, j, k, 7), metric(i, j, k, 8));

  const real u{bv(i, j, k, 1)}, v{bv(i, j, k, 2)}, w{bv(i, j, k, 3)};
  const real U = u * metric(i, j, k, 0) + v * metric(i, j, k, 1) + w *metric(i, j, k, 2);
  const real V = u * metric(i, j, k, 3) + v * metric(i, j, k, 4) + w * metric(i, j, k, 5);
  const real W = u * metric(i, j, k, 6) + v * metric(i, j, k, 7) + w * metric(i, j, k, 8);

  real acoustic_speed{0};
  acoustic_speed = zone->acoustic_speed(i, j, k);

  auto &inviscid_spectral_radius = zone->inv_spectr_rad(i, j, k);
  inviscid_spectral_radius[0] = std::abs(U) + acoustic_speed * grad_xi;
  inviscid_spectral_radius[1] = std::abs(V) + acoustic_speed * grad_eta;
  real max_spectral_radius = max(inviscid_spectral_radius[0], inviscid_spectral_radius[1]);
  inviscid_spectral_radius[2] = 0;
  if (dim == 3) {
    inviscid_spectral_radius[2] = std::abs(W) + acoustic_speed * grad_zeta;
    max_spectral_radius = max(max_spectral_radius, inviscid_spectral_radius[2]);
  }
  // const real spectral_radius_inv =
  //     inviscid_spectral_radius[0] + inviscid_spectral_radius[1] + inviscid_spectral_radius[2];

  // Next, compute the viscous spectral radius
  real max_length{grad_xi};
  max_length = max(max_length, grad_eta);
  if (dim == 3)
    max_length = max(max_length, grad_zeta);

  real max_diffuse_vel{0.0};
  const real iRho = 1.0 / bv(i, j, k, 0);
  max_diffuse_vel = zone->mul(i, j, k) * iRho;
  max_diffuse_vel = max(max_diffuse_vel,
                        zone->thermal_conductivity(i, j, k) * iRho * zone->gamma(i, j, k) / zone->cp(i, j, k));
  real max_rhoD{0};
  for (int l = 0; l < param->n_spec; ++l) {
    max_rhoD = max(max_rhoD, zone->rho_D(i, j, k, l));
    // D[l] = zone->rho_D(i, j, k, l) * iRho;
  }
  max_diffuse_vel = max(max_diffuse_vel, max_rhoD * iRho);


  max_spectral_radius = max(max_spectral_radius, max_length * max_length * max_diffuse_vel);
  const real dt = param->cfl / max_spectral_radius;

  zone->dt_local(i, j, k) = dt;
}

__global__ void cfd::compute_square_of_dbv(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->bv_last;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  const real vel = sqrt(
    bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
  bv_last(i, j, k, 1) = (vel - bv_last(i, j, k, 1)) * (vel - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

real cfd::global_time_step(const Mesh &mesh, const Parameter &parameter, const std::vector<Field> &field) {
  real dt{1e+6};

  constexpr int TPB{128};
  real dt_block;
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, min_of_arr, TPB, 0);
  for (int b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const int size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    min_of_arr<<<n_blocks, TPB>>>(field[b].h_ptr->dt_local.data(), size); //, TPB * sizeof(real)
    min_of_arr<<<1, TPB>>>(field[b].h_ptr->dt_local.data(), n_blocks);    //, TPB * sizeof(real)
    cudaMemcpy(&dt_block, field[b].h_ptr->dt_local.data(), sizeof(real), cudaMemcpyDeviceToHost);
    dt = std::min(dt, dt_block);
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    const real dt_temp{dt};
    MPI_Allreduce(&dt_temp, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }

  return dt;
}

__global__ void cfd::update_physical_time(DParameter *param, real t) {
  param->physical_time = t;
}

__global__ void cfd::min_of_arr(real *arr, int size) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int t = static_cast<int>(threadIdx.x);

  if (i >= size) {
    return;
  }
  real inp{1e+6};
  for (int idx = i; idx < size; idx += static_cast<int>(blockDim.x * gridDim.x)) {
    inp = min(inp, arr[idx]);
  }
  __syncthreads();

  inp = block_reduce_min(inp, i, size);
  __syncthreads();

  if (t == 0) {
    arr[blockIdx.x] = inp;
  }
}

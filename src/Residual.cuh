#pragma once

#include "Driver.cuh"

namespace cfd {
template<int N> __global__ void reduction_of_dv_squared(real *arr, int size);

__global__ void reduction_of_dv_squared(real *arr, int size);

__global__ void check_nan(DZone *zone, int blk, int myid, int n_scalar);

template<MixtureModel mix_model> real compute_residual(Driver<mix_model> &driver, int step);

void steady_screen_output(int step, real err_max, gxl::Time &time, const std::array<real, 4> &res);

void unsteady_screen_output(int step, real err_max, gxl::Time &time, const std::array<real, 4> &res, real dt,
  real solution_time);
} // cfd

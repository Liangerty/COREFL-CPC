#pragma once

#include "Define.h"
#include "Driver.cuh"

namespace cfd {
__global__ void compute_qn_star(DZone *zone, int n_var, real dt_global);

__global__ void compute_modified_rhs(DZone *zone, int n_var, real dt_global);

bool inner_converged(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, int iter,
  std::array<real, 4> &res_scale, int myid, int step, int &inner_iter);

__global__ void compute_square_of_dbv_wrt_last_inner_iter(DZone *zone);

__global__ void store_last_iter(DZone *zone);

template<MixtureModel mix_model> void dual_time_stepping(Driver<mix_model> &driver);

template<MixtureModel mixture_model> void dual_time_stepping_implicit_treat(const Block &block, DParameter *param,
  DZone *d_ptr, DZone *h_ptr, Parameter &parameter, DBoundCond &bound_cond, real diag_factor);
}

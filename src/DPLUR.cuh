#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"

namespace cfd {
struct DZone;

__global__ void compute_DQ_0(DZone *zone, const DParameter *param, real diag_factor = 0);

template<MixtureModel mixture_model> __device__ void compute_jacobian_times_dq(const DParameter *param, DZone *zone,
  int i, int j, int k, int dir, real pm_spectral_radius, real *convJacTimesDq);

template<MixtureModel mixture_model> __global__ void DPLUR_inner_iteration(const DParameter *param, DZone *zone,
  real diag_factor = 0);

__global__ void convert_dq_back_to_dqDt(DZone *zone, const DParameter *param);

struct DBoundCond;

void set_wall_dq_to_0(const Block &block, const DParameter *param, DZone *zone, const DBoundCond &bound_cond,
  bool ngg_extended);

template<MixtureModel mixture_model> void DPLUR(const Block &block, const DParameter *param, DZone *d_ptr, DZone *h_ptr,
  const Parameter &parameter, DBoundCond &bound_cond, real diag_factor = 0);
}

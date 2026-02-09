#pragma once

#include "Define.h"

namespace cfd {
template<MixtureModel mix_model> __device__ void riemannSolver_ausmPlus(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared);

template<MixtureModel mix_model> __device__ void riemannSolver_hllc(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared);

template<MixtureModel mixtureModel> __device__ void compute_half_sum_left_right_flux(const real *pv_l, const real *pv_r,
  DParameter *param, const real *jac, const real *metric, int i_shared, real *fc);

template<MixtureModel mix_model> __device__ void riemannSolver_Roe(DZone *zone, real *pv, int tid, DParameter *param,
  real *fc, real *metric, const real *jac, const real *entropy_fix_delta);

template<MixtureModel mix_model> __device__ void riemannSolver_laxFriedrich(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared);

template<> __device__ void riemannSolver_laxFriedrich<MixtureModel::Air>(const real *pv_l, const real *pv_r,
  DParameter *param, int tid, const real *metric, const real *jac, real *fc, int i_shared);
}

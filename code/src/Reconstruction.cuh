#pragma once

#include "Define.h"

namespace cfd {
__device__ void first_order_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var);

__device__ void
MUSCL_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var, int limiter);

__device__ void
NND2_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var, int limiter);
} // cfd

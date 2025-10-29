#pragma once
#include "Define.h"

namespace cfd{
struct DParameter;
struct DZone;
struct Species;

__device__ void compute_enthalpy(real t, real *enthalpy, const DParameter* param);
// __device__ void compute_enthalpy_1(real t, real *enthalpy, const DParameter* param);

__device__ void compute_cp(real t, real *cp, DParameter* param);
// __device__ void compute_cp_1(real t, real *cp, DParameter* param);

__device__ void compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param);
// __device__ void compute_enthalpy_and_cp_1(real t, real *enthalpy, real *cp, const DParameter *param);

__device__ void compute_gibbs_div_rt(real t, const DParameter* param, real* gibbs_rt);
// __device__ void compute_gibbs_div_rt_1(real t, real* gibbs_rt);
}

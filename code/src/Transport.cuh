#pragma once

#include "Define.h"
#include "ChemData.h"

namespace cfd {
__host__ __device__ real Sutherland(real temperature);

real compute_viscosity(real temperature, real mw_total, real const *Y, const Species &spec);

struct DParameter;
struct DZone;

__device__ void compute_transport_property(int i, int j, int k, real temperature, real mw_total, real *cp,
  DParameter *param, DZone *zone);

__device__ real compute_viscosity(real temperature, real mw_total, real const *Y, DParameter *param);

__device__ real compute_Omega_D(real t_red);

__device__ real
compute_viscosity(int i, int j, int k, real temperature, real mw_total, DParameter *param, const DZone *zone);
}

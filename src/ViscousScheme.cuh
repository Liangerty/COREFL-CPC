#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "Field.h"

namespace cfd {
struct DParameter;

template<MixtureModel mix_model>
void compute_viscous_flux(const Mesh &mesh, std::vector<Field> &field, DParameter *param, const Parameter &parameter);

__global__ void compute_dFv_dx(DZone *zone, const DParameter *param);

__global__ void compute_dGv_dy(DZone *zone, const DParameter *param);

__global__ void compute_dHv_dz(DZone *zone, const DParameter *param);

template<int ORDER = 8>
__device__ real d_dXi(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nx,
  int phyBoundLeft, int phyBoundRight);

template<int ORDER = 8>
__device__ real d_dEta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int ny,
  int phyBoundLeft, int phyBoundRight);

template<int ORDER = 8>
__device__ real d_dZeta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nz,
  int phyBoundLeft, int phyBoundRight);

template<MixtureModel mix_model, int ORDER = 8>
__global__ void compute_viscous_flux_collocated(DZone *zone, const DParameter *param);

template<int ORDER = 8>
__global__ void compute_viscous_flux_collocated_scalar(DZone *zone, const DParameter *param);

template<int ORDER = 8> __global__ void compute_viscous_flux_derivative(DZone *zone, const DParameter *param);

template<MixtureModel mix_model> __global__ void compute_fv_2nd_order(DZone *zone, DParameter *param);

template<MixtureModel mix_model> __global__ void compute_gv_2nd_order(DZone *zone, DParameter *param);

template<MixtureModel mix_model> __global__ void compute_hv_2nd_order(DZone *zone, DParameter *param);
} // cfd

#pragma once

#include "Mesh.h"
// #include "gxl_lib/Array.cuh"

namespace cfd {
void compute_jac_metric(int myid, Block &block, Parameter &parameter);

// template<int ORDER> __global__ void compute_xyz123(ggxl::Array3D<real> *x_d, ggxl::Array3D<real> *y_d,
//   ggxl::Array3D<real> *z_d, ggxl::VectorField3D<real> *xyz123, int nx, int ny, int nz, int ng);
//
// template<int ORDER> __global__ void compute_D123(ggxl::Array3D<real> *x_d, ggxl::Array3D<real> *y_d,
//   ggxl::Array3D<real> *z_d, ggxl::VectorField3D<real> *xyz123, ggxl::VectorField3D<real> *D123,
//   int nx, int ny, int nz, int ng);
//
// template<int ORDER> __global__ void compute_metric_jac_symm_con(ggxl::VectorField3D<real> *xyz123,
//   ggxl::VectorField3D<real> *D123, ggxl::Array3D<real> *jac, ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> *metric, int nx,
//   int ny, int nz, int ng);
} // namespace cfd

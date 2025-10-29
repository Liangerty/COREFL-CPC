#include "Mesh.cuh"
#include "Parameter.h"
#include "Mesh.h"
#include <mpi.h>
#include <fmt/core.h>

#include "gxl_lib/Array.cuh"
#include "gxl_lib/Math.cuh"
#include "gxl_lib/Math.hpp"

namespace cfd {
template<int ORDER> __global__ void compute_xyz123(ggxl::Array3D<real> *x_d, ggxl::Array3D<real> *y_d,
  ggxl::Array3D<real> *z_d, ggxl::VectorField3D<real> *xyz123, int nx, int ny, int nz, int ng) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ng;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ng;
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) - ng;
  if (i >= nx + ng || j >= ny + ng || k >= nz + ng) {
    return;
  }

  const real x = (*x_d)(i, j, k), y = (*y_d)(i, j, k), z = (*z_d)(i, j, k);
  const real x1 = ggxl::Derivatives<ORDER, 1>(*x_d, i, j, k, nx, ng);
  const real y1 = ggxl::Derivatives<ORDER, 1>(*y_d, i, j, k, nx, ng);
  const real z1 = ggxl::Derivatives<ORDER, 1>(*z_d, i, j, k, nx, ng);
  const real x2 = ggxl::Derivatives<ORDER, 2>(*x_d, i, j, k, ny, ng);
  const real y2 = ggxl::Derivatives<ORDER, 2>(*y_d, i, j, k, ny, ng);
  const real z2 = ggxl::Derivatives<ORDER, 2>(*z_d, i, j, k, ny, ng);
  const real x3 = ggxl::Derivatives<ORDER, 3>(*x_d, i, j, k, nz, ng);
  const real y3 = ggxl::Derivatives<ORDER, 3>(*y_d, i, j, k, nz, ng);
  const real z3 = ggxl::Derivatives<ORDER, 3>(*z_d, i, j, k, nz, ng);

  (*xyz123)(i, j, k, 0) = x1 * y - y1 * x; // x1y_y1x - 0
  (*xyz123)(i, j, k, 1) = y2 * x - x2 * y; // y2x_x2y - 1
  (*xyz123)(i, j, k, 2) = y1 * z - z1 * y; // y1z_z1y - 2
  (*xyz123)(i, j, k, 3) = z2 * y - y2 * z; // z2y_y2z - 3

  (*xyz123)(i, j, k, 4) = z1 * x - x1 * z; // z1x_x1z - 4
  (*xyz123)(i, j, k, 5) = x2 * z - z2 * x; // x2z_z2x - 5
  (*xyz123)(i, j, k, 6) = x3 * y - y3 * x; // x3y_y3x - 6
  (*xyz123)(i, j, k, 7) = y1 * x - x1 * y; // y1x_x1y - 7

  (*xyz123)(i, j, k, 8) = y3 * z - z3 * y;  // y3z_z3y - 8
  (*xyz123)(i, j, k, 9) = z1 * y - y1 * z;  // z1y_y1z - 9
  (*xyz123)(i, j, k, 10) = z3 * x - x3 * z; // z3x_x3z - 10
  (*xyz123)(i, j, k, 11) = x1 * z - z1 * x; // x1z_z1x - 11

  (*xyz123)(i, j, k, 12) = x2 * y - y2 * x; // x2y_y2x - 12
  (*xyz123)(i, j, k, 13) = y3 * x - x3 * y; // y3x_x3y - 13
  (*xyz123)(i, j, k, 14) = y2 * z - z2 * y; // y2z_z2y - 14
  (*xyz123)(i, j, k, 15) = z3 * y - y3 * z; // z3y_y3z - 15

  (*xyz123)(i, j, k, 16) = z2 * x - x2 * z; // z2x_x2z - 16
  (*xyz123)(i, j, k, 17) = x3 * z - z3 * x; // x3z_z3x - 17
}

template<int ORDER> __global__ void compute_D123(ggxl::Array3D<real> *x_d, ggxl::Array3D<real> *y_d,
  ggxl::Array3D<real> *z_d, ggxl::VectorField3D<real> *xyz123, ggxl::VectorField3D<real> *D123,
  int nx, int ny, int nz, int ng) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ng;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ng;
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) - ng;
  if (i >= nx + ng || j >= ny + ng || k >= nz + ng) {
    return;
  }

  const real x = (*x_d)(i, j, k), y = (*y_d)(i, j, k), z = (*z_d)(i, j, k);
  (*D123)(i, j, k, 2) = (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 0, ny, ng) +
                         ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 1, nx, ng)) * z +
                        (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 2, ny, ng) +
                         ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 3, nx, ng)) * x +
                        (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 4, ny, ng) +
                         ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 5, nx, ng)) * y;

  (*D123)(i, j, k, 1) = (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 6, nx, ng) +
                         ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 7, nz, ng)) * z +
                        (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 8, nx, ng) +
                         ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 9, nz, ng)) * x +
                        (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 10, nx, ng) +
                         ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 11, nz, ng)) * y;

  (*D123)(i, j, k, 0) = (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 12, nz, ng) +
                         ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 13, ny, ng)) * z +
                        (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 14, nz, ng) +
                         ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 15, ny, ng)) * x +
                        (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 16, nz, ng) +
                         ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 17, ny, ng)) * y;
}

template<int ORDER> __global__ void compute_metric_jac_symm_con(ggxl::VectorField3D<real> *xyz123,
  ggxl::VectorField3D<real> *D123, ggxl::Array3D<real> *jac, ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> *metric, int nx,
  int ny, int nz, int ng) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ng;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ng;
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) - ng;
  if (i >= nx + ng || j >= ny + ng || k >= nz + ng) {
    return;
  }

  const real ja = (ggxl::Derivatives<ORDER, 3>(*D123, i, j, k, 2, nz, ng) +
                   ggxl::Derivatives<ORDER, 2>(*D123, i, j, k, 1, ny, ng) +
                   ggxl::Derivatives<ORDER, 1>(*D123, i, j, k, 0, nx, ng)) / 6.0;
  const real jI = 0.5 / ja;
  const real xi_x = (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 14, nz, ng) -
                     ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 8, ny, ng)) * jI;
  const real xi_y = (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 16, nz, ng) -
                     ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 10, ny, ng)) * jI;
  const real xi_z = (ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 12, nz, ng) -
                     ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 6, ny, ng)) * jI;
  const real eta_x = (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 8, nx, ng) -
                      ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 2, nz, ng)) * jI;
  const real eta_y = (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 10, nx, ng) -
                      ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 4, nz, ng)) * jI;
  const real zeta_x = (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 2, ny, ng) -
                       ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 14, nx, ng)) * jI;
  const real zeta_y = (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 4, ny, ng) -
                       ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 16, nx, ng)) * jI;
  const real zeta_z = (ggxl::Derivatives<ORDER, 2>(*xyz123, i, j, k, 0, ny, ng) -
                       ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 12, nx, ng)) * jI;

  // when this line is added, the result is wrong
  // const real eta_z = (ggxl::Derivatives<ORDER, 1>(*xyz123, i, j, k, 6, nx, ng) -
  //                      ggxl::Derivatives<ORDER, 3>(*xyz123, i, j, k, 0, nz, ng)) * jI;
  real eta_z = 0.5;

  (*jac)(i, j, k) = ja;
  auto &m = (*metric)(i, j, k);
  m(1, 1) = xi_x;
  m(1, 2) = xi_y;
  m(1, 3) = xi_z;
  m(2, 1) = eta_x;
  m(2, 2) = eta_y;
  m(2, 3) = eta_z;
  m(3, 1) = zeta_x;
  m(3, 2) = zeta_y;
  m(3, 3) = zeta_z;
}

void compute_jac_metric(int myid, Block &block, Parameter &parameter) {
  const int n = parameter.get_int("symmetric_conservative_metric");
  if (n == 0) {
    block.compute_jac_metric(myid);
    return;
  }
  // compute on gpu!
  // First, we need to copy the x, y, z to gpu
  const int nx = block.mx, ny = block.my, nz = block.mz, ng = block.ngg;
  const int64_t N = (nx + 2 * ng) * (ny + 2 * ng) * (nz + 2 * ng);
  ggxl::Array3D<real> x_h, y_h, z_h, *x_d, *y_d, *z_d;
  x_h.allocate_memory(nx, ny, nz, ng + 1);
  y_h.allocate_memory(nx, ny, nz, ng + 1);
  z_h.allocate_memory(nx, ny, nz, ng + 1);
  cudaMemcpy(x_h.data(), block.x.data(), ((nx + 2 * ng + 2) * (ny + 2 * ng + 2) * (nz + 2 * ng + 2)) * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(y_h.data(), block.y.data(), ((nx + 2 * ng + 2) * (ny + 2 * ng + 2) * (nz + 2 * ng + 2)) * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(z_h.data(), block.z.data(), ((nx + 2 * ng + 2) * (ny + 2 * ng + 2) * (nz + 2 * ng + 2)) * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMalloc(&x_d, sizeof(ggxl::Array3D<real>));
  cudaMalloc(&y_d, sizeof(ggxl::Array3D<real>));
  cudaMalloc(&z_d, sizeof(ggxl::Array3D<real>));
  cudaMemcpy(x_d, &x_h, sizeof(ggxl::Array3D<real>), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, &y_h, sizeof(ggxl::Array3D<real>), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, &z_h, sizeof(ggxl::Array3D<real>), cudaMemcpyHostToDevice);

  // Then, we need to compute the metric
  // compute the first derivatives
  ggxl::VectorField3D<real> xyz123_h, *xyz123_d;
  xyz123_h.allocate_memory(nx, ny, nz, 18, ng);
  cudaMalloc(&xyz123_d, sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(xyz123_d, &xyz123_h, sizeof(ggxl::VectorField3D<real>), cudaMemcpyHostToDevice);

  ggxl::VectorField3D<real> D123_h, *D123_d;
  D123_h.allocate_memory(nx, ny, nz, 3, ng);
  cudaMalloc(&D123_d, sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(D123_d, &D123_h, sizeof(ggxl::VectorField3D<real>), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // ggxl::Array3D<real> jac_h, *jac_d;
  // jac_h.allocate_memory(nx, ny, nz, ng);
  // cudaMalloc(&jac_d, sizeof(ggxl::Array3D<real>));
  // cudaMemcpy(jac_d, &jac_h, sizeof(ggxl::Array3D<real>), cudaMemcpyHostToDevice);
  // ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric_h, *metric_d;
  // metric_h.allocate_memory(nx, ny, nz, ng);
  // cudaMalloc(&metric_d, sizeof(ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>>));
  // cudaMemcpy(metric_d, &metric_h, sizeof(ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>>), cudaMemcpyHostToDevice);

  const dim3 TPB{8, 4, 4};
  const dim3 BPG{
    (nx + 2 * ng + TPB.x - 1) / TPB.x, (ny + 2 * ng + TPB.y - 1) / TPB.y, (nz + 2 * ng + TPB.z - 1) / TPB.z
  };
  switch (n) {
    case 4:
      compute_xyz123<4><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, nx, ny, nz, ng);
      compute_D123<4><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, D123_d, nx, ny, nz, ng);
      break;
    case 6:
      compute_xyz123<6><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, nx, ny, nz, ng);
      compute_D123<6><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, D123_d, nx, ny, nz, ng);
      break;
    case 2:
    default:
      compute_xyz123<2><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, nx, ny, nz, ng);
      compute_D123<2><<<BPG, TPB>>>(x_d, y_d, z_d, xyz123_d, D123_d, nx, ny, nz, ng);
      break;
  }

  gxl::VectorField3D<real> xyz123, D123;
  xyz123.resize(nx, ny, nz, 18, ng);
  D123.resize(nx, ny, nz, 3, ng);
  cudaMemcpy(xyz123.data(), xyz123_h.data(), N * 18 * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(D123.data(), D123_h.data(), N * 3 * sizeof(real), cudaMemcpyDeviceToHost);

  for (int k = -ng; k < nz + ng; ++k) {
    for (int j = -ng; j < ny + ng; ++j) {
      for (int i = -ng; i < nx + ng; ++i) {
        const real jac = (gxl::Derivatives<6>(D123, i, j, k, 2, 3) +
                          gxl::Derivatives<6>(D123, i, j, k, 1, 2) +
                          gxl::Derivatives<6>(D123, i, j, k, 0, 1)) / 6.0;
        if (jac <= 0) {
          fmt::print("Negative Jacobian from process {}, Block {}, index is ({}, {}, {}), "
                     "Stop simulation.\n", myid, block.block_id, i, j, k);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        block.jacobian(i, j, k) = jac;
        const real jI = 0.5 / jac;
        auto &m = block.metric;
        m(i, j, k, 0) = (gxl::Derivatives<6>(xyz123, i, j, k, 14, 3) - gxl::Derivatives<6>(xyz123, i, j, k, 8, 2)) * jI;
        m(i, j, k, 1) = (gxl::Derivatives<6>(xyz123, i, j, k, 16, 3) - gxl::Derivatives<6>(xyz123, i, j, k, 10, 2)) * jI;
        m(i, j, k, 2) = (gxl::Derivatives<6>(xyz123, i, j, k, 12, 3) - gxl::Derivatives<6>(xyz123, i, j, k, 6, 2)) * jI;
        m(i, j, k, 3) = (gxl::Derivatives<6>(xyz123, i, j, k, 8, 1) - gxl::Derivatives<6>(xyz123, i, j, k, 2, 3)) * jI;
        m(i, j, k, 4) = (gxl::Derivatives<6>(xyz123, i, j, k, 10, 1) - gxl::Derivatives<6>(xyz123, i, j, k, 4, 3)) * jI;
        m(i, j, k, 5) = (gxl::Derivatives<6>(xyz123, i, j, k, 6, 1) - gxl::Derivatives<6>(xyz123, i, j, k, 0, 3)) * jI;
        m(i, j, k, 6) = (gxl::Derivatives<6>(xyz123, i, j, k, 2, 2) - gxl::Derivatives<6>(xyz123, i, j, k, 14, 1)) * jI;
        m(i, j, k, 7) = (gxl::Derivatives<6>(xyz123, i, j, k, 4, 2) - gxl::Derivatives<6>(xyz123, i, j, k, 16, 1)) * jI;
        m(i, j, k, 8) = (gxl::Derivatives<6>(xyz123, i, j, k, 0, 2) - gxl::Derivatives<6>(xyz123, i, j, k, 12, 1)) * jI;
      }
    }
  }

  // free the memory
  cudaDeviceSynchronize();
  x_h.deallocate_memory();
  y_h.deallocate_memory();
  z_h.deallocate_memory();
  xyz123_h.deallocate_memory();
  D123_h.deallocate_memory();
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(xyz123_d);
  cudaFree(D123_d);
}
} // namespace cfd

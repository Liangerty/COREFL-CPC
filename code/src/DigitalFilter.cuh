#pragma once

#include <curand_kernel.h>
#include "Define.h"
#include "Parameter.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct DBoundCond;

struct DParameter;

struct DZone;

struct Inflow;

void assume_gaussian_reynolds_stress(Parameter &parameter, const DBoundCond &dBoundCond, const std::vector<int> &N1,
                                     const std::vector<std::vector<real>> &y_scaled);

__global__ void
compute_lundMat_with_assumed_gaussian_reynolds_stress(const real *Rij, ggxl::VectorField1D<real> *df_lundMatrix_hPtr,
                                                      int i_face, const real *y_scaled, int my, int ngg);

__global__ void
compute_convolution_kernel(const real *y_scaled, ggxl::VectorField2D<real> *df_by, ggxl::VectorField2D<real> *df_bz,
                           real dz_scaled, int iFace, int my, int ngg);

__global__ void
compute_fluctuations_first_step(ggxl::VectorField3D<real> *fluctuation_dPtr, ggxl::VectorField1D<real> *lundMatrix_dPtr,
                                ggxl::VectorField2D<real> *df_velFluc_old_dPtr, int iFace, int my, int mz, int ngg);

__global__ void
generate_random_numbers_kernel(ggxl::VectorField2D<curandState> *rng_states, ggxl::VectorField2D<real> *random_numbers,
                               int iFace, int my, int mz, int ngg);

__global__ void
remove_mean_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg);

__global__ void
apply_periodic_in_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg);

__global__ void perform_convolution_y(ggxl::VectorField2D<real> *random_numbers, ggxl::VectorField2D<real> *df_by,
                                      ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, int ngg);

__global__ void
perform_convolution_z(ggxl::VectorField2D<real> *df_fy, ggxl::VectorField2D<real> *df_bz,
                      ggxl::VectorField2D<real> *velFluc, int iFace, int my, int mz, int ngg);

__global__ void
Castro_time_correlation_and_fluc_computation(const DParameter *param, DZone *zone, const Inflow *inflow,
                                             ggxl::VectorField2D<real> *velFluc_old,
                                             ggxl::VectorField2D<real> *velFluc_new,
                                             ggxl::VectorField1D<real> *lundMatrix_dPtr,
                                             ggxl::VectorField3D<real> *fluctuation_dPtr, int iFace, int my, int mz,
                                             int ngg);
}

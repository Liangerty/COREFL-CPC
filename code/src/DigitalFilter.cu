#include "DigitalFilter.cuh"
#include "BoundCond.cuh"
#include <ctime>
#include "mpi.h"
#include "Parallel.h"

namespace cfd {
void DBoundCond::initialize_digital_filter(Parameter &parameter, Mesh &mesh) {
  if (parameter.get_int("problem_type") != 1) {
    // The current implementation is only for mixing layers
    return;
  }
  parameter.update_parameter("use_df", false);

  int n_df = 0;
  std::vector<int> df_bc;
  std::vector<std::string> df_related_boundary = {};
  auto &bcs = parameter.get_string_array("boundary_conditions");
  if (n_inflow > 0) {
    int i_inflow = 0;
    for (auto &bc_name: bcs) {
      auto &bc = parameter.get_struct(bc_name);
      auto &type = std::get<std::string>(bc.at("type"));

      if (type == "inflow") {
        int fluc = std::get<int>(bc.at("fluctuation_type"));
        if (fluc == 11) {
          ++n_df;
          df_bc.push_back(std::get<int>(bc.at("label")));
          df_related_boundary.push_back(bc_name);
          df_label[i_inflow] = i_inflow;
        }
        ++i_inflow;
      }
    }
  }

  if (n_df == 0) {
    return;
  }

  // Careful! The current implementation is based on the Touber's implementation, the z direction is assumed to be periodic.
  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    printf("\t\tInitialize digital filter for turbulence inflow generation.\n");
    printf("\t\tThe z direction is assumed to be periodic, and the inflow is assumed to be in the yz(const x) plane!\n");
  }

  // Although the implementation below is for multiple boundaries, the current implementation only supports one boundary.
  std::vector<int> N1, N2, iBlock;
  std::vector<std::vector<real>> scaled_y;
  real y_ref{0}, dz{0};
  const int ngg{parameter.get_int("ngg")};
  if (parameter.get_int("problem_type") == 1) {
    y_ref = parameter.get_real("delta_omega");
  }
  for (int type = 0; type < n_df; ++type) {
    int label = df_bc[type];
    const auto &boundary_name = df_related_boundary[type];
    // Find the corresponding boundary faces
    for (auto blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (auto &b: bs) {
        if (label == b.type_label) {
          ++n_df_face;
          df_related_block.push_back(blk);
          iBlock.push_back(blk);

          int n1{mesh[blk].my}, n2{mesh[blk].mz};
          if (b.face == 1) {
            printf("\tBoundary %s is on the xz plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
            // n1 = mesh[blk].mx;
          } else if (b.face == 2) {
            printf("\tBoundary %s is on the xy plane, which is not supported!\n", boundary_name.c_str());
            MpiParallel::exit();
            // n1 = mesh[blk].mx;
            // n2 = mesh[blk].my;
          }

          int i = b.direction == 1 ? mesh[blk].mx - 1 : 0;
          std::vector<real> scaled_y_temp(mesh[blk].my + 2 * ngg);
          for (int j = -ngg; j < mesh[blk].my + ngg; ++j) {
            scaled_y_temp[j + ngg] = mesh[blk].y(i, j, 0) / y_ref;
          }
          scaled_y.push_back(scaled_y_temp);
          for (int k = 1; k < n2; ++k) {
            dz += mesh[blk].z(i, 0, k) - mesh[blk].z(i, 0, k - 1);
          }
          dz /= n2 - 1;

          printf("\t\tInflow boundary %s with (%d, %d) grid points for digital filter on  process %d, dz = %f.\n",
                 boundary_name.c_str(), n1, n2, myid, dz);

          N1.push_back(n1);
          N2.push_back(n2);
        }
      }
    }
  }
  bool use_df = n_df_face > 0;
  parameter.update_parameter("use_df", use_df);
  if (n_df_face == 0) {
    // When parallel, there may be only one inflow plane in a single processor
    return;
  }

  initialize_df_memory(mesh, N1, N2);
  if (myid == 0)
    printf("\t\tThe memory for the digital filter is allocated.\n");

  // compute lund matrix
  get_digital_filter_lund_matrix(parameter, N1, scaled_y);
  if (myid == 0)
    printf("\t\tThe Lund matrix for the digital filter is computed.\n");

  // compute convolution kernel
  get_digital_filter_convolution_kernel(parameter, N1, scaled_y, dz);
  if (myid == 0)
    printf("\t\tThe convolution kernel for the digital filter is computed.\n");

  int init = parameter.get_int("initial");
  if (init == 1) {
    // continue from previous computation
    bool initialized{true};
    for (int iFace = 0; iFace < n_df_face; ++iFace) {
      std::string filename = "./output/df-p" + std::to_string(myid) + "-f" + std::to_string(iFace) + ".bin";
      MPI_File fp;
      MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
      if (fp == nullptr) {
        printf("Error: cannot open the file %s, the df states will be restarted.\n", filename.c_str());
        initialized = false;
        break;
      }
      int my = N1[iFace], mz = N2[iFace];
      int my_read, mz_read;
      MPI_Offset offset = 0;
      MPI_File_read_at(fp, offset, &my_read, 1, MPI_INT, MPI_STATUS_IGNORE);
      offset += 4;
      MPI_File_read_at(fp, offset, &mz_read, 1, MPI_INT, MPI_STATUS_IGNORE);
      offset += 4;
      if (my != my_read || mz != mz_read) {
        printf(
          "Error: the grid size in the file %s is not consistent with the current grid size, the df states will be restarted.\n",
          filename.c_str());
        initialized = false;
        break;
      }
      int ngg_read, DF_N_read;
      MPI_File_read_at(fp, offset, &ngg_read, 1, MPI_INT, MPI_STATUS_IGNORE);
      offset += 4;
      MPI_File_read_at(fp, offset, &DF_N_read, 1, MPI_INT, MPI_STATUS_IGNORE);
      offset += 4;

      int old_width_single_side = ngg_read + DF_N_read;
      if (old_width_single_side >= ngg + DF_N) {
        MPI_Datatype ty;
        int lSize[2]{my + 2 * old_width_single_side, mz + 2 * old_width_single_side};
        int sSize[2]{my + 2 * ngg + 2 * DF_N, mz + 2 * ngg + 2 * DF_N};
        int start[2]{old_width_single_side - ngg - DF_N, old_width_single_side - ngg - DF_N};
        // The old data type is curandState
        MPI_Datatype mpi_curandState;
        MPI_Type_contiguous(sizeof(curandState), MPI_BYTE, &mpi_curandState);
        MPI_Type_commit(&mpi_curandState);
        MPI_Type_create_subarray(2, lSize, sSize, start, MPI_ORDER_FORTRAN, mpi_curandState, &ty);
        MPI_Type_commit(&ty);

        MPI_File_read_at(fp, offset, df_rng_state_cpu[iFace].data(), 3, ty, MPI_STATUS_IGNORE);
        offset += static_cast<MPI_Offset>((my + 2 * old_width_single_side) * (mz + 2 * old_width_single_side) * 3 *
                                          sizeof(curandState));
        cudaMemcpy(rng_states_hPtr[iFace].data(), df_rng_state_cpu[iFace].data(),
                   (my + 2 * ngg + 2 * DF_N) * (mz + 2 * ngg + 2 * DF_N) * 3 * sizeof(curandState),
                   cudaMemcpyHostToDevice);
        MPI_Type_free(&ty);
        MPI_Type_free(&mpi_curandState);
      } else {
        auto &rng_cpu = df_rng_state_cpu[iFace];
        for (int l = 0; l < 3; ++l) {
          for (int k = -old_width_single_side; k < mz + old_width_single_side; ++k) {
            for (int j = -old_width_single_side; j < my + old_width_single_side; ++j) {
              MPI_File_read_at(fp, offset, &rng_cpu(j, k, l), 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
              offset += sizeof(curandState);
            }
          }
        }
        cudaMemcpy(rng_states_hPtr[iFace].data(), df_rng_state_cpu[iFace].data(),
                   (my + 2 * ngg + 2 * DF_N) * (mz + 2 * ngg + 2 * DF_N) * 3 * sizeof(curandState),
                   cudaMemcpyHostToDevice);
        dim3 TPB{32, 32};
        dim3 BPG{((my + 2 * ngg + 2 * DF_N - 1) / TPB.x + 1), ((mz + 2 * ngg + 2 * DF_N - 1) / TPB.y + 1)};
        time_t time_curr;
        initialize_rest_rng<<<BPG, TPB>>>(rng_states_dPtr, iFace, time(&time_curr), ngg + DF_N - old_width_single_side,
                                          ngg + DF_N - old_width_single_side, ngg, my, mz);
      }

      // The velocity fluctuation of last step
      if (ngg_read >= ngg) {
        MPI_Datatype ty;
        int lSize[2]{my + 2 * ngg_read, mz + 2 * ngg_read};
        int sSize[2]{my + 2 * ngg, mz + 2 * ngg};
        int start[2]{ngg_read - ngg, ngg_read - ngg};
        MPI_Type_create_subarray(2, lSize, sSize, start, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
        MPI_Type_commit(&ty);

        MPI_File_read_at(fp, offset, df_velFluc_cpu[iFace].data(), 3, ty, MPI_STATUS_IGNORE);
        // offset += (MPI_Offset) ((my + 2 * ngg_read) * (mz + 2 * ngg_read) * 3 * sizeof(real));
        MPI_Type_free(&ty);
        cudaMemcpy(df_velFluc_old_hPtr[iFace].data(), df_velFluc_cpu[iFace].data(),
                   (my + 2 * ngg) * (mz + 2 * ngg) * 3 * sizeof(real), cudaMemcpyHostToDevice);
      } else {
        auto &velFluc_cpu = df_velFluc_cpu[iFace];
        for (int l = 0; l < 3; ++l) {
          for (int k = -ngg_read; k < mz + ngg_read; ++k) {
            for (int j = -ngg_read; j < my + ngg_read; ++j) {
              MPI_File_read_at(fp, offset, &velFluc_cpu(j, k, l), 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
              offset += sizeof(real);
            }
          }
        }
        cudaMemcpy(df_velFluc_old_hPtr[iFace].data(), df_velFluc_cpu[iFace].data(),
                   (my + 2 * ngg) * (mz + 2 * ngg) * 3 * sizeof(real), cudaMemcpyHostToDevice);
      }
    }
    if (initialized)
      return;
  }

  // From start
  // initialize the random number states
  for (int i = 0; i < n_df_face; ++i) {
    int sz = (N1[i] + 2 * DF_N + 2 * ngg) * (N2[i] + 2 * DF_N + 2 * ngg) * 3;
    dim3 TPB = {1024, 1, 1};
    dim3 BPG = {(sz - 1) / TPB.x + 1, 1, 1};
    // Get the current time
    time_t time_curr;
    initialize_rng<<<BPG, TPB>>>(rng_states_hPtr[i].data(), sz, time(&time_curr));
    // Initialize the old velocity fluctuation
    generate_random_numbers(i, N1[i], N2[i], ngg);
    apply_convolution(i, N1[i], N2[i], ngg);
    sz = (N1[i] + 2 * ngg) * (N2[i] + 2 * ngg) * 3;
    cudaMemcpy(df_velFluc_old_hPtr[i].data(), df_velFluc_new_hPtr[i].data(), sz * sizeof(real),
               cudaMemcpyDeviceToDevice);
    TPB = {32, 32};
    BPG = {((N1[i] + 2 * ngg + TPB.x - 1) / TPB.x), ((N2[i] + 2 * ngg + TPB.y - 1) / TPB.y)};
    compute_fluctuations_first_step<<<BPG, TPB>>>(fluctuation_dPtr, df_lundMatrix_dPtr, df_velFluc_old_dPtr, i, N1[i],
                                                  N2[i], ngg);
  }
  if (myid == 0)
    printf("\t\tThe velocity fluctuations are computed.\n");
}

void DBoundCond::initialize_df_memory(const Mesh &mesh, const std::vector<int> &N1, const std::vector<int> &N2) {
  std::vector<ggxl::VectorField1D<real>> df_lundMatrix_hPtr(n_df_face);
  std::vector<ggxl::VectorField2D<real>> df_by_hPtr(n_df_face);
  std::vector<ggxl::VectorField2D<real>> df_bz_hPtr(n_df_face);
  rng_states_hPtr = new ggxl::VectorField2D<curandState>[n_df_face];
  random_values_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  std::vector<ggxl::VectorField2D<real>> df_fy_hPtr(n_df_face);
  df_velFluc_old_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  df_velFluc_new_hPtr = new ggxl::VectorField2D<real>[n_df_face];
  std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr(n_df_face);

  df_rng_state_cpu = new ggxl::VectorField2DHost<curandState>[n_df_face];
  df_velFluc_cpu = new ggxl::VectorField2DHost<real>[n_df_face];

  const int ngg = mesh.ngg;
  for (int i = 0; i < n_df_face; ++i) {
    const int my = N1[i], mz = N2[i];
    df_lundMatrix_hPtr[i].allocate_memory(my, 6, ngg);
    df_by_hPtr[i].allocate_memory(my, 1, 3, std::max(ngg, DF_N));
    df_bz_hPtr[i].allocate_memory(my, 1, 3, std::max(ngg, DF_N));
    rng_states_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    random_values_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_fy_hPtr[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_velFluc_old_hPtr[i].allocate_memory(my, mz, 3, ngg);
    df_velFluc_new_hPtr[i].allocate_memory(my, mz, 3, ngg);
    fluctuation_hPtr[i].allocate_memory(1, my, mz, 3, ngg);

    df_rng_state_cpu[i].allocate_memory(my, mz, 3, DF_N + ngg);
    df_velFluc_cpu[i].allocate_memory(my, mz, 3, ngg);
  }

  cudaMalloc(&df_lundMatrix_dPtr, n_df_face * sizeof(ggxl::VectorField1D<real>));
  cudaMemcpy(df_lundMatrix_dPtr, df_lundMatrix_hPtr.data(), n_df_face * sizeof(ggxl::VectorField1D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_by_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_by_dPtr, df_by_hPtr.data(), n_df_face * sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&df_bz_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_bz_dPtr, df_bz_hPtr.data(), n_df_face * sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&rng_states_dPtr, n_df_face * sizeof(ggxl::VectorField2D<curandState>));
  cudaMemcpy(rng_states_dPtr, rng_states_hPtr, n_df_face * sizeof(ggxl::VectorField2D<curandState>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&random_values_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(random_values_dPtr, random_values_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_fy_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_fy_dPtr, df_fy_hPtr.data(), n_df_face * sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);
  cudaMalloc(&df_velFluc_old_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_velFluc_old_dPtr, df_velFluc_old_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&df_velFluc_new_dPtr, n_df_face * sizeof(ggxl::VectorField2D<real>));
  cudaMemcpy(df_velFluc_new_dPtr, df_velFluc_new_hPtr, n_df_face * sizeof(ggxl::VectorField2D<real>),
             cudaMemcpyHostToDevice);
  cudaMalloc(&fluctuation_dPtr, n_df_face * sizeof(ggxl::VectorField3D<real>));
  cudaMemcpy(fluctuation_dPtr, fluctuation_hPtr.data(), n_df_face * sizeof(ggxl::VectorField3D<real>),
             cudaMemcpyHostToDevice);
}

void DBoundCond::get_digital_filter_lund_matrix(Parameter &parameter, const std::vector<int> &N1,
                                                const std::vector<std::vector<real>> &scaled_y) const {
  const int method = parameter.get_int("reynolds_stress_supplier");
  if (method == 2) {
    // Assume the Reynolds stress is Gaussian
    assume_gaussian_reynolds_stress(parameter, *this, N1, scaled_y);
  } else {
    printf("The method %d is not supported for the Reynolds stress supplier.\n", method);
    MpiParallel::exit();
  }
}

void
assume_gaussian_reynolds_stress(Parameter &parameter, const DBoundCond &dBoundCond, const std::vector<int> &N1,
                                const std::vector<std::vector<real>> &y_scaled) {
  const auto Rij = parameter.get_real_array("df_reynolds_gaussian_peak");

  real *Rij_dPtr;
  cudaMalloc(&Rij_dPtr, 6 * sizeof(real));
  cudaMemcpy(Rij_dPtr, Rij.data(), 6 * sizeof(real), cudaMemcpyHostToDevice);

  // Assume the Reynolds stress is Gaussian
  const int ngg = parameter.get_int("ngg");
  for (int i = 0; i < dBoundCond.n_df_face; ++i) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[i].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[i].data(), y_scaled[i].size() * sizeof(real), cudaMemcpyHostToDevice);

    int TPB = 512;
    int BPG = (N1[i] + 2 * ngg + TPB - 1) / TPB;
    compute_lundMat_with_assumed_gaussian_reynolds_stress<<<BPG, TPB>>>(Rij_dPtr, dBoundCond.df_lundMatrix_dPtr, i,
                                                                        y_scaled_dPtr, N1[i], ngg);

    cudaFree(y_scaled_dPtr);
  }
  cudaFree(Rij_dPtr);
}

void DBoundCond::get_digital_filter_convolution_kernel(Parameter &parameter, const std::vector<int> &N1,
                                                       const std::vector<std::vector<real>> &y_scaled, real dz) const {
  // For mixing layer, the length scale is the initial vorticity thickness for 3 directions
  const real DF_IntegralLength = parameter.get_real("delta_omega");

  const int ngg = parameter.get_int("ngg");

  for (int iFace = 0; iFace < n_df_face; ++iFace) {
    real *y_scaled_dPtr;
    cudaMalloc(&y_scaled_dPtr, y_scaled[iFace].size() * sizeof(real));
    cudaMemcpy(y_scaled_dPtr, y_scaled[iFace].data(), y_scaled[iFace].size() * sizeof(real), cudaMemcpyHostToDevice);

    int TPB = 512;
    int BPG = (N1[iFace] + 2 * ngg + TPB - 1) / TPB;
    compute_convolution_kernel<<<BPG, TPB>>>(y_scaled_dPtr, df_by_dPtr, df_bz_dPtr, dz / DF_IntegralLength, iFace,
                                             N1[iFace], ngg);
    cudaFree(y_scaled_dPtr);
  }
}

void DBoundCond::generate_random_numbers(int iFace, int my, int mz, int ngg) const {
  // 1. Generate random numbers
  dim3 TPB(32, 32);
  dim3 BPG{
    ((my + 2 * DF_N + 2 * ngg + TPB.x - 1) / TPB.x),
    ((mz + 2 * DF_N + 2 * ngg + TPB.y - 1) / TPB.y)
  };
  generate_random_numbers_kernel<<<BPG, TPB>>>(rng_states_dPtr, random_values_dPtr, iFace, my, mz, ngg);

  // 2. remove the mean spanwise
  TPB = 256;
  BPG = (my + 2 * DF_N + 2 * ngg - 1) / TPB.x + 1;
  remove_mean_spanwise<<<BPG, TPB>>>(random_values_dPtr, iFace, my, mz, ngg);

  // 3. Apply periodic boundary condition in spanwise direction
  TPB = {32, 32};
  BPG = {(my + 2 * DF_N + 2 * ngg - 1) / TPB.x + 1, ((DF_N + ngg + TPB.y - 1) / TPB.y)};
  apply_periodic_in_spanwise<<<BPG, TPB>>>(random_values_dPtr, iFace, my, mz, ngg);
}

void DBoundCond::apply_convolution(int iFace, int my, int mz, int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{
    ((my + 2 * ngg + TPB.x - 1) / TPB.x),
    ((mz + 2 * ngg + 2 * DF_N + TPB.y - 1) / TPB.y)
  };
  perform_convolution_y<<<BPG, TPB>>>(random_values_dPtr, df_by_dPtr, df_fy_dPtr, iFace, my, mz, ngg);

  BPG = {((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  perform_convolution_z<<<BPG, TPB>>>(df_fy_dPtr, df_bz_dPtr, df_velFluc_new_dPtr, iFace, my, mz, ngg);
}

void
DBoundCond::compute_fluctuations(const DParameter *param, DZone *zone, const Inflow *inflowHere, int iFace, int my, int mz,
                                 int ngg) const {
  dim3 TPB(32, 8);
  dim3 BPG{((my + 2 * ngg + TPB.x - 1) / TPB.x), ((mz + 2 * ngg + TPB.y - 1) / TPB.y)};
  Castro_time_correlation_and_fluc_computation<<<BPG, TPB>>>(param, zone, inflowHere, df_velFluc_old_dPtr,
                                                             df_velFluc_new_dPtr, df_lundMatrix_dPtr, fluctuation_dPtr,
                                                             iFace, my, mz, ngg);
}

__global__ void
compute_lundMat_with_assumed_gaussian_reynolds_stress(const real *Rij, ggxl::VectorField1D<real> *df_lundMatrix_hPtr,
                                                      int i_face, const real *y_scaled, int my, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) {
    return;
  }

  auto &mat = df_lundMatrix_hPtr[i_face];
  const real y_ = y_scaled[j + ngg]; // The first element is the ghost cell, the last element is the ghost cell

  // Assume the Reynolds stress is Gaussian, Rij[0] is the peak value, sigma = 1/2
  const real gaussian_y = exp(-y_ * y_ * 0.5 * 4);

  const real R11_y = Rij[0] * gaussian_y;
  const real R12_y = Rij[1] * gaussian_y;
  const real R22_y = Rij[2] * gaussian_y;
  const real R13_y = Rij[3] * gaussian_y;
  const real R23_y = Rij[4] * gaussian_y;
  const real R33_y = Rij[5] * gaussian_y;
//  printf("R(%d, 0:5) = (%e, %e, %e, %e, %e, %e)\n", j, R11_y, R12_y, R22_y, R13_y, R23_y, R33_y);

  mat(j, 0) = sqrt(abs(R11_y));                                                    // a(1, 1)
  mat(j, 1) = mat(j, 0) < 1e-40 ? 0 : R12_y / mat(j, 0);                           // a(2, 1)
  mat(j, 2) = sqrt(abs(R22_y - mat(j, 1) * mat(j, 1)));                            // a(2, 2)
  mat(j, 3) = mat(j, 0) < 1e-40 ? 0 : R13_y / mat(j, 0);                           // a(3, 1)
  mat(j, 4) = mat(j, 2) < 1e-40 ? 0 : (R23_y - mat(j, 3) * mat(j, 1)) / mat(j, 2); // a(3, 2)
  mat(j, 5) = sqrt(abs(R33_y - mat(j, 3) * mat(j, 3) - mat(j, 4) * mat(j, 4)));    // a(3, 3)
//  printf("Lund(%d, 0:5) = (%e, %e, %e, %e, %e, %e)\n", j, mat(j, 0), mat(j, 1), mat(j, 2), mat(j, 3), mat(j, 4),
//         mat(j, 5));
}

__global__ void
compute_convolution_kernel(const real *y_scaled, ggxl::VectorField2D<real> *df_by, ggxl::VectorField2D<real> *df_bz,
                           real dz_scaled, int iFace, int my, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  if (j >= my + ngg) {
    return;
  }

  real dy;
  if (j == -ngg) {
    dy = y_scaled[2 * ngg + 1] - y_scaled[2 * ngg];
  } else {
    dy = y_scaled[j + ngg] - y_scaled[j + ngg - 1];
  }
  // y filter coefficients
  // The length scales for 3 directions are the same, which is the initial vorticity thickness
  // Therefore, here we only compute 1 direction, and assign the same value to the other 2 directions
  real sum = 0;
  // the y has been scaled by the initial vorticity thickness; therefore, the dy/delta_omega here is just dy
  real expMulti = -pi * dy/* / DF_IntegralLength*/;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real expValue = exp(expMulti * abs(jf));
    df_by[iFace](j, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real value = df_by[iFace](j, jf, 0) / sum;
    df_by[iFace](j, jf, 0) = value;
    df_by[iFace](j, jf, 1) = value;
    df_by[iFace](j, jf, 2) = value;
  }

  // z filter coefficients
  sum = 0;
  expMulti = -pi * dz_scaled;
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real expValue = exp(expMulti * abs(jf));
    df_bz[iFace](j, jf, 0) = expValue;
    sum += expValue * expValue;
  }
  sum = sqrt(sum);
  for (int jf = -DBoundCond::DF_N; jf <= DBoundCond::DF_N; ++jf) {
    const real value = df_bz[iFace](j, jf, 0) / sum;
    df_bz[iFace](j, jf, 0) = value;
    df_bz[iFace](j, jf, 1) = value;
    df_bz[iFace](j, jf, 2) = value;
  }
}

__global__ void
generate_random_numbers_kernel(ggxl::VectorField2D<curandState> *rng_states, ggxl::VectorField2D<real> *random_numbers,
                               int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg || k >= mz + DBoundCond::DF_N + ngg) {
    return;
  }

  random_numbers[iFace](j, k, 0) = curand_normal_double(&rng_states[iFace](j, k, 0));
  random_numbers[iFace](j, k, 1) = curand_normal_double(&rng_states[iFace](j, k, 1));
  random_numbers[iFace](j, k, 2) = curand_normal_double(&rng_states[iFace](j, k, 2));
}

__global__ void remove_mean_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg) {
    return;
  }

  auto &r = random_numbers[iFace];

  real mean[3]{0, 0, 0}, rms[3]{0, 0, 0};
  for (int k = 0; k < mz; ++k) {
    mean[0] += r(j, k, 0);
    mean[1] += r(j, k, 1);
    mean[2] += r(j, k, 2);

    rms[0] += r(j, k, 0) * r(j, k, 0);
    rms[1] += r(j, k, 1) * r(j, k, 1);
    rms[2] += r(j, k, 2) * r(j, k, 2);
  }
  const real mz_inv = 1.0 / mz;
  mean[0] *= mz_inv;
  mean[1] *= mz_inv;
  mean[2] *= mz_inv;

  rms[0] = sqrt(rms[0] * mz_inv - mean[0] * mean[0]);
  rms[1] = sqrt(rms[1] * mz_inv - mean[1] * mean[1]);
  rms[2] = sqrt(rms[2] * mz_inv - mean[2] * mean[2]);

  for (int k = 0; k < mz; ++k) {
    r(j, k, 0) = (r(j, k, 0) - mean[0]) / rms[0];
    r(j, k, 1) = (r(j, k, 1) - mean[1]) / rms[1];
    r(j, k, 2) = (r(j, k, 2) - mean[2]) / rms[2];
  }
}

__global__ void
apply_periodic_in_spanwise(ggxl::VectorField2D<real> *random_numbers, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (j >= my + DBoundCond::DF_N + ngg || k > DBoundCond::DF_N + ngg) {
    return;
  }

  auto &r = random_numbers[iFace];
  r(j, mz - 1 + k, 0) = r(j, k, 0);
  r(j, -k, 0) = r(j, mz - 1 - k, 0);
  r(j, mz - 1 + k, 1) = r(j, k, 1);
  r(j, -k, 1) = r(j, mz - 1 - k, 1);
  r(j, mz - 1 + k, 2) = r(j, k, 2);
  r(j, -k, 2) = r(j, mz - 1 - k, 2);
}

__global__ void
perform_convolution_y(ggxl::VectorField2D<real> *random_numbers, ggxl::VectorField2D<real> *df_by,
                      ggxl::VectorField2D<real> *df_fy, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg - DBoundCond::DF_N;
  if (j >= my + ngg || k >= mz + DBoundCond::DF_N + ngg) {
    return;
  }

  auto &by = df_by[iFace];
  auto &r = random_numbers[iFace];

  auto &fy = df_fy[iFace];
  auto &fy0 = fy(j, k, 0);
  auto &fy1 = fy(j, k, 1);
  auto &fy2 = fy(j, k, 2);

  // y convolution
  fy0 = 0;
  fy1 = 0;
  fy2 = 0;

  for (int jj = -DBoundCond::DF_N; jj <= DBoundCond::DF_N; ++jj) {
    fy0 += by(j, jj, 0) * r(j + jj, k, 0);
    fy1 += by(j, jj, 1) * r(j + jj, k, 1);
    fy2 += by(j, jj, 2) * r(j + jj, k, 2);
  }
}

__global__ void
perform_convolution_z(ggxl::VectorField2D<real> *df_fy, ggxl::VectorField2D<real> *df_bz,
                      ggxl::VectorField2D<real> *velFluc, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) {
    return;
  }

  auto &fy = df_fy[iFace];
  // z convolution
  auto &bz = df_bz[iFace];

  auto &vf = velFluc[iFace];

  real upp = 0, vpp = 0, wpp = 0;

  for (int kk = -DBoundCond::DF_N; kk <= DBoundCond::DF_N; ++kk) {
    upp += bz(j, kk, 0) * fy(j, k + kk, 0);
    vpp += bz(j, kk, 1) * fy(j, k + kk, 1);
    wpp += bz(j, kk, 2) * fy(j, k + kk, 2);
  }
  vf(j, k, 0) = upp;
  vf(j, k, 1) = vpp;
  vf(j, k, 2) = wpp;
}

__global__ void
compute_fluctuations_first_step(ggxl::VectorField3D<real> *fluctuation_dPtr, ggxl::VectorField1D<real> *lundMatrix_dPtr,
                                ggxl::VectorField2D<real> *df_velFluc_old_dPtr, int iFace, int my, int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) {
    return;
  }

  auto &fluc = fluctuation_dPtr[iFace];
  auto &lund = lundMatrix_dPtr[iFace];
  auto &velGen = df_velFluc_old_dPtr[iFace];

  fluc(0, j, k, 0) = lund(j, 0) * velGen(j, k, 0);
  fluc(0, j, k, 1) = lund(j, 1) * velGen(j, k, 0) + lund(j, 2) * velGen(j, k, 1);
  fluc(0, j, k, 2) = lund(j, 3) * velGen(j, k, 0) + lund(j, 4) * velGen(j, k, 1) + lund(j, 5) * velGen(j, k, 2);
}

__global__ void
Castro_time_correlation_and_fluc_computation(const DParameter *param, DZone *zone, const Inflow *inflow,
                                             ggxl::VectorField2D<real> *velFluc_old,
                                             ggxl::VectorField2D<real> *velFluc_new,
                                             ggxl::VectorField1D<real> *lundMatrix_dPtr,
                                             ggxl::VectorField3D<real> *fluctuation_dPtr, int iFace, int my,
                                             int mz, int ngg) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= my + ngg || k >= mz + ngg) {
    return;
  }

  const real dt = 1.0 / 3.0 * param->dt;

  const real u_upper = inflow->u, u_lower = inflow->u_lower;
  const auto y = zone->y(0, j, k);
  const real y_ref = inflow->delta_omega;
  const real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / y_ref);

  // The integral timescale is treated as a single value here, thus only the first element is used.
  const real tLen_x = y_ref / u;
  //real tLen_y = y_ref / u;
  //real tLen_z = y_ref / u;

  const real PiDtDivTInt = -pi * dt / tLen_x;

  const real arg1 = exp(PiDtDivTInt * 0.5);
  const real arg2 = sqrt(1 - exp(PiDtDivTInt));

  auto &old = velFluc_old[iFace];
  auto &vf = velFluc_new[iFace];

  const real val1 = arg1 * old(j, k, 0) + arg2 * vf(j, k, 0);
  const real val2 = arg1 * old(j, k, 1) + arg2 * vf(j, k, 1);
  const real val3 = arg1 * old(j, k, 2) + arg2 * vf(j, k, 2);
  vf(j, k, 0) = val1;
  vf(j, k, 1) = val2;
  vf(j, k, 2) = val3;
  old(j, k, 0) = val1;
  old(j, k, 1) = val2;
  old(j, k, 2) = val3;

  auto &lund = lundMatrix_dPtr[iFace];
  auto &fluc = fluctuation_dPtr[iFace];
  fluc(0, j, k, 0) = lund(j, 0) * val1;
  fluc(0, j, k, 1) = lund(j, 1) * val1 + lund(j, 2) * val2;
  fluc(0, j, k, 2) = lund(j, 3) * val1 + lund(j, 4) * val2 + lund(j, 5) * val3;
}

void DBoundCond::write_df(Parameter &parameter, const Mesh &mesh) const {
  const int myid = parameter.get_int("myid");
  printf("Process %d is writing the digital filter to the file.\n", myid);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }
  for (int iFace = 0; iFace < n_df_face; ++iFace) {
    std::string filename = "./output/df-p" + std::to_string(myid) + "-f" + std::to_string(iFace) + ".bin";
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
      printf("Error: cannot open the file %s.\n", filename.c_str());
      MpiParallel::exit();
    }

    // First, write the state of the random number generator
    const int blk = df_related_block[iFace];
    int my = mesh[blk].my, mz = mesh[blk].mz, ngg = mesh.ngg;
    const int sz1 = (my + 2 * DF_N + 2 * ngg) * (mz + 2 * DF_N + 2 * ngg) * 3;
    const int sz2 = (my + 2 * ngg) * (mz + 2 * ngg) * 3;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
    cudaMemcpy(df_rng_state_cpu[iFace].data(), rng_states_hPtr[iFace].data(), sz1 * sizeof(curandState),
               cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
    cudaMemcpy(df_velFluc_cpu[iFace].data(), df_velFluc_new_hPtr[iFace].data(),
               sz2 * sizeof(real), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
    fwrite(&my, sizeof(int), 1, fp);
    fwrite(&mz, sizeof(int), 1, fp);
    fwrite(&ngg, sizeof(int), 1, fp);
    fwrite(&DF_N, sizeof(int), 1, fp);

    fwrite(df_rng_state_cpu[iFace].data(), sizeof(curandState), sz1, fp);
    fwrite(df_velFluc_cpu[iFace].data(), sizeof(real), sz2, fp);

    fclose(fp);
  }
}
} // namespace cfd

#include "SpongeLayer.cuh"
#include "DParameter.cuh"
#include "mpi.h"
#include "gxl_lib/MyString.h"
#include <filesystem>

void cfd::initialize_sponge_layer(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                                  const Species &spec) {
  if (const auto init = parameter.get_int("initial");init == 1) {
    // Computation with previous results
    // We first find if there exists previous mean conservation variables' information
    // If so, we will use it to initialize the sponge layer
    // If not, we initialize the sponge layer as the same method when starting a new computation.
    // TODO: Implement this function after we have implemented the output part.
    // Find if the file "output/sponge_layer_mean_cv.bin" exists
    const std::filesystem::path file_path("output/sponge_layer_mean_cv.bin");
    if (exists(file_path)) {
      // Read the file
      read_sponge_layer(parameter, mesh, field, spec);
      return;
    }
  }
  // Start a new computation
  parameter.update_parameter("sponge_iter", 0);
  parameter.update_parameter("sponge_scalar_iter", std::vector<int>(parameter.get_int("n_scalar"), 0));
}

__global__ void cfd::update_sponge_layer_value(DZone *zone, const DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  // const int n_iter = param->sponge_iter;
  // const auto n_scalar_iter = param->sponge_scalar_iter;
  //
  // auto &sponge = zone->sponge_mean_cv;
  // auto &cv = zone->cv;
  //
  // sponge(i, j, k, 0) = (sponge(i, j, k, 0) * n_iter + cv(i, j, k, 0)) / (n_iter + 1);
  // sponge(i, j, k, 1) = (sponge(i, j, k, 1) * n_iter + cv(i, j, k, 1)) / (n_iter + 1);
  // sponge(i, j, k, 2) = (sponge(i, j, k, 2) * n_iter + cv(i, j, k, 2)) / (n_iter + 1);
  // sponge(i, j, k, 3) = (sponge(i, j, k, 3) * n_iter + cv(i, j, k, 3)) / (n_iter + 1);
  // sponge(i, j, k, 4) = (sponge(i, j, k, 4) * n_iter + cv(i, j, k, 4)) / (n_iter + 1);
  // for (int l = 0; l < param->n_scalar; l++) {
  //   sponge(i, j, k, 5 + l) = (sponge(i, j, k, 5 + l) * n_scalar_iter[l] + cv(i, j, k, 5 + l)) / (n_scalar_iter[l] + 1);
  // }
}

void cfd::update_sponge_iter(DParameter *param, Parameter &parameter) {
  parameter.update_parameter("sponge_iter", parameter.get_int("sponge_iter") + 1);
  auto para = parameter.get_int_array("sponge_scalar_iter");
  for (auto &l: para) {
    ++l;
  }
  parameter.update_parameter("sponge_scalar_iter", para);
  update_sponge_iter_dev<<<1, 1>>>(param);
}

__global__ void cfd::update_sponge_iter_dev(DParameter *param) {
  // ++param->sponge_iter;
  // for (int i = 0; i < param->n_scalar; i++) {
  //   ++param->sponge_scalar_iter[i];
  // }
}

__device__ void cfd::sponge_layer_source(DZone *zone, int i, int j, int k, const DParameter *param) {
//   real sigma = 0;
//   if (param->spongeX == 1 || param->spongeX == 3) {
//     // X inlet sponge layer
//     const real x1 = param->spongeXMinusStart;
//     const real x = zone->x(i, j, k);
//     if (x < x1) {
//       const real x2 = param->spongeXMinusEnd;
//       real xi = (x - x1) / (x2 - x1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma0 * S;
//     }
//   }
//   if (param->spongeX == 2 || param->spongeX == 3) {
//     // X outflow sponge layer
//     const real x1 = param->spongeXPlusStart;
//     const real x = zone->x(i, j, k);
//     if (x > x1) {
//       const real x2 = param->spongeXPlusEnd;
//       real xi = (x - x1) / (x2 - x1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma1 * S;
//     }
//   }
//   if (param->spongeY == 1 || param->spongeY == 3) {
//     // Y- sponge layer
//     const real y1 = param->spongeYMinusStart;
//     const real y = zone->y(i, j, k);
//     if (y < y1) {
//       const real y2 = param->spongeYMinusEnd;
//       real xi = (y - y1) / (y2 - y1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma2 * S;
//     }
//   }
//   if (param->spongeY == 2 || param->spongeY == 3) {
//     // Y+ sponge layer
//     const real y1 = param->spongeYPlusStart;
//     const real y = zone->y(i, j, k);
//     if (y > y1) {
//       const real y2 = param->spongeYPlusEnd;
//       real xi = (y - y1) / (y2 - y1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma3 * S;
//     }
//   }
//   if (param->spongeZ == 1 || param->spongeZ == 3) {
//     // Z- sponge layer
//     const real z1 = param->spongeZMinusStart;
//     const real z = zone->z(i, j, k);
//     if (z < z1) {
//       const real z2 = param->spongeZMinusEnd;
//       real xi = (z - z1) / (z2 - z1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma4 * S;
//     }
//   }
//   if (param->spongeZ == 2 || param->spongeZ == 3) {
//     // Z+ sponge layer
//     const real z1 = param->spongeZPlusStart;
//     const real z = zone->z(i, j, k);
//     if (z > z1) {
//       const real z2 = param->spongeZPlusEnd;
//       real xi = (z - z1) / (z2 - z1);
//       if (xi > 1)
//         xi = 1;
//       const real S = sponge_function(xi, param->sponge_function);
//       sigma += param->sponge_sigma5 * S;
//     }
//   }
// //  if (abs(sigma) > 1e-5)
// //    printf("sigma(%d,%d)=%e\n", i, j, sigma);
//
//   for (int l = 0; l < param->n_var; ++l) {
//     if (isnan(zone->cv(i, j, k, l))) {
//       printf("dq(%d,%d,%d)=%e, sigma=%e, cv=%e, sponge=%e\n", i, j, l, zone->dq(i, j, k, l), sigma,
//              zone->cv(i, j, k, l), zone->sponge_mean_cv(i, j, k, l));
//     }
//     zone->dq(i, j, k, l) -= sigma * (zone->cv(i, j, k, l) - zone->sponge_mean_cv(i, j, k, l));
// //    if (abs(zone->dq(i,j,k,l))>1e-10)
// //      printf("dq(%d,%d,%d)=%e, sigma=%e, cv=%e, sponge=%e\n", i, j, l, zone->dq(i, j, k, l), sigma,
// //             zone->cv(i, j, k, l), zone->sponge_mean_cv(i, j, k, l));
//   }
}

__device__ real cfd::sponge_function(real xi, int method) {
  real S{0};
  if (method == 0) {
    // (Nektar++, CPC, 2024)
    const real xi3 = xi * xi * xi;
    S = 6 * xi3 * xi * xi - 15 * xi3 * xi + 10 * xi3;
  }
  return S;
}

void cfd::output_sponge_layer(const Parameter &parameter, const std::vector<Field> &field, const Mesh &mesh,
                              const Species &spec) {
  for (const auto &b: field) {
    cudaMemcpy(b.sponge_mean_cv.data(), b.h_ptr->sponge_mean_cv.data(),
               b.h_ptr->sponge_mean_cv.size() * sizeof(real), cudaMemcpyDeviceToHost);
  }

  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, "output/sponge_layer_mean_cv.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &fp);

  const int myid{parameter.get_int("myid")};
  MPI_Offset offset{0};
  if (myid == 0) {
    const int n_iter = parameter.get_int("sponge_iter");
    MPI_File_write_at(fp, offset, &n_iter, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset += 4;
    const int n_scalar = parameter.get_int("n_scalar");
    MPI_File_write_at(fp, offset, &n_scalar, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset += 4;
    auto &list = spec.spec_name;
    for (auto &name: list) {
      gxl::write_str(name.c_str(), fp, offset);
    }
    auto &spec_iter = parameter.get_int_array("sponge_scalar_iter");
    for (auto &iter: spec_iter) {
      MPI_File_write_at(fp, offset, &iter, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
      offset += 4;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int blk = 0;
  int64_t n_grid_before{0};
  for (int p = 0; p < myid; ++p) {
    if (p > 0)
      blk += mesh.nblk[p - 1];
    for (int b = 0; b < mesh.nblk[p]; ++b) {
      n_grid_before += mesh.mx_blk[blk + b] * mesh.my_blk[blk + b] * mesh.mz_blk[blk + b];
    }
  }
  const int n_var = parameter.get_int("n_var");
  offset += static_cast<MPI_Offset>(n_grid_before * n_var * sizeof(real));
  for (auto &f: field) {
    MPI_File_write_at(fp, offset, f.sponge_mean_cv.data(), f.sponge_mean_cv.size() * n_var, MPI_REAL,
                      MPI_STATUS_IGNORE);
    offset += static_cast<MPI_Offset>(f.sponge_mean_cv.size() * n_var * sizeof(real));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

void cfd::read_sponge_layer(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                            const Species &spec) {
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, "output/sponge_layer_mean_cv.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

  MPI_Offset offset{0};
  int n_iter;
  MPI_File_read_at(fp, offset, &n_iter, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
  offset += 4;
  parameter.update_parameter("sponge_iter", n_iter);
  int n_scalar;
  MPI_File_read_at(fp, offset, &n_scalar, 1, MPI_INT32_T, MPI_STATUS_IGNORE);
  offset += 4;
  std::vector<std::string> spec_name;
  for (int l = 0; l < n_scalar; ++l) {
    std::string nn = gxl::read_str_from_binary_MPI_ver(fp, offset);
    spec_name.push_back(nn);
  }
  std::vector<int> sponge_last_iter(n_scalar, 0);
  for (int l = 0; l < n_scalar; ++l) {
    MPI_File_read_at(fp, offset, &sponge_last_iter[l], 1, MPI_INT32_T, MPI_STATUS_IGNORE);
    offset += 4;
  }
  std::vector<int> sponge_scalar_iter(parameter.get_int("n_scalar"), 0);
  auto &list = spec.spec_list;
  for (int l = 0; l < n_scalar; ++l) {
    const auto& nn = spec_name[l];
    if (list.find(nn) != list.end()) {
      sponge_scalar_iter[list.at(nn)] = sponge_last_iter[l];
    }
  }
  parameter.update_parameter("sponge_scalar_iter", sponge_scalar_iter);

  const int myid{parameter.get_int("myid")};
  int blk = 0;
  int64_t n_grid_before{0};
  for (int p = 0; p < myid; ++p) {
    if (p > 0)
      blk += mesh.nblk[p - 1];
    for (int b = 0; b < mesh.nblk[p]; ++b) {
      n_grid_before += mesh.mx_blk[blk + b] * mesh.my_blk[blk + b] * mesh.mz_blk[blk + b];
    }
  }
  const int n_var = parameter.get_int("n_var");
  offset += static_cast<MPI_Offset>(n_grid_before * n_var * sizeof(real));
  for (auto &f: field) {
    MPI_File_read_at(fp, offset, f.sponge_mean_cv.data(), f.sponge_mean_cv.size() * n_var, MPI_REAL,
                     MPI_STATUS_IGNORE);
    offset += static_cast<MPI_Offset>(f.sponge_mean_cv.size() * n_var * sizeof(real));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);

  for (auto &f: field) {
    cudaMemcpy(f.h_ptr->sponge_mean_cv.data(), f.sponge_mean_cv.data(), f.sponge_mean_cv.size() * sizeof(real),
               cudaMemcpyHostToDevice);
  }
}

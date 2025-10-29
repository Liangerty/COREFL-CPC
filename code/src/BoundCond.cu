#include "BoundCond.cuh"
#include "Parallel.h"
#include <fstream>
#include "gxl_lib/MyString.h"
#include "gxl_lib/MyAlgorithm.h"
#include "MixingLayer.cuh"
// This file should not include "algorithm", which will result in an error for gcc.
// This persuades me to write a new function "exists" in MyAlgorithm.h and MyAlgorithm.cpp.

namespace cfd {
template<typename BCType>
void register_bc(BCType *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
  Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(BCType));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      BCType bound_cond(bc_name, parameter);
      cudaMemcpy(&bc[i], &bound_cond, sizeof(BCType), cudaMemcpyHostToDevice);
      break;
    }
  }
}

template<>
void register_bc<Wall>(Wall *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
  Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Wall));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      Wall wall(this_bc, parameter);
      bc_info[i].label = bc_label;
      cudaMemcpy(&bc[i], &wall, sizeof(Wall), cudaMemcpyHostToDevice);
    }
  }
}

template<>
void register_bc<Inflow>(Inflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
  Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Inflow));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      Inflow inflow(bc_name, species, parameter);
      inflow.copy_to_gpu(&bc[i], parameter);
      break;
    }
  }
}

template<>
void
register_bc<FarField>(FarField *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
  Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(FarField));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      FarField farfield(bc_name, species, parameter);
      farfield.copy_to_gpu(&bc[i], parameter);
      break;
    }
  }
}

template<>
void register_bc<SubsonicInflow>(SubsonicInflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
  Species &species, Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(SubsonicInflow));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      SubsonicInflow subsonic_inflow(bc_name, parameter);
      subsonic_inflow.copy_to_gpu(&bc[i], parameter);
      break;
    }
  }
}

template<>
void
register_bc<BackPressure>(BackPressure *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
  Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(BackPressure));
  bc_info = new BCInfo[n_bc];
  for (int i = 0; i < n_bc; ++i) {
    const int index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      const int bc_label = std::get<int>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      BackPressure back_pressure(bc_name, parameter);
      cudaMemcpy(&bc[i], &back_pressure, sizeof(BackPressure), cudaMemcpyHostToDevice);
      break;
    }
  }
}

__global__ void adjust_inviscid_range(DZone *zone, const int *nr_labels, int n_nr) {
  int n_boundary = zone->n_boundary;
  for (int ib = 0; ib < n_boundary; ++ib) {
    const auto &b = zone->boundary[ib];
    const int l = b.type_label;
    for (int i = 0; i < n_nr; ++i) {
      if (l == nr_labels[i]) {
        // This boundary is a non-reflecting boundary.
        if (b.face == 0) {
          if (b.direction == 1) {
            zone->iMax = zone->mx - 2;
          } else {
            zone->iMin = 1;
          }
        } else if (b.face == 1) {
          if (b.direction == 1) {
            zone->jMax = zone->my - 2;
          } else {
            zone->jMin = 1;
          }
        } else if (b.face == 2) {
          if (b.direction == 1) {
            zone->kMax = zone->mz - 2;
          } else {
            zone->kMin = 1;
          }
        }
      }
    }
  }
}

void DBoundCond::initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter) {
  std::vector<int> bc_labels;
  // Count the number of distinct boundary conditions
  for (auto i = 0; i < mesh.n_block; i++) {
    for (auto &b: mesh[i].boundary) {
      auto lab = b.type_label;
      bool has_this_bc = false;
      for (auto l: bc_labels) {
        if (l == lab) {
          has_this_bc = true;
          break;
        }
      }
      if (!has_this_bc) {
        bc_labels.push_back(lab);
      }
    }
  }
  // Initialize the inflow and wall conditions which are different among cases.
  std::vector<int> wall_idx, symmetry_idx, inflow_idx, outflow_idx, farfield_idx, subsonic_inflow_idx, back_pressure_idx
    , periodic_idx;
  auto &bcs = parameter.get_string_array("boundary_conditions");
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto label = std::get<int>(bc.at("label"));

    auto this_iter = bc_labels.end();
    for (auto iter = bc_labels.begin(); iter != bc_labels.end(); ++iter) {
      if (*iter == label) {
        this_iter = iter;
        break;
      }
    }
    if (this_iter != bc_labels.end()) {
      bc_labels.erase(this_iter);
      auto type = std::get<std::string>(bc.at("type"));
      if (type == "wall") {
        wall_idx.push_back(label);
        ++n_wall;
      } else if (type == "inflow") {
        inflow_idx.push_back(label);
        ++n_inflow;
      }
      // Note: Normally, this would not happen for outflow, symmetry, and periodic conditions.
      // Because the above-mentioned conditions normally do not need to specify special treatments.
      // If we need to add support for these conditions, then we add them here.
      else if (type == "outflow") {
        outflow_idx.push_back(label);
        ++n_outflow;
      } else if (type == "symmetry") {
        symmetry_idx.push_back(label);
        ++n_symmetry;
      } else if (type == "farfield") {
        farfield_idx.push_back(label);
        ++n_farfield;
      } else if (type == "subsonic_inflow") {
        subsonic_inflow_idx.push_back(label);
        ++n_subsonic_inflow;
      } else if (type == "back_pressure") {
        back_pressure_idx.push_back(label);
        ++n_back_pressure;
      } else if (type == "periodic") {
        periodic_idx.push_back(label);
        ++n_periodic;
      }
    }
  }
  for (int lab: bc_labels) {
    if (lab == 2) {
      wall_idx.push_back(lab);
      ++n_wall;
    } else if (lab == 3) {
      symmetry_idx.push_back(lab);
      ++n_symmetry;
    } else if (lab == 4) {
      farfield_idx.push_back(lab);
      ++n_farfield;
    } else if (lab == 5) {
      inflow_idx.push_back(lab);
      ++n_inflow;
    } else if (lab == 6) {
      outflow_idx.push_back(lab);
      ++n_outflow;
    } else if (lab == 7) {
      subsonic_inflow_idx.push_back(lab);
      ++n_subsonic_inflow;
    } else if (lab == 8) {
      periodic_idx.push_back(lab);
      ++n_periodic;
    } else if (lab == 9) {
      back_pressure_idx.push_back(lab);
      ++n_back_pressure;
    }
  }

  // Read specific conditions
  // We always first initialize the Farfield and Inflow conditions, because they may set the reference values.
  register_bc<FarField>(farfield, n_farfield, farfield_idx, farfield_info, species, parameter);
  register_bc<Inflow>(inflow, n_inflow, inflow_idx, inflow_info, species, parameter);
  register_bc<SubsonicInflow>(subsonic_inflow, n_subsonic_inflow, subsonic_inflow_idx, subsonic_inflow_info, species,
                              parameter);
  register_bc<Wall>(wall, n_wall, wall_idx, wall_info, species, parameter);
  register_bc<Symmetry>(symmetry, n_symmetry, symmetry_idx, symmetry_info, species, parameter);
  register_bc<Outflow>(outflow, n_outflow, outflow_idx, outflow_info, species, parameter);
  register_bc<BackPressure>(back_pressure, n_back_pressure, back_pressure_idx, back_pressure_info, species, parameter);
  register_bc<Periodic>(periodic, n_periodic, periodic_idx, periodic_info, species, parameter);

  link_bc_to_boundaries(mesh, field);

  MpiParallel::barrier();

  initialize_profile_and_rng(parameter, mesh, species, field);

  df_label.resize(n_inflow, -1);
  initialize_digital_filter(parameter, mesh);

  // non-reflecting boundary condition
  if (parameter.has_int_array("nonReflectingBCs"))
    nonReflectingBCs = parameter.get_int_array("nonReflectingBCs");
  const int n_nonReflectingBCs = static_cast<int>(nonReflectingBCs.size());
  int *nr_labels_gpu = nullptr;
  cudaMalloc(&nr_labels_gpu, n_nonReflectingBCs * sizeof(int));
  cudaMemcpy(nr_labels_gpu, nonReflectingBCs.data(), n_nonReflectingBCs * sizeof(int), cudaMemcpyHostToDevice);
  for (const auto &f: field) {
    adjust_inviscid_range<<<1,1>>>(f.d_ptr, nr_labels_gpu, n_nonReflectingBCs);
  }

  MpiParallel::barrier();
  if (parameter.get_int("myid") == 0)
    printf("\tBoundary conditions are set.\n");
}

void DBoundCond::link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const {
  const int n_block{mesh.n_block};
  auto **i_wall = new int *[n_wall];
  for (size_t i = 0; i < n_wall; i++) {
    i_wall[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_wall[i][j] = 0;
    }
  }
  auto **i_symm = new int *[n_symmetry];
  for (size_t i = 0; i < n_symmetry; i++) {
    i_symm[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_symm[i][j] = 0;
    }
  }
  auto **i_farfield = new int *[n_farfield];
  for (size_t i = 0; i < n_farfield; ++i) {
    i_farfield[i] = new int[n_block];
    for (int j = 0; j < n_block; ++j) {
      i_farfield[i][j] = 0;
    }
  }
  auto **i_inflow = new int *[n_inflow];
  for (size_t i = 0; i < n_inflow; i++) {
    i_inflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_inflow[i][j] = 0;
    }
  }
  auto **i_outflow = new int *[n_outflow];
  for (size_t i = 0; i < n_outflow; i++) {
    i_outflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_outflow[i][j] = 0;
    }
  }
  auto **i_subsonic_inflow = new int *[n_subsonic_inflow];
  for (size_t i = 0; i < n_subsonic_inflow; i++) {
    i_subsonic_inflow[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_subsonic_inflow[i][j] = 0;
    }
  }
  auto **i_back_pressure = new int *[n_back_pressure];
  for (size_t i = 0; i < n_back_pressure; i++) {
    i_back_pressure[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_back_pressure[i][j] = 0;
    }
  }
  auto **i_periodic = new int *[n_periodic];
  for (size_t i = 0; i < n_periodic; i++) {
    i_periodic[i] = new int[n_block];
    for (int j = 0; j < n_block; j++) {
      i_periodic[i][j] = 0;
    }
  }

  // We first count how many faces corresponds to a given boundary condition
  for (int i = 0; i < n_block; i++) {
    count_boundary_of_type_bc(mesh[i].boundary, n_wall, i_wall, i, n_block, wall_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_symmetry, i_symm, i, n_block, symmetry_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_farfield, i_farfield, i, n_block, farfield_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_inflow, i_inflow, i, n_block, inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_outflow, i_outflow, i, n_block, outflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_subsonic_inflow, i_subsonic_inflow, i, n_block,
                              subsonic_inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_back_pressure, i_back_pressure, i, n_block, back_pressure_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_periodic, i_periodic, i, n_block, periodic_info);
  }
  for (size_t l = 0; l < n_wall; l++) {
    wall_info[l].boundary = new int2[wall_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_symmetry; ++l) {
    symmetry_info[l].boundary = new int2[symmetry_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_farfield; ++l) {
    farfield_info[l].boundary = new int2[farfield_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_inflow; l++) {
    inflow_info[l].boundary = new int2[inflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_outflow; l++) {
    outflow_info[l].boundary = new int2[outflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_subsonic_inflow; ++l) {
    subsonic_inflow_info[l].boundary = new int2[subsonic_inflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_back_pressure; ++l) {
    back_pressure_info[l].boundary = new int2[back_pressure_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_periodic; ++l) {
    periodic_info[l].boundary = new int2[periodic_info[l].n_boundary];
  }

  const auto ngg{mesh[0].ngg};
  for (auto i = 0; i < n_block; i++) {
    link_boundary_and_condition(mesh[i].boundary, wall_info, n_wall, i_wall, i);
    link_boundary_and_condition(mesh[i].boundary, symmetry_info, n_symmetry, i_symm, i);
    link_boundary_and_condition(mesh[i].boundary, farfield_info, n_farfield, i_farfield, i);
    link_boundary_and_condition(mesh[i].boundary, inflow_info, n_inflow, i_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, outflow_info, n_outflow, i_outflow, i);
    link_boundary_and_condition(mesh[i].boundary, subsonic_inflow_info, n_subsonic_inflow, i_subsonic_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, back_pressure_info, n_back_pressure, i_back_pressure, i);
    link_boundary_and_condition(mesh[i].boundary, periodic_info, n_periodic, i_periodic, i);
  }
  //  for (auto i = 0; i < n_block; i++) {
  //    for (size_t l = 0; l < n_wall; l++) {
  //      const auto nb = wall_info[l].n_boundary;
  //      for (size_t m = 0; m < nb; m++) {
  //        auto i_zone = wall_info[l].boundary[m].x;
  //        if (i_zone != i) {
  //          continue;
  //        }
  //        auto &b = mesh[i].boundary[wall_info[l].boundary[m].y];
  //        for (int q = 0; q < 3; ++q) {
  //          if (q == b.face) continue;
  //          b.range_start[q] += ngg;
  //          b.range_end[q] -= ngg;
  //        }
  //      }
  //    }
  //    cudaMemcpy(field[i].h_ptr->boundary, mesh[i].boundary.data(), mesh[i].boundary.size() * sizeof(Boundary),
  //               cudaMemcpyHostToDevice);
  //  }
  for (int i = 0; i < n_wall; i++) {
    delete[]i_wall[i];
  }
  for (int i = 0; i < n_symmetry; i++) {
    delete[]i_symm[i];
  }
  for (int i = 0; i < n_farfield; ++i) {
    delete[]i_farfield[i];
  }
  for (int i = 0; i < n_inflow; i++) {
    delete[]i_inflow[i];
  }
  for (int i = 0; i < n_outflow; i++) {
    delete[]i_outflow[i];
  }
  for (int i = 0; i < n_subsonic_inflow; ++i) {
    delete[]i_subsonic_inflow[i];
  }
  for (int i = 0; i < n_back_pressure; ++i) {
    delete[]i_back_pressure[i];
  }
  for (int i = 0; i < n_periodic; ++i) {
    delete[]i_periodic[i];
  }
  delete[]i_wall;
  delete[]i_symm;
  delete[]i_farfield;
  delete[]i_inflow;
  delete[]i_outflow;
  delete[]i_subsonic_inflow;
  delete[]i_back_pressure;
  delete[]i_periodic;
}

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, int n_bc, int **sep, int blk_idx, int n_block,
  BCInfo *bc_info) {
  if (n_bc <= 0) {
    return;
  }

  // Count how many faces correspond to the given bc
  const auto n_boundary{boundary.size()};
  auto *n = new int[n_bc];
  memset(n, 0, sizeof(int) * n_bc);
  for (size_t l = 0; l < n_bc; l++) {
    const int label = bc_info[l].label; // This means every bc should have a member "label"
    for (size_t i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        ++bc_info[l].n_boundary;
        ++n[l];
      }
    }
  }
  if (blk_idx < n_block - 1) {
    for (size_t l = 0; l < n_bc; l++) {
      sep[l][blk_idx + 1] = n[l] + sep[l][blk_idx];
    }
  }
  delete[]n;
}

void link_boundary_and_condition(const std::vector<Boundary> &boundary, const BCInfo *bc, int n_bc, int **sep,
  int i_zone) {
  const auto n_boundary{boundary.size()};
  for (size_t l = 0; l < n_bc; l++) {
    const int label = bc[l].label;
    int has_read{sep[l][i_zone]};
    for (auto i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        bc[l].boundary[has_read] = make_int2(i_zone, i);
        ++has_read;
      }
    }
  }
}

void Inflow::copy_to_gpu(Inflow *d_inflow, const Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  const auto h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);
  if (inflow_type == 2) {
    // For mixing layer flow, there are another group of sv.
    const auto h_sv_lower = new real[n_scalar];
    for (int l = 0; l < n_scalar; ++l) {
      h_sv_lower[l] = sv_lower[l];
    }
    delete[]sv_lower;
    cudaMalloc(&sv_lower, n_scalar * sizeof(real));
    cudaMemcpy(sv_lower, h_sv_lower, n_scalar * sizeof(real), cudaMemcpyHostToDevice);
  }
  if (fluctuation_type == 3) {
    // For the case of fluctuation_type == 3, we need to copy the fluctuation field to GPU.
    const auto h_rand_values = new real[199];
    for (int l = 0; l < 199; ++l) {
      h_rand_values[l] = random_phase[l];
    }
    cudaMalloc(&random_phase, sizeof(real) * 199);
    cudaMemcpy(random_phase, h_rand_values, sizeof(real) * 199, cudaMemcpyHostToDevice);
  }

  cudaMemcpy(d_inflow, this, sizeof(Inflow), cudaMemcpyHostToDevice);
}

void FarField::copy_to_gpu(FarField *d_farfield, const Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  const auto h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);

  cudaMemcpy(d_farfield, this, sizeof(FarField), cudaMemcpyHostToDevice);
}

void SubsonicInflow::copy_to_gpu(SubsonicInflow *d_inflow, const Parameter &parameter) {
  const int n_scalar{parameter.get_int("n_scalar")};
  const auto h_sv = new real[n_scalar];
  for (int l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);

  cudaMemcpy(d_inflow, this, sizeof(SubsonicInflow), cudaMemcpyHostToDevice);
}

void initialize_profile_with_inflow(const Boundary &boundary, const Block &block, const Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile,
  const std::string &profile_related_bc_name) {
  const int direction = boundary.face;
  const int extent[3]{block.mx, block.my, block.mz};
  const int ngg = block.ngg;
  int range_0[2]{-ngg, block.mx + ngg - 1}, range_1[2]{-ngg, block.my + ngg - 1},
      range_2[2]{-ngg, block.mz + ngg - 1};
  int n0 = extent[0], n1 = extent[1], n2 = extent[2];
  //  if (direction == 1) {
  //    n1 = extent[0];
  //    range_1[1] = block.mx + ngg - 1;
  //  } else if (direction == 2) {
  //    n1 = extent[0];
  //    n2 = extent[1];
  //    range_1[1] = block.mx + ngg - 1;
  //    range_2[1] = block.my + ngg - 1;
  //  }
  if (direction == 0) {
    n0 = 1 + ngg;
    range_0[0] = 0;
    range_0[1] = ngg;
  } else if (direction == 1) {
    n1 = 1 + ngg;
    range_1[0] = 0;
    range_1[1] = ngg;
  } else if (direction == 2) {
    n2 = 1 + ngg;
    range_2[0] = 0;
    range_2[1] = ngg;
  }

  ggxl::VectorField3DHost<real> profile_host;
  const int n_var = parameter.get_int("n_var");
  profile_host.resize(n0, n1, n2, n_var + 1, ngg);
  const Inflow inflow1(profile_related_bc_name, species, parameter);
  for (int i0 = range_0[0]; i0 <= range_0[1]; ++i0) {
    for (int i1 = range_1[0]; i1 <= range_1[1]; ++i1) {
      for (int i2 = range_2[0]; i2 <= range_2[1]; ++i2) {
        profile_host(i0, i1, i2, 0) = inflow1.density;
        profile_host(i0, i1, i2, 1) = inflow1.u;
        profile_host(i0, i1, i2, 2) = inflow1.v;
        profile_host(i0, i1, i2, 3) = inflow1.w;
        profile_host(i0, i1, i2, 4) = inflow1.pressure;
        profile_host(i0, i1, i2, 5) = inflow1.temperature;
        for (int i = 0; i < species.n_spec; ++i) {
          profile_host(i0, i1, i2, 6 + i) = inflow1.sv[i];
        }
      }
    }
  }
  profile.allocate_memory(n0, n1, n2, n_var + 1, ngg);
  cudaMemcpy(profile.data(), profile_host.data(), sizeof(real) * profile_host.size() * (n_var + 1),
             cudaMemcpyHostToDevice);
}


void init_mixingLayer_prof_compatible_cpu(const Boundary &boundary, Parameter &parameter, const Block &b,
  const Species &species, ggxl::VectorField3D<real> &profile) {
  const int direction = boundary.face;
  const int ngg = b.ngg;
  const int mx = b.mx, my = b.my, mz = b.mz;
  int range_x[2]{-ngg, mx + ngg - 1}, range_y[2]{-ngg, my + ngg - 1},
      range_z[2]{-ngg, mz + ngg - 1};
  int n0 = mx + 2 * ngg, n1 = my + 2 * ngg, n2 = mz + 2 * ngg;
  if (direction == 0) {
    n0 = 1 + ngg;
    range_x[0] = -ngg;
    range_x[1] = 0;
  } else if (direction == 1) {
    n1 = 1 + ngg;
    range_y[0] = -ngg;
    range_y[1] = 0;
  } else if (direction == 2) {
    n2 = 1 + ngg;
    range_z[0] = -ngg;
    range_z[1] = 0;
  }

  std::vector<real> var_info;
  get_mixing_layer_info(parameter, species, var_info);

  ggxl::VectorField3DHost<real> profile_host;
  int extent[3]{mx, my, mz};
  extent[direction] = 1;
  int nv = parameter.get_int("n_var");
  profile_host.resize(extent[0], extent[1], extent[2], nv + 1, ngg);

  const int n_spec = species.n_spec;
  const real delta_omega = parameter.get_real("delta_omega");
  int n_fl{0};
  if ((species.n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2)
    n_fl = 2;
  const int i_fl = parameter.get_int("i_fl");
  const int n_ps = parameter.get_int("n_ps");
  const int i_ps = parameter.get_int("i_ps");
  const int n_turb = parameter.get_int("n_turb");
  const int n_scalar = parameter.get_int("n_scalar");

  for (int j = range_y[0]; j <= range_y[1]; ++j) {
    real y = b.y(0, j, 0);
    const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
    real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / delta_omega);
    real p = var_info[4];
    real density, t;
    const real t_upper = var_info[5], t_lower = var_info[12 + n_spec];
    real yk[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER + 4] = {};

    real y_upper = (u - u_lower) / (u_upper - u_lower);
    real y_lower = 1 - y_upper;

    if (n_spec > 0) {
      // multi-species
      auto sv_upper = &var_info[6];
      auto sv_lower = &var_info[7 + n_spec + 6];

      for (int l = 0; l < n_spec; ++l) {
        yk[l] = y_upper * sv_upper[l] + y_lower * sv_lower[l];
      }

      // compute the total enthalpy of upper and lower streams
      real h0_upper{0.5 * u_upper * u_upper}, h0_lower{0.5 * u_lower * u_lower};

      real h_upper[MAX_SPEC_NUMBER], h_lower[MAX_SPEC_NUMBER];
      species.compute_enthalpy(t_upper, h_upper);
      species.compute_enthalpy(t_lower, h_lower);

      real mw_inv = 0;
      for (int l = 0; l < n_spec; ++l) {
        h0_upper += yk[l] * h_upper[l];
        h0_lower += yk[l] * h_lower[l];
        mw_inv += yk[l] / species.mw[l];
      }

      real h = y_upper * h0_upper + y_lower * h0_lower;
      h -= 0.5 * u * u;

      auto hs = h_upper, cps = h_lower;
      real err{1};
      t = t_upper * y_upper + t_lower * y_lower;
      constexpr int max_iter{1000};
      constexpr real eps{1e-3};
      int iter = 0;

      while (err > eps && iter++ < max_iter) {
        species.compute_enthalpy_and_cp(t, hs, cps);
        real cp{0}, h_new{0};
        for (int l = 0; l < n_spec; ++l) {
          cp += yk[l] * cps[l];
          h_new += yk[l] * hs[l];
        }
        const real t1 = t - (h_new - h) / cp;
        err = abs(1 - t1 / t);
        t = t1;
      }
      density = p / (R_u * mw_inv * t);

      if (n_fl > 0) {
        yk[i_fl] = var_info[6 + n_spec] * y_upper + var_info[13 + n_spec + n_spec] * y_lower;
        yk[i_fl + 1] = 0;
      }
      if (n_ps > 0) {
        for (int l = 0; l < n_ps; ++l) {
          yk[i_ps + l] =
              var_info[14 + 2 * n_spec + 4 + 2 * l] * y_upper + var_info[14 + 2 * n_spec + 4 + 2 * l + 1] * y_lower;
        }
      }
    } else {
      // Air
      constexpr real cp = gamma_air * R_air / (gamma_air - 1);
      real h0_upper = 0.5 * u_upper * u_upper + cp * var_info[5];
      real h0_lower = 0.5 * u_lower * u_lower + cp * var_info[12 + n_spec];
      real h = y_upper * h0_upper + y_lower * h0_lower - 0.5 * u * u;
      t = h / cp;
      density = p / (R_air * t);

      if (n_ps > 0) {
        if (y > 0) {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l];
          }
        } else {
          for (int l = 0; l < n_ps; ++l) {
            yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l + 1];
          }
        }
      }
    }
    if (n_turb > 0) {
      if (y > 0) {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + 1 + l];
        }
      } else {
        for (int l = 0; l < n_turb; ++l) {
          yk[l + n_spec] = var_info[13 + 2 * n_spec + n_turb + 1 + l];
        }
      }
    }

    auto &prof = profile_host;
    for (int k = range_z[0]; k <= range_z[1]; ++k) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        prof(i, j, k, 0) = density;
        prof(i, j, k, 1) = u;
        prof(i, j, k, 2) = 0;
        prof(i, j, k, 3) = 0;
        prof(i, j, k, 4) = p;
        prof(i, j, k, 5) = t;
        for (int l = 0; l < n_scalar; ++l) {
          prof(i, j, k, 6 + l) = yk[l];
        }
      }
    }
  }

  profile.allocate_memory(extent[0], extent[1], extent[2], nv + 1, ngg);
  cudaMemcpy(profile.data(), profile_host.data(), profile_host.size() * (nv + 1) * sizeof(real),
             cudaMemcpyHostToDevice);

  // Write to file
  FILE *fp = fopen("./mixingLayerProfile.dat", "w");
  if (fp == nullptr) {
    printf("Cannot open file %s\n", "./mixingLayerProfile.dat");
    MpiParallel::exit();
  }
  fprintf(fp, "VARIABLES = \"X\"\n\"Y\"\n\"Z\"\n\"RHO\"\n\"U\"\n\"V\"\n\"W\"\n\"p\"\n\"T\"\n");
  if (n_spec > 0) {
    for (int l = 0; l < n_spec; ++l) {
      fprintf(fp, "\"%s\"\n", species.spec_name[l].c_str());
    }
  }
  if (n_turb == 2) {
    fprintf(fp, "\"TKE\"\n\"omega\"\n");
  }
  if (n_fl > 0) {
    fprintf(fp, "\"MixtureFraction\"\n\"MixtureFractionVariance\"\n");
  }
  fprintf(fp, "ZONE T=\"INFLOW\"\n");
  fprintf(fp, "I=%d, J=%d, K=%d, f=BLOCK\n", n0, n1, n2);
  // Print nv DOUBLE in DT=(...) into a char array
  std::string temp = "DT=(";
  for (int l = 0; l < nv + 4; ++l) {
    temp += "DOUBLE ";
  }
  temp += ")\n";
  fprintf(fp, "%s", temp.c_str());
  // First, x, y, z
  int everyFour = 0;
  constexpr int numberToChangeLine = 4;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.x(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  everyFour = 0;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.y(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  everyFour = 0;
  for (int k = range_z[0]; k <= range_z[1]; ++k) {
    for (int j = range_y[0]; j <= range_y[1]; ++j) {
      for (int i = range_x[0]; i <= range_x[1]; ++i) {
        fprintf(fp, " %.15e", b.z(i, j, k));
        everyFour++;
        if (everyFour == numberToChangeLine) {
          fprintf(fp, "\n");
          everyFour = 0;
        }
      }
    }
  }
  if (everyFour != 0) {
    fprintf(fp, "\n");
  }
  // Then, the variables
  for (int l = 0; l < nv + 1; ++l) {
    everyFour = 0;
    for (int k = range_z[0]; k <= range_z[1]; ++k) {
      for (int j = range_y[0]; j <= range_y[1]; ++j) {
        for (int i = range_x[0]; i <= range_x[1]; ++i) {
          fprintf(fp, " %.15e", profile_host(i, j, k, l));
          everyFour++;
          if (everyFour == numberToChangeLine) {
            fprintf(fp, "\n");
            everyFour = 0;
          }
        }
      }
    }
    if (everyFour != 0) {
      fprintf(fp, "\n");
    }
  }
  fclose(fp);

  profile_host.deallocate_memory();
}

void read_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile,
  const std::string &profile_related_bc_name) {
  if (file == "MYSELF") {
    // The profile is initialized by the inflow condition.
    initialize_profile_with_inflow(boundary, block, parameter, species, profile, profile_related_bc_name);
    return;
  }
  if (file == "mixingLayerProfile.dat") {
    // The profile is initialized by the inflow condition.
    init_mixingLayer_prof_compatible_cpu(boundary, parameter, block, species, profile);
    return;
  }
  const auto dot = file.find_last_of('.');
  if (const auto suffix = file.substr(dot + 1, file.size()); suffix == "dat") {
    read_dat_profile(boundary, file, block, parameter, species, profile, profile_related_bc_name);
  } else if (suffix == "plt") {
    //    read_plt_profile();
  }
}

std::vector<int>
identify_variable_labels(const Parameter &parameter, std::vector<std::string> &var_name, const Species &species,
  bool &has_pressure, bool &has_temperature, bool &has_tke) {
  std::vector<int> labels;
  const int n_spec = species.n_spec;
  const int n_turb = parameter.get_int("n_turb");
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0, and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0;
    } else if (n == "U") {
      l = 1;
    } else if (n == "V") {
      l = 2;
    } else if (n == "W") {
      l = 3;
    } else if (n == "P" || n == "PRESSURE") {
      l = 4;
      has_pressure = true;
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5;
      has_temperature = true;
    } else {
      if (n_spec > 0) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label;
            break;
          }
        }
        if (n == "MIXTUREFRACTION") {
          // Mixture fraction
          l = 6 + n_spec + n_turb;
        } else if (n == "MIXTUREFRACTIONVARIANCE") {
          // Mixture fraction variance
          l = 6 + n_spec + n_turb + 1;
        }
      }
      if (n_turb > 0) {
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K" || n == "TKE") { // turbulent kinetic energy
          if (n_turb == 2) {
            l = 6 + n_spec;
            has_tke = true;
          }
        } else if (n == "OMEGA") { // specific dissipation rate
          if (n_turb == 2) {
            l = 6 + n_spec + 1;
          }
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          if (n_turb == 1) {
            l = 6 + n_spec;
          }
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

void
read_dat_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile,
  const std::string &profile_related_bc_name) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    MpiParallel::exit();
  }
  const int direction = boundary.face;
  const int n_spec = species.n_spec;

  std::string input;
  std::vector<std::string> var_name;
  read_until(file_in, input, "VARIABLES", gxl::Case::upper);
  while (!(input.substr(0, 4) == "ZONE" || input.substr(0, 5) == " zone")) {
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    auto equal = input.find('=');
    if (equal != std::string::npos)
      input.erase(0, equal + 1);
    std::istringstream line(input);
    std::string v_name;
    while (line >> v_name) {
      var_name.emplace_back(v_name);
    }
    gxl::getline(file_in, input, gxl::Case::upper);
  }
  auto label_order = parameter.identify_variable_labels(var_name, species);

  bool has_pressure = gxl::exists(label_order, 4);
  bool has_temperature = gxl::exists(label_order, 5);
  bool has_tke = gxl::exists(label_order, 6 + n_spec);

  if (!has_temperature && !has_pressure) {
    printf("The temperature or pressure is not given in the profile, please provide at least one of them!\n");
    MpiParallel::exit();
  }
  real turb_viscosity_ratio{0}, turb_intensity{0};
  if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
    auto &info = parameter.get_struct(profile_related_bc_name);
    if (info.find("turb_viscosity_ratio") == info.end() || info.find("turbulence_intensity") == info.end()) {
      printf(
        "The turbulence intensity or turbulent viscosity ratio is not given in the profile, please provide both of them!\n");
      MpiParallel::exit();
    }
    turb_viscosity_ratio = std::get<real>(info.at("turb_viscosity_ratio"));
    turb_intensity = std::get<real>(info.at("turbulence_intensity"));
  }

  int mx, my, mz;
  bool i_read{false}, j_read{false}, k_read{false}, packing_read{false};
  std::string key;
  std::string data_packing{"POINT"};
  while (!(i_read && j_read && k_read && packing_read)) {
    std::getline(file_in, input);
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    gxl::replace(input, '=', ' ');
    std::istringstream line(input);
    while (line >> key) {
      if (key == "i" || key == "I") {
        line >> mx;
        i_read = true;
      } else if (key == "j" || key == "J") {
        line >> my;
        j_read = true;
      } else if (key == "k" || key == "K") {
        line >> mz;
        k_read = true;
      } else if (key == "f" || key == "DATAPACKING" || key == "datapacking") {
        line >> data_packing;
        data_packing = gxl::to_upper(data_packing);
        packing_read = true;
      }
    }
  }

  // This line is the DT=(double ...) line, which must exist if we output the data from Tecplot.
  std::getline(file_in, input);

  int extent[3]{mx, my, mz};

  // Then we read the variables.
  auto nv_read = static_cast<int>(var_name.size());
  gxl::VectorField3D<real> profile_read;
  profile_read.resize(extent[0], extent[1], extent[2], nv_read, 0);

  if (data_packing == "POINT") {
    for (int k = 0; k < extent[2]; ++k) {
      for (int j = 0; j < extent[1]; ++j) {
        for (int i = 0; i < extent[0]; ++i) {
          for (int l = 0; l < nv_read; ++l) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  } else if (data_packing == "BLOCK") {
    for (int l = 0; l < nv_read; ++l) {
      for (int k = 0; k < extent[2]; ++k) {
        for (int j = 0; j < extent[1]; ++j) {
          for (int i = 0; i < extent[0]; ++i) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  }

  const int n_var = parameter.get_int("n_var");
  const int n_scalar = parameter.get_int("n_scalar");

  const auto ngg = block.ngg;
  int range_i[2]{-ngg, block.mx + ngg - 1},
      range_j[2]{-ngg, block.my + ngg - 1},
      range_k[2]{-ngg, block.mz + ngg - 1};
  if (direction == 0) {
    // i direction
    const bool bigNumberFace = boundary.direction == 1;
    if (bigNumberFace) {
      range_i[0] = block.mx - 1;
      range_i[1] = block.mx + ngg - 1;
    } else {
      range_i[0] = -ngg;
      range_i[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    // The 2*ngg ghost layers in x direction are not used.
    profile_to_match.resize(1, block.my, block.mz, n_var + 1, ngg);
    //    ggxl::VectorField2DHost<real> profile_to_match;
    //    profile_to_match.allocate_memory(block.my, block.mz, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        for (int ic = 0; ic <= ngg; ++ic) {
          int i = bigNumberFace ? block.mx - 1 + ic : -ic;
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          // ir for i_real, where real means the real index in the computational mesh.
          // const int ir = bigNumberFace ? extent[0] - 1 + ic : -ic;
          const int ir = -ic;
          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(ir, j, k, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(ir, j, k, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(ir, j, k, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(ir, j, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(ir, j, k, 5) = profile_to_match(ir, j, k, 4) * mw / (R_u * profile_to_match(ir, j, k, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(ir, j, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(ir, j, k, 4) = profile_to_match(ir, j, k, 5) * R_u * profile_to_match(ir, j, k, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu{};
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(ir, j, k, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(ir, j, k, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(ir, j, k, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(ir, j, k, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(ir, j, k, 1) * profile_to_match(ir, j, k, 1) +
                              profile_to_match(ir, j, k, 2) * profile_to_match(ir, j, k, 2) +
                              profile_to_match(ir, j, k, 3) * profile_to_match(ir, j, k, 3);
            profile_to_match(ir, j, k, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(ir, j, k, 6 + n_spec + 1) =
                profile_to_match(ir, j, k, 0) * profile_to_match(ir, j, k, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(1, block.my, block.mz, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 1) {
    // j direction
    const bool bigNumberFace = boundary.direction == 1;
    if (bigNumberFace) {
      range_j[0] = block.my - 1;
      range_j[1] = block.my + ngg - 1;
    } else {
      range_j[0] = -ngg;
      range_j[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, 1, block.mz, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int jc = 0; jc <= ngg; ++jc) {
        for (int i = range_i[0]; i <= range_i[1]; ++i) {
          int j = bigNumberFace ? block.my - 1 + jc : -jc;
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          const int jr = -jc;
          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(i, jr, k, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(i, jr, k, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(i, jr, k, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, jr, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, jr, k, 5) = profile_to_match(i, jr, k, 4) * mw / (R_u * profile_to_match(i, jr, k, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, jr, k, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, jr, k, 4) = profile_to_match(i, jr, k, 5) * R_u * profile_to_match(i, jr, k, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu;
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(i, jr, k, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(i, jr, k, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(i, jr, k, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(i, jr, k, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(i, jr, k, 1) * profile_to_match(i, jr, k, 1) +
                              profile_to_match(i, jr, k, 2) * profile_to_match(i, jr, k, 2) +
                              profile_to_match(i, jr, k, 3) * profile_to_match(i, jr, k, 3);
            profile_to_match(i, jr, k, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(i, jr, k, 6 + n_spec + 1) =
                profile_to_match(i, jr, k, 0) * profile_to_match(i, jr, k, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, 1, block.mz, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 2) {
    // k direction
    const bool bigNumberFace = boundary.direction == 1;
    if (bigNumberFace) {
      range_k[0] = block.mz - 1;
      range_k[1] = block.mz + ngg - 1;
    } else {
      range_k[0] = -ngg;
      range_k[1] = 0;
    }
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, block.my, 1, n_var + 1, ngg);
    // Then we interpolate the profile to the mesh.
    for (int kc = 0; kc <= ngg; ++kc) {
      int k = bigNumberFace ? block.mz - 1 + kc : -kc;
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        for (int i = range_i[0]; i <= range_i[1]; ++i) {
          real d_min = 1e+6;
          int i0 = 0, j0 = 0, k0 = 0;
          for (int kk = 0; kk < extent[2]; ++kk) {
            for (int jj = 0; jj < extent[1]; ++jj) {
              for (int ii = 0; ii < extent[0]; ++ii) {
                real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                              (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                              (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                              (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
                if (d <= d_min) {
                  d_min = d;
                  i0 = ii;
                  j0 = jj;
                  k0 = kk;
                }
              }
            }
          }

          const int kr = -kc;
          // Assign the values in 0th order
          for (int l = 3; l < nv_read; ++l) {
            if (label_order[l] < 6) {
              // Basic variables
              profile_to_match(i, j, kr, label_order[l]) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] >= 1000 && label_order[l] < 1000 + n_spec) {
              // Species variables
              int ls = label_order[l] - 1000;
              profile_to_match(i, j, kr, 6 + ls) = profile_read(i0, j0, k0, l);
            } else if (label_order[l] < 6 + n_scalar - n_spec) {
              // Turbulence variables or mixture fraction variables
              int ls = label_order[l] + n_spec;
              profile_to_match(i, j, kr, ls) = profile_read(i0, j0, k0, l);
            }
          }

          // If T or p is not given, compute it.
          if (!has_temperature) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, j, kr, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, j, kr, 5) = profile_to_match(i, j, kr, 4) * mw / (R_u * profile_to_match(i, j, kr, 0));
          }
          if (!has_pressure) {
            real mw{mw_air};
            if (n_spec > 0) {
              mw = 0;
              for (int l = 0; l < n_spec; ++l) mw += profile_to_match(i, j, kr, 6 + l) / species.mw[l];
              mw = 1 / mw;
            }
            profile_to_match(i, j, kr, 4) = profile_to_match(i, j, kr, 5) * R_u * profile_to_match(i, j, kr, 0) / mw;
          }
          if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
            // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
            real mu;
            if (n_spec > 0) {
              real mw = 0;
              std::vector<real> Y;
              for (int l = 0; l < n_spec; ++l) {
                mw += profile_to_match(i, j, kr, 6 + l) / species.mw[l];
                Y.push_back(profile_to_match(i, j, kr, 6 + l));
              }
              mw = 1 / mw;
              mu = compute_viscosity(profile_to_match(i, j, kr, 5), mw, Y.data(), species);
            } else {
              mu = Sutherland(profile_to_match(i, j, kr, 5));
            }
            real mut = mu * turb_viscosity_ratio;
            const real vel2 = profile_to_match(i, j, kr, 1) * profile_to_match(i, j, kr, 1) +
                              profile_to_match(i, j, kr, 2) * profile_to_match(i, j, kr, 2) +
                              profile_to_match(i, j, kr, 3) * profile_to_match(i, j, kr, 3);
            profile_to_match(i, j, kr, 6 + n_spec) = 1.5 * vel2 * turb_intensity * turb_intensity;
            profile_to_match(i, j, kr, 6 + n_spec + 1) =
                profile_to_match(i, j, kr, 0) * profile_to_match(i, j, kr, 6 + n_spec) / mut;
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, block.my, 1, n_var + 1, ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  }
}

__global__ void
initialize_rng(curandState *rng_states, int size, int64_t time_stamp) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= size)
    return;

  curand_init(time_stamp + i, i, 0, &rng_states[i]);
}

void read_lst_profile(const Boundary &boundary, const std::string &file, const Block &block, const Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile,
  const std::string &profile_related_bc_name) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    MpiParallel::exit();
  }
  const int direction = boundary.face;

  std::string input;
  std::vector<std::string> var_name;
  read_until(file_in, input, "VARIABLES", gxl::Case::upper);
  while (!(input.substr(0, 4) == "ZONE" || input.substr(0, 5) == " zone")) {
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    auto equal = input.find('=');
    if (equal != std::string::npos)
      input.erase(0, equal + 1);
    std::istringstream line(input);
    std::string v_name;
    while (line >> v_name) {
      var_name.emplace_back(v_name);
    }
    gxl::getline(file_in, input, gxl::Case::upper);
  }
  bool has_pressure{false}, has_temperature{false}, has_rho{false};
  // Identify the labels of the variables
  std::vector<int> label_order;
  const int n_spec = species.n_spec;
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0, and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "RHOR") {
      l = 0;
      has_rho = true;
    } else if (n == "UR") {
      l = 1;
    } else if (n == "VR") {
      l = 2;
    } else if (n == "WR") {
      l = 3;
    } else if (n == "PR") {
      l = 4;
      has_pressure = true;
    } else if (n == "TR") {
      l = 5;
      has_temperature = true;
    } else if (n == "RHOI") {
      l = 6;
    } else if (n == "UI") {
      l = 7;
    } else if (n == "VI") {
      l = 8;
    } else if (n == "WI") {
      l = 9;
    } else if (n == "PI") {
      l = 10;
    } else if (n == "TI") {
      l = 11;
    } else {
      if (n_spec > 0) {
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec) + "R") {
            l = 12 + sp_label;
            break;
          }
          if (n == gxl::to_upper(spec) + "I") {
            l = 12 + n_spec + sp_label;
            break;
          }
        }
      }
    }
    label_order.emplace_back(l);
  }
  // At least 2 of the 3 variables, pressure, temperature and density should be given.
  if (!has_rho) {
    if (!has_pressure || !has_temperature) {
      printf(
        "The fluctuation of density, temperature or pressure is not given in the profile, please provide at least two of them!\n");
      MpiParallel::exit();
    }
  } else if (!has_pressure) {
    if (!has_temperature) {
      printf(
        "The fluctuation of temperature or pressure is not given in the profile when rho is given, please provide at least one of them!\n");
      MpiParallel::exit();
    }
  }

  int mx, my, mz;
  bool i_read{false}, j_read{false}, k_read{false}, packing_read{false};
  std::string key;
  std::string data_packing{"POINT"};
  while (!(i_read && j_read && k_read && packing_read)) {
    std::getline(file_in, input);
    gxl::replace(input, '"', ' ');
    gxl::replace(input, ',', ' ');
    gxl::replace(input, '=', ' ');
    std::istringstream line(input);
    while (line >> key) {
      if (key == "i" || key == "I") {
        line >> mx;
        i_read = true;
      } else if (key == "j" || key == "J") {
        line >> my;
        j_read = true;
      } else if (key == "k" || key == "K") {
        line >> mz;
        k_read = true;
      } else if (key == "f" || key == "DATAPACKING" || key == "datapacking") {
        line >> data_packing;
        data_packing = gxl::to_upper(data_packing);
        packing_read = true;
      }
    }
  }

  // This line is the DT=(double ...) line, which must exist if we output the data from Tecplot.
  std::getline(file_in, input);

  int extent[3]{mx, my, mz};

  // Then we read the variables.
  auto nv_read = static_cast<int>(var_name.size());
  gxl::VectorField3D<real> profile_read;
  profile_read.resize(extent[0], extent[1], extent[2], nv_read, 0);

  if (data_packing == "POINT") {
    for (int k = 0; k < extent[2]; ++k) {
      for (int j = 0; j < extent[1]; ++j) {
        for (int i = 0; i < extent[0]; ++i) {
          for (int l = 0; l < nv_read; ++l) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  } else if (data_packing == "BLOCK") {
    for (int l = 0; l < nv_read; ++l) {
      for (int k = 0; k < extent[2]; ++k) {
        for (int j = 0; j < extent[1]; ++j) {
          for (int i = 0; i < extent[0]; ++i) {
            file_in >> profile_read(i, j, k, l);
          }
        }
      }
    }
  }

  const int n_var = parameter.get_int("n_var");
  const auto ngg = block.ngg;
  int range_i[2]{-ngg, block.mx + ngg - 1},
      range_j[2]{-ngg, block.my + ngg - 1},
      range_k[2]{-ngg, block.mz + ngg - 1};
  if (direction == 0) {
    // i direction
    ggxl::VectorField3DHost<real> profile_to_match;
    // The 2*ngg ghost layers in x direction are not used.
    profile_to_match.resize(1, block.my, block.mz, 2 * (n_var + 1), ngg);
    const int i = boundary.direction == 1 ? block.mx - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int j = range_j[0]; j <= range_j[1]; ++j) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1 + 3)) {
            profile_to_match(0, j, k, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(1, block.my, block.mz, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 1) {
    // j direction
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, 1, block.mz, 2 * (n_var + 1), ngg);
    const int j = boundary.direction == 1 ? block.my - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int k = range_k[0]; k <= range_k[1]; ++k) {
      for (int i = range_i[0]; i <= range_i[1]; ++i) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1)) {
            profile_to_match(i, 0, k, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, 1, block.mz, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  } else if (direction == 2) {
    // k direction
    ggxl::VectorField3DHost<real> profile_to_match;
    profile_to_match.resize(block.mx, block.my, 1, 2 * (n_var + 1), ngg);
    const int k = boundary.direction == 1 ? block.mz - 1 : 0;
    // Then we interpolate the profile to the mesh.
    for (int j = range_j[0]; j <= range_j[1]; ++j) {
      for (int i = range_i[0]; i <= range_i[1]; ++i) {
        real d_min = 1e+6;
        int i0 = 0, j0 = 0, k0 = 0;
        for (int kk = 0; kk < extent[2]; ++kk) {
          for (int jj = 0; jj < extent[1]; ++jj) {
            for (int ii = 0; ii < extent[0]; ++ii) {
              real d = sqrt((block.x(i, j, k) - profile_read(ii, jj, kk, 0)) *
                            (block.x(i, j, k) - profile_read(ii, jj, kk, 0)) +
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) *
                            (block.y(i, j, k) - profile_read(ii, jj, kk, 1)) +
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)) *
                            (block.z(i, j, k) - profile_read(ii, jj, kk, 2)));
              if (d <= d_min) {
                d_min = d;
                i0 = ii;
                j0 = jj;
                k0 = kk;
              }
            }
          }
        }

        // Assign the values in 0th order
        for (int l = 3; l < nv_read; ++l) {
          if (label_order[l] < 2 * (n_var + 1)) {
            profile_to_match(i, j, 0, label_order[l]) = profile_read(i0, j0, k0, l);
          }
        }
      }
    }
    // Then we copy the data to the profile array.
    profile.allocate_memory(block.mx, block.my, 1, 2 * (n_var + 1), ngg);
    cudaMemcpy(profile.data(), profile_to_match.data(), sizeof(real) * profile_to_match.size() * 2 * (n_var + 1),
               cudaMemcpyHostToDevice);
    profile_to_match.deallocate_memory();
  }
}

void
DBoundCond::initialize_profile_and_rng(Parameter &parameter, Mesh &mesh, const Species &species,
  std::vector<Field> &field) {
  if (const int n_profile = parameter.get_int("n_profile"); n_profile > 0) {
    profile_hPtr_withGhost.resize(n_profile);
    for (int i = 0; i < n_profile; ++i) {
      const auto file_name = parameter.get_string_array("profile_file_names")[i];
      const auto profile_related_bc_name = parameter.get_string_array("profile_related_bc_names")[i];
      const auto &nn = parameter.get_struct(profile_related_bc_name);
      const auto label = std::get<int>(nn.at("label"));
      for (int blk = 0; blk < mesh.n_block; ++blk) {
        auto &bs = mesh[blk].boundary;
        for (auto &b: bs) {
          if (b.type_label == label) {
            read_profile(b, file_name, mesh[blk], parameter, species, profile_hPtr_withGhost[i],
                         profile_related_bc_name);
            break;
          }
        }
      }
    }
    cudaMalloc(&profile_dPtr_withGhost, sizeof(ggxl::VectorField3D<real>) * n_profile);
    cudaMemcpy(profile_dPtr_withGhost, profile_hPtr_withGhost.data(), sizeof(ggxl::VectorField3D<real>) * n_profile,
               cudaMemcpyHostToDevice);
  }

  // Count the max number of rng needed
  auto size{0};
  if (const auto need_rng = parameter.get_int_array("need_rng"); !need_rng.empty()) {
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (const auto &b: bs) {
        if (gxl::exists(need_rng, b.type_label)) {
          int n1{mesh[blk].my}, n2{mesh[blk].mz};
          if (b.face == 1) {
            n1 = mesh[blk].mx;
          } else if (b.face == 2) {
            n1 = mesh[blk].mx;
            n2 = mesh[blk].my;
          }
          const auto ngg = mesh[blk].ngg;
          const auto this_size = (n1 + 2 * ngg) * (n2 + 2 * ngg);
          if (this_size > size)
            size = this_size;
        }
      }
    }
  }
  if (size > 0) {
    cudaMalloc(&rng_d_ptr, sizeof(curandState) * size);
    dim3 TPB = {128, 1, 1};
    dim3 BPG = {(size - 1) / TPB.x + 1, 1, 1};
    // Get the current time
    time_t time_curr;
    initialize_rng<<<BPG, TPB>>>(rng_d_ptr, size, time(&time_curr));
  }

  // Read the fluctuation profiles if needed
  const auto need_fluctuation_profile = parameter.get_int_array("need_fluctuation_profile");
  if (!need_fluctuation_profile.empty()) {
    std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr;
    const int n_fluc_profile = static_cast<int>(need_fluctuation_profile.size());
    fluctuation_hPtr.resize(n_fluc_profile);
    for (int i = 0; i < n_fluc_profile; ++i) {
      const auto file_name = parameter.get_string_array("fluctuation_profile_file")[i];
      auto bc_name = parameter.get_string_array("fluctuation_profile_related_bc_name")[i];
      const auto &nn = parameter.get_struct(bc_name);
      const auto label = std::get<int>(nn.at("label"));
      for (int blk = 0; blk < mesh.n_block; ++blk) {
        auto &bs = mesh[blk].boundary;
        for (auto &b: bs) {
          if (b.type_label == label) {
            read_lst_profile(b, file_name, mesh[blk], parameter, species, fluctuation_hPtr[i], bc_name);
            break;
          }
        }
      }
    }
    cudaMalloc(&fluctuation_dPtr, sizeof(ggxl::VectorField3D<real>) * n_fluc_profile);
    cudaMemcpy(fluctuation_dPtr, fluctuation_hPtr.data(), sizeof(ggxl::VectorField3D<real>) * n_fluc_profile,
               cudaMemcpyHostToDevice);
  }

  // If the white noise is added on wall, we need to record the fluctuation values
  if (parameter.get_bool("wall_white_noise")) {
    std::vector<ggxl::VectorField3D<real>> fluctuation_hPtr;
    fluctuation_hPtr.resize(1);
    const auto &nn = parameter.get_struct("wall");
    const auto label = std::get<int>(nn.at("label"));
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      auto &bs = mesh[blk].boundary;
      for (auto &b: bs) {
        if (b.type_label == label) {
          fluctuation_hPtr[0].allocate_memory(mesh[blk].mx, 1, mesh[blk].mz, 1, mesh[blk].ngg);
          cudaMalloc(&fluctuation_dPtr, sizeof(ggxl::VectorField3D<real>) * 1);
          cudaMemcpy(fluctuation_dPtr, &fluctuation_hPtr, sizeof(ggxl::VectorField3D<real>), cudaMemcpyHostToDevice);
          break;
        }
      }
    }
  }

  // If random number generators are needed for all points
  if (const int n_rand = parameter.get_int("random_number_per_point"); n_rand > 0) {
    const int initialize = parameter.get_int("initial");
    const int n_fluc_val = parameter.get_int("fluctuation_variable_number");
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      const int mx = mesh[blk].mx, my = mesh[blk].my, mz = mesh[blk].mz, ngg = mesh[blk].ngg;
      bool init = true;
      if (initialize == 1) {
        std::string filename = "./output/rng-p" + std::to_string(parameter.get_int("myid")) + ".bin";
        MPI_File fp;
        MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
        if (fp == nullptr) {
          printf("Warning: cannot open the file %s, the random number states will be restarted.\n", filename.c_str());
          break;
        }
        int n1_read, n2_read, n3_read;
        MPI_Offset offset = 0;
        MPI_File_read_at(fp, offset, &n1_read, 1, MPI_INT, MPI_STATUS_IGNORE);
        offset += 4;
        MPI_File_read_at(fp, offset, &n2_read, 1, MPI_INT, MPI_STATUS_IGNORE);
        offset += 4;
        MPI_File_read_at(fp, offset, &n3_read, 1, MPI_INT, MPI_STATUS_IGNORE);
        offset += 4;
        int ngg_read;
        MPI_File_read_at(fp, offset, &ngg_read, 1, MPI_INT, MPI_STATUS_IGNORE);
        offset += 4;
        if (mx != n1_read || my != n2_read || mz != n3_read || ngg != ngg_read) {
          printf(
            "Warning: the grid size in the file %s is not consistent with the current grid size, the random number states will be restarted.\n",
            filename.c_str());
          break;
        }

        init = false;
        MPI_Datatype ty;
        int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
        int start[3]{0, 0, 0};
        // The old data type is curandState
        MPI_Datatype mpi_curandState;
        MPI_Type_contiguous(sizeof(curandState), MPI_BYTE, &mpi_curandState);
        MPI_Type_commit(&mpi_curandState);
        MPI_Type_create_subarray(3, lSize, lSize, start, MPI_ORDER_FORTRAN, mpi_curandState, &ty);
        MPI_Type_commit(&ty);
        MPI_File_read_at(fp, offset, field[blk].rng_state.data(), n_rand, ty, MPI_STATUS_IGNORE);
        offset += static_cast<MPI_Offset>((mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * n_rand * sizeof(
                                            curandState));
        cudaMemcpy(field[blk].h_ptr->rng_state.data(), field[blk].rng_state.data(),
                   (my + 2 * ngg) * (mz + 2 * ngg) * 3 * sizeof(curandState), cudaMemcpyHostToDevice);
        MPI_Type_free(&ty);
        MPI_Type_free(&mpi_curandState);

        // The fluctuation values
        ggxl::VectorField3DHost<real> hData;
        hData.resize(mx, my, mz, n_fluc_val, ngg);
        if (ngg_read >= ngg) {
          MPI_Datatype tty;
          int LSz[3]{mx + 2 * ngg_read, my + 2 * ngg_read, mz + 2 * ngg_read};
          int sSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
          int START[3]{ngg_read - ngg, ngg_read - ngg, ngg_read - ngg};
          MPI_Type_create_subarray(3, LSz, sSize, START, MPI_ORDER_FORTRAN, MPI_DOUBLE, &tty);
          MPI_Type_commit(&tty);

          MPI_File_read_at(fp, offset, hData.data(), n_fluc_val, tty, MPI_STATUS_IGNORE);
          MPI_Type_free(&tty);
        } else {
          for (int l = 0; l < n_fluc_val; ++l) {
            for (int k = -ngg_read; k < mz + ngg_read; ++k) {
              for (int j = -ngg_read; j < my + ngg_read; ++j) {
                for (int i = -ngg_read; i < mx + ngg_read; ++i) {
                  MPI_File_read_at(fp, offset, &hData(i, j, k, l), 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
                  offset += sizeof(real);
                }
              }
            }
          }
        }
        cudaMemcpy(field[blk].h_ptr->fluc_val.data(), hData.data(), hData.size() * n_fluc_val * sizeof(real),
                   cudaMemcpyHostToDevice);
        hData.deallocate_memory();
      }

      if (init) {
        dim3 TPB = {16, 8, 8};
        dim3 BPG = {(mx + 2 * ngg - 1) / TPB.x + 1, (my + 2 * ngg - 1) / TPB.y + 1, (mz + 2 * ngg - 1) / TPB.z + 1};
        initialize_rng<<<BPG, TPB>>>(field[blk].d_ptr, n_rand);
      }
    }
  }
}

__global__ void initialize_rng(DZone *zone, int n_rand) {
  const int ngg = zone->ngg, n1 = zone->mx, n2 = zone->my, n3 = zone->mz;
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= n1 + ngg || j >= n2 + ngg || k >= n3 + ngg)
    return;

  auto &rng = zone->rng_state;
  for (int l = 0; l < n_rand; ++l)
    curand_init((i + ngg) * n1 + (j + ngg) * n2 + (k + ngg) * n3, l, 0, &rng(i, j, k, l));
}

void write_rng(const Mesh &mesh, Parameter &parameter, std::vector<Field> &field) {
  const int n_rand = parameter.get_int("random_number_per_point");
  if (n_rand < 1)
    return;
  const int myid = parameter.get_int("myid");
  printf("Process %d is writing the white noise to the file.\n", myid);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }
  std::string filename = "./output/rng-p" + std::to_string(myid) + ".bin";
  FILE *fp = fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    printf("Error: cannot open the file %s.\n", filename.c_str());
    MpiParallel::exit();
  }
  const int n_val = parameter.get_int("fluctuation_variable_number");
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const int mx = mesh[blk].mx, my = mesh[blk].my, mz = mesh[blk].mz, ngg = mesh[blk].ngg;
    fwrite(&mx, sizeof(int), 1, fp);
    fwrite(&my, sizeof(int), 1, fp);
    fwrite(&mz, sizeof(int), 1, fp);
    fwrite(&ngg, sizeof(int), 1, fp);

    fwrite(field[blk].rng_state.data(), sizeof(curandState),
           (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * n_rand, fp);
    fwrite(field[blk].fluc_val.data(), sizeof(real),
           (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * n_val, fp);
    fclose(fp);
  }
}

__global__ void
initialize_rest_rng(ggxl::VectorField2D<curandState> *rng_states, int iFace, int64_t time_stamp, int dy, int dz,
  int ngg, int my, int mz) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - DBoundCond::DF_N - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - DBoundCond::DF_N - ngg;
  if (j >= my + DBoundCond::DF_N + ngg || k >= mz + DBoundCond::DF_N + ngg) {
    return;
  }
  if (j < -ngg - DBoundCond::DF_N + dy || j > my + ngg + DBoundCond::DF_N - 1 - dy ||
      k < -ngg - DBoundCond::DF_N + dz || k > mz + ngg + DBoundCond::DF_N - 1 - dz) {
    const int sz = (my + 2 * ngg + 2 * DBoundCond::DF_N) * (mz + 2 * ngg + 2 * DBoundCond::DF_N);
    int i = k * (my + 2 * ngg + 2 * DBoundCond::DF_N) + j +
            (my + 2 * ngg + 2 * DBoundCond::DF_N + 1) * (ngg + DBoundCond::DF_N);
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 0));
    i += sz;
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 1));
    i += sz;
    curand_init(time_stamp + i, i, 0, &rng_states[iFace](j, k, 2));
  }
}
}

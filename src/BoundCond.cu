#include "BoundCond.cuh"
#include "Parallel.h"
#include "MixingLayer.cuh"
#include "gxl_lib/MyString.h"
#include <fstream>
#include "gxl_lib/MyAlgorithm.h"
#include "DParameter.cuh"
#include "FieldOperation.cuh"

namespace cfd {
template<typename BCType> void register_bc(BCType *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
  Species &species, Parameter &parameter) {
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

template<> void register_bc<Wall>(Wall *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info, Species &species,
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
      bc_info[i].label = bc_label;
      Wall wall(this_bc, parameter, species);
      wall.copy_to_gpu(&bc[i], parameter);
      break;
    }
  }
}

template<> void register_bc<Inflow>(Inflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
  Species &species, Parameter &parameter) {
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

template<> void register_bc<FarField>(FarField *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
  Species &species, Parameter &parameter) {
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

template<> void register_bc<SubsonicInflow>(SubsonicInflow *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
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

template<> void register_bc<BackPressure>(BackPressure *&bc, int n_bc, std::vector<int> &indices, BCInfo *&bc_info,
  Species &species, Parameter &parameter) {
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

void Wall::copy_to_gpu(Wall *d_wall, const Parameter &parameter) {
  if (catalytic_type != 0) {
    const int ns{parameter.get_int("n_spec")};
    const auto h_sv = new real[ns];
    for (int l = 0; l < ns; ++l) {
      h_sv[l] = yk[l];
    }
    delete[]yk;
    cudaMalloc(&yk, ns * sizeof(real));
    cudaMemcpy(yk, h_sv, ns * sizeof(real), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_wall, this, sizeof(Wall), cudaMemcpyHostToDevice);
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
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name) {
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

  // Ferrer's method
  for (int j = range_y[0]; j <= range_y[1]; ++j) {
    real y = b.y(0, j, 0);
    const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
    const real hyperbolicTanh_val = tanh(2 * y / delta_omega);
    real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * hyperbolicTanh_val;
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

      real imw = 0, imw_upper = 0, imw_lower = 0;
      for (int l = 0; l < n_spec; ++l) {
        yk[l] = 0.5 * (sv_upper[l] + sv_lower[l]) +
                0.5 * (sv_upper[l] - sv_lower[l]) * hyperbolicTanh_val;
        imw += yk[l] / species.mw[l];
        imw_upper += sv_upper[l] / species.mw[l];
        imw_lower += sv_lower[l] / species.mw[l];
      }

      const real rho_upper = p / (t_upper * R_u * imw_upper);
      const real rho_lower = p / (t_lower * R_u * imw_lower);
      density = 0.5 * (rho_upper + rho_lower) +
                0.5 * (rho_upper - rho_lower) * hyperbolicTanh_val;

      t = p / (imw * R_u * density);

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
      const real rho_upper = p / (R_air * t_upper);
      const real rho_lower = p / (R_air * t_lower);
      density = 0.5 * (rho_upper + rho_lower) +
                0.5 * (rho_upper - rho_lower) * hyperbolicTanh_val;
      t = p / (R_air * density);

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

  // // Chen Qian's compatible mixing layer profile
  // for (int j = range_y[0]; j <= range_y[1]; ++j) {
  //   real y = b.y(0, j, 0);
  //   const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
  //   real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / delta_omega);
  //   real p = var_info[4];
  //   real density, t;
  //   const real t_upper = var_info[5], t_lower = var_info[12 + n_spec];
  //   real yk[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER + 4] = {};
  //
  //   real y_upper = (u - u_lower) / (u_upper - u_lower);
  //   real y_lower = 1 - y_upper;
  //
  //   if (n_spec > 0) {
  //     // multi-species
  //     auto sv_upper = &var_info[6];
  //     auto sv_lower = &var_info[7 + n_spec + 6];
  //
  //     for (int l = 0; l < n_spec; ++l) {
  //       yk[l] = y_upper * sv_upper[l] + y_lower * sv_lower[l];
  //     }
  //
  //     // compute the total enthalpy of upper and lower streams
  //     real h0_upper{0.5 * u_upper * u_upper}, h0_lower{0.5 * u_lower * u_lower};
  //
  //     real h_upper[MAX_SPEC_NUMBER], h_lower[MAX_SPEC_NUMBER];
  //     species.compute_enthalpy(t_upper, h_upper);
  //     species.compute_enthalpy(t_lower, h_lower);
  //
  //     real mw_inv = 0;
  //     for (int l = 0; l < n_spec; ++l) {
  //       h0_upper += yk[l] * h_upper[l];
  //       h0_lower += yk[l] * h_lower[l];
  //       mw_inv += yk[l] / species.mw[l];
  //     }
  //
  //     real h = y_upper * h0_upper + y_lower * h0_lower;
  //     h -= 0.5 * u * u;
  //
  //     auto hs = h_upper, cps = h_lower;
  //     real err{1};
  //     t = t_upper * y_upper + t_lower * y_lower;
  //     constexpr int max_iter{1000};
  //     constexpr real eps{1e-3};
  //     int iter = 0;
  //
  //     while (err > eps && iter++ < max_iter) {
  //       species.compute_enthalpy_and_cp(t, hs, cps);
  //       real cp{0}, h_new{0};
  //       for (int l = 0; l < n_spec; ++l) {
  //         cp += yk[l] * cps[l];
  //         h_new += yk[l] * hs[l];
  //       }
  //       const real t1 = t - (h_new - h) / cp;
  //       err = abs(1 - t1 / t);
  //       t = t1;
  //     }
  //     density = p / (R_u * mw_inv * t);
  //
  //     if (n_fl > 0) {
  //       yk[i_fl] = var_info[6 + n_spec] * y_upper + var_info[13 + n_spec + n_spec] * y_lower;
  //       yk[i_fl + 1] = 0;
  //     }
  //     if (n_ps > 0) {
  //       for (int l = 0; l < n_ps; ++l) {
  //         yk[i_ps + l] =
  //             var_info[14 + 2 * n_spec + 4 + 2 * l] * y_upper + var_info[14 + 2 * n_spec + 4 + 2 * l + 1] * y_lower;
  //       }
  //     }
  //   } else {
  //     // Air
  //     constexpr real cp = gamma_air * R_air / (gamma_air - 1);
  //     real h0_upper = 0.5 * u_upper * u_upper + cp * var_info[5];
  //     real h0_lower = 0.5 * u_lower * u_lower + cp * var_info[12 + n_spec];
  //     real h = y_upper * h0_upper + y_lower * h0_lower - 0.5 * u * u;
  //     t = h / cp;
  //     density = p / (R_air * t);
  //
  //     if (n_ps > 0) {
  //       if (y > 0) {
  //         for (int l = 0; l < n_ps; ++l) {
  //           yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l];
  //         }
  //       } else {
  //         for (int l = 0; l < n_ps; ++l) {
  //           yk[i_ps + l] = var_info[14 + 2 * n_spec + 4 + 2 * l + 1];
  //         }
  //       }
  //     }
  //   }
  //   if (n_turb > 0) {
  //     if (y > 0) {
  //       for (int l = 0; l < n_turb; ++l) {
  //         yk[l + n_spec] = var_info[13 + 2 * n_spec + 1 + l];
  //       }
  //     } else {
  //       for (int l = 0; l < n_turb; ++l) {
  //         yk[l + n_spec] = var_info[13 + 2 * n_spec + n_turb + 1 + l];
  //       }
  //     }
  //   }
  //
  //   auto &prof = profile_host;
  //   for (int k = range_z[0]; k <= range_z[1]; ++k) {
  //     for (int i = range_x[0]; i <= range_x[1]; ++i) {
  //       prof(i, j, k, 0) = density;
  //       prof(i, j, k, 1) = u;
  //       prof(i, j, k, 2) = 0;
  //       prof(i, j, k, 3) = 0;
  //       prof(i, j, k, 4) = p;
  //       prof(i, j, k, 5) = t;
  //       for (int l = 0; l < n_scalar; ++l) {
  //         prof(i, j, k, 6 + l) = yk[l];
  //       }
  //     }
  //   }
  // }

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
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name) {
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

std::vector<int> identify_variable_labels(const Parameter &parameter, std::vector<std::string> &var_name,
  const Species &species, bool &has_pressure, bool &has_temperature, bool &has_tke) {
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

void read_dat_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name) {
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

  if (!has_temperature && !has_pressure) {
    printf("The temperature or pressure is not given in the profile, please provide at least one of them!\n");
    MpiParallel::exit();
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

__global__ void initialize_rng(curandState *rng_states, int size, int64_t time_stamp) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= size)
    return;

  curand_init(time_stamp + i, i, 0, &rng_states[i]);
}

void read_lst_profile(const Boundary &boundary, const std::string &file, const Block &block, const Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name) {
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

void DBoundCond::initialize_profile_and_rng(Parameter &parameter, Mesh &mesh, const Species &species,
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

__global__ void initialize_rest_rng(ggxl::VectorField2D<curandState> *rng_states, int iFace, int64_t time_stamp, int dy,
  int dz, int ngg, int my, int mz) {
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

template<MixtureModel mix_model> __global__ void apply_symmetry(DZone *zone, int i_face, DParameter *param) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  const auto face = b.face;
  int dir[]{0, 0, 0};
  dir[face] = b.direction;

  const int inner_idx[3]{i - dir[0], j - dir[1], k - dir[2]};

  const auto &metric = zone->metric;
  real k_x{metric(i, j, k, face * 3)}, k_y{metric(i, j, k, face * 3 + 1)}, k_z{metric(i, j, k, face * 3 + 2)};
  const real k_magnitude = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
  k_x /= k_magnitude;
  k_y /= k_magnitude;
  k_z /= k_magnitude;

  auto &bv = zone->bv;
  const real u1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 1)},
      v1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 2)},
      w1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 3)};
  real u_k{k_x * u1 + k_y * v1 + k_z * w1};
  const real u_t{u1 - k_x * u_k}, v_t{v1 - k_y * u_k}, w_t{w1 - k_z * u_k};

  // The gradient of tangential velocity should be zero.
  bv(i, j, k, 1) = u_t;
  bv(i, j, k, 2) = v_t;
  bv(i, j, k, 3) = w_t;
  // The gradient of pressure, density, and scalars should also be zero.
  bv(i, j, k, 0) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 0);
  bv(i, j, k, 4) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 4);
  bv(i, j, k, 5) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 5);
  auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = sv(inner_idx[0], inner_idx[1], inner_idx[2], l);
  }

  // For ghost grids
  for (int g = 1; g <= zone->ngg; ++g) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

    bv(gi, gj, gk, 0) = bv(ii, ij, ik, 0);

    const auto u{bv(ii, ij, ik, 1)}, v{bv(ii, ij, ik, 2)}, w{bv(ii, ij, ik, 3)};
    u_k = k_x * u + k_y * v + k_z * w;
    bv(gi, gj, gk, 1) = u - 2 * u_k * k_x;
    bv(gi, gj, gk, 2) = v - 2 * u_k * k_y;
    bv(gi, gj, gk, 3) = w - 2 * u_k * k_z;
    bv(gi, gj, gk, 4) = bv(ii, ij, ik, 4);
    bv(gi, gj, gk, 5) = bv(ii, ij, ik, 5);
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }

    compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model> __global__ void apply_outflow(DZone *zone, Outflow *outflow, int i_face,
  const DParameter *param) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  if (outflow->if_backPressure) {
    // The multi-species type is not implemented.
    real nx{zone->metric(i, j, k, b.face * 3)},
        ny{zone->metric(i, j, k, b.face * 3 + 1)},
        nz{zone->metric(i, j, k, b.face * 3 + 2)};
    const real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
    nx *= grad_n_inv;
    ny *= grad_n_inv;
    nz *= grad_n_inv;
    real u_b{bv(i, j, k, 1)}, v_b{bv(i, j, k, 2)}, w_b{bv(i, j, k, 3)};
    const real u_face{nx * u_b + ny * v_b + nz * w_b};
    real p_b{bv(i, j, k, 4)}, rho_b{bv(i, j, k, 0)};
    real gamma = gamma_air;
    real yk_b[MAX_SPEC_NUMBER];
    if constexpr (mix_model != MixtureModel::Air) {
      real cpk[MAX_SPEC_NUMBER];
      compute_cp(bv(i, j, k, 5), cpk, param);
      real cp = 0, R = 0;
      for (int l = 0; l < param->n_spec; ++l) {
        yk_b[l] = sv(i, j, k, l);
        cp += sv(i, j, k, l) * cpk[l];
        R += sv(i, j, k, l) * param->gas_const[l];
      }
      gamma = cp / (cp - R);
    }
    const real a_b = sqrt(gamma * p_b / rho_b);
    const real mach_b{abs(u_face / a_b)};

    if (mach_b < 1) {
      p_b = outflow->backPressure;
      const int i1 = i - dir[0], j1 = j - dir[1], k1 = k - dir[2];
      const real d1{bv(i1, j1, k1, 0)}, u1{bv(i1, j1, k1, 1)}, v1{bv(i1, j1, k1, 2)}, w1{bv(i1, j1, k1, 3)},
          p1{bv(i1, j1, k1, 4)};

      gamma = gamma_air;
      real yk1[MAX_SPEC_NUMBER];
      if constexpr (mix_model != MixtureModel::Air) {
        real cpk[MAX_SPEC_NUMBER];
        compute_cp(bv(i1, j1, k1, 5), cpk, param);
        real cp = 0, R = 0;
        for (int l = 0; l < param->n_spec; ++l) {
          yk1[l] = sv(i1, j1, k1, l);
          cp += sv(i1, j1, k1, l) * cpk[l];
          R += sv(i1, j1, k1, l) * param->gas_const[l];
        }
        gamma = cp / (cp - R);
      }
      const real c1{sqrt(gamma * p1 / d1)};
      const auto temp = (p_b - p1) / c1;
      rho_b = d1 + temp / c1;
      u_b = u1 - nx * temp / d1;
      v_b = v1 - ny * temp / d1;
      w_b = w1 - nz * temp / d1;
      if constexpr (mix_model != MixtureModel::Air) {
        for (int l = 0; l < param->n_spec; ++l) {
          yk_b[l] = yk1[l] + yk1[l] * temp / (d1 * c1);
        }
      }

      for (int g = 1; g <= ngg; ++g) {
        const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
        const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

        const real p_g{2 * p_b - bv(ii, ij, ik, 4)}, rho_g{2 * rho_b - bv(ii, ij, ik, 0)};

        bv(gi, gj, gk, 0) = rho_g;
        bv(gi, gj, gk, 1) = 2 * u_b - bv(ii, ij, ik, 1);
        bv(gi, gj, gk, 2) = 2 * v_b - bv(ii, ij, ik, 2);
        bv(gi, gj, gk, 3) = 2 * w_b - bv(ii, ij, ik, 3);
        bv(gi, gj, gk, 4) = p_g;
        bv(gi, gj, gk, 5) = p_g / (rho_g * R_air);
        for (int l = 0; l < param->n_scalar; ++l) {
          sv(gi, gj, gk, l) = sv(i, j, k, l);
        }

        compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
      }
    } else {
      for (int g = 1; g <= ngg; ++g) {
        const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};

        bv(gi, gj, gk, 0) = rho_b;
        bv(gi, gj, gk, 1) = u_b;
        bv(gi, gj, gk, 2) = v_b;
        bv(gi, gj, gk, 3) = w_b;
        bv(gi, gj, gk, 4) = p_b;
        bv(gi, gj, gk, 5) = p_b / (rho_b * R_air);
        for (int l = 0; l < param->n_scalar; ++l) {
          sv(gi, gj, gk, l) = sv(i, j, k, l);
        }

        compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
      }
    }
  } else {
    for (int g = 1; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      for (int l = 0; l < 6; ++l) {
        bv(gi, gj, gk, l) = bv(i, j, k, l);
      }
      for (int l = 0; l < param->n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }
      compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
    }
  }
}

template<MixtureModel mix_model> __global__ void apply_inflow(DZone *zone, Inflow *inflow, int i_face,
  DParameter *param, ggxl::VectorField3D<real> *profile_d_ptr, curandState *rng_states_d_ptr,
  ggxl::VectorField3D<real> *fluctuation_dPtr) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const int n_scalar = param->n_scalar;

  real density, u, v, w, p, T, vel;
  real sv_b[MAX_SPEC_NUMBER + 4 + MAX_PASSIVE_SCALAR_NUMBER];

  if (inflow->inflow_type == 1) {
    // Profile inflow
    // For this type of inflow, the profile may specify the values for not only the inflow face, but also the ghost layers.
    // Therefore, all parts including fluctuations and assigning values to ghost layers are done in this function.
    // After all operations, we return directly.

    const auto &prof = profile_d_ptr[inflow->profile_idx];
    int idx[3] = {i, j, k};
    idx[b.face] = 0;
    // idx[b.face] = b.direction == 1 ? 0 : ngg;

    density = prof(idx[0], idx[1], idx[2], 0);
    u = prof(idx[0], idx[1], idx[2], 1);
    v = prof(idx[0], idx[1], idx[2], 2);
    w = prof(idx[0], idx[1], idx[2], 3);
    p = prof(idx[0], idx[1], idx[2], 4);
    T = prof(idx[0], idx[1], idx[2], 5);
    for (int l = 0; l < n_scalar; ++l) {
      sv_b[l] = prof(idx[0], idx[1], idx[2], 6 + l);
    }
    vel = sqrt(u * u + v * v + w * w);

    real bv_fluc_real[6], bv_fluc_imag[6];
    real uf{0}, vf{0}, wf{0};
    if (inflow->fluctuation_type == 1) {
      // White noise fluctuation
      // We assume it obeying a N(0,rms^2) distribution
      // The fluctuation is added to the velocity
      auto y = zone->y(i, j, k);
      // if (y < 4 * inflow->delta_omega && y > -4 * inflow->delta_omega) {
      auto index{0};
      switch (b.face) {
        case 1:
          index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 2:
          index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 0:
        default:
          index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
          break;
      }
      auto &rng_state = rng_states_d_ptr[index];

      real rms = inflow->fluctuation_intensity * u * exp(-y * y);

      uf = curand_normal_double(&rng_state) * rms;
      vf = curand_normal_double(&rng_state) * rms;
      wf = curand_normal_double(&rng_state) * rms;
      u += uf;
      v += vf;
      w += wf;
      vel = sqrt(u * u + v * v + w * w);
      // }
    } else if (inflow->fluctuation_type == 2) {
      // LST fluctuation
      int idx_fluc[3]{i, j, k};
      idx_fluc[b.face] = 0;
      const auto &fluc_info = fluctuation_dPtr[inflow->fluc_prof_idx];

      // rho u v w p t
      bv_fluc_real[0] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 0);
      bv_fluc_real[1] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 1);
      bv_fluc_real[2] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 2);
      bv_fluc_real[3] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 3);
      bv_fluc_real[4] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 4);
      bv_fluc_real[5] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 5);
      bv_fluc_imag[0] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 6);
      bv_fluc_imag[1] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 7);
      bv_fluc_imag[2] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 8);
      bv_fluc_imag[3] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 9);
      bv_fluc_imag[4] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 10);
      bv_fluc_imag[5] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 11);

      real x = zone->x(i, j, k), z = zone->z(i, j, k);

      real A0 = inflow->fluctuation_intensity;
      real omega = 2.0 * pi * inflow->fluctuation_frequency;
      real alpha = 2.0 * pi / inflow->streamwise_wavelength;
      real beta = 2.0 * pi / inflow->spanwise_wavelength;
      real t = param->physical_time;
      real phi = alpha * x - omega * t;
      density += A0 * (bv_fluc_real[0] * cos(phi) - bv_fluc_imag[0] * sin(phi)) * cos(beta * z) * param->rho_ref;
      u += A0 * (bv_fluc_real[1] * cos(phi) - bv_fluc_imag[1] * sin(phi)) * cos(beta * z) * param->v_ref;
      v += A0 * (bv_fluc_real[2] * cos(phi) - bv_fluc_imag[2] * sin(phi)) * cos(beta * z) * param->v_ref;
      w += A0 * (bv_fluc_real[3] * cos(phi) - bv_fluc_imag[3] * sin(phi)) * cos(beta * z) * param->v_ref;
      T += A0 * (bv_fluc_real[5] * cos(phi) - bv_fluc_imag[5] * sin(phi)) * cos(beta * z) * param->T_ref;
      p = density * R_u / mw_air * T;
    }

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = p;
    bv(i, j, k, 5) = T;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv_b[l];
    }
    compute_cv_from_bv_1_point<mix_model>(zone, param, i, j, k);

    // For ghost grids
    for (int g = 1; g <= ngg; g++) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      idx[0] = gi, idx[1] = gj, idx[2] = gk;
      idx[b.face] = -g;
      // idx[b.face] = b.direction == 1 ? g : ngg - g;

      density = prof(idx[0], idx[1], idx[2], 0);
      u = prof(idx[0], idx[1], idx[2], 1) + uf;
      v = prof(idx[0], idx[1], idx[2], 2) + vf;
      w = prof(idx[0], idx[1], idx[2], 3) + wf;
      p = prof(idx[0], idx[1], idx[2], 4);
      T = prof(idx[0], idx[1], idx[2], 5);
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = prof(idx[0], idx[1], idx[2], 6 + l);
      }
      vel = sqrt(u * u + v * v + w * w);

      if (inflow->fluctuation_type == 2) {
        // LST fluctuation
        // int idx_fluc[3]{i, j, k};
        // idx_fluc[b.face] = 0;

        real x = zone->x(gi, gj, gk), z = zone->z(gi, gj, gk);

        real A0 = inflow->fluctuation_intensity;
        real omega = 2.0 * pi * inflow->fluctuation_frequency;
        real alpha = 2.0 * pi / inflow->streamwise_wavelength;
        real beta = 2.0 * pi / inflow->spanwise_wavelength;
        real t = param->physical_time;
        real phi = alpha * x - omega * t;
        density += A0 * (bv_fluc_real[0] * cos(phi) - bv_fluc_imag[0] * sin(phi)) * cos(beta * z) * param->rho_ref;
        u += A0 * (bv_fluc_real[1] * cos(phi) - bv_fluc_imag[1] * sin(phi)) * cos(beta * z) * param->v_ref;
        v += A0 * (bv_fluc_real[2] * cos(phi) - bv_fluc_imag[2] * sin(phi)) * cos(beta * z) * param->v_ref;
        w += A0 * (bv_fluc_real[3] * cos(phi) - bv_fluc_imag[3] * sin(phi)) * cos(beta * z) * param->v_ref;
        T += A0 * (bv_fluc_real[5] * cos(phi) - bv_fluc_imag[5] * sin(phi)) * cos(beta * z) * param->T_ref;
        p = density * R_u / mw_air * T;
      }

      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = p;
      bv(gi, gj, gk, 5) = T;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv_b[l];
      }
      compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
    }
    return;
  }

  if (inflow->inflow_type == 2) {
    // Mixing layer inflow
    const real u_upper = inflow->u, u_lower = inflow->u_lower;
    auto y = zone->y(i, j, k);
    u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / inflow->delta_omega);
    if (y >= 0) {
      density = inflow->density;
      v = inflow->v;
      w = inflow->w;
      p = inflow->pressure;
      T = inflow->temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = inflow->sv[l];
      }
      vel = sqrt(u * u + v * v + w * w);
    } else {
      // The lower stream
      density = inflow->density_lower;
      v = inflow->v_lower;
      w = inflow->w_lower;
      p = inflow->p_lower;
      T = inflow->t_lower;
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = inflow->sv_lower[l];
      }
      vel = sqrt(u * u + v * v + w * w);
    }

    if (inflow->fluctuation_type == 1) {
      // White noise fluctuation
      // We assume it obeying a N(0,rms^2) distribution
      // The fluctuation is added to the velocity
      // Besides, we assume the range of fluctuation is restricted to 4*delta_omega ranges.
      // if (y < 4 * inflow->delta_omega && y > -4 * inflow->delta_omega) {
      auto index{0};
      switch (b.face) {
        case 1:
          index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 2:
          index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 0:
        default:
          index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
          break;
      }
      auto &rng_state = rng_states_d_ptr[index];

      real rms = inflow->fluctuation_intensity * u * exp(-y * y);

      u += curand_normal_double(&rng_state) * rms;
      v += curand_normal_double(&rng_state) * rms;
      w += curand_normal_double(&rng_state) * rms;
      vel = sqrt(u * u + v * v + w * w);
      // }
    }
  } else {
    // Constant inflow
    if (inflow->inflow_sub) {
      real nx{zone->metric(i, j, k, b.face * 3)},
          ny{zone->metric(i, j, k, b.face * 3 + 1)},
          nz{zone->metric(i, j, k, b.face * 3 + 2)};
      const real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
      nx *= grad_n_inv;
      ny *= grad_n_inv;
      nz *= grad_n_inv;

      real c0{0};
      if constexpr (mix_model != MixtureModel::Air) {
        c0 = zone->acoustic_speed(i, j, k);
      } else {
        c0 = sqrt(gamma_air * R_air * bv(i, j, k, 5));
      }
      if (const real Mn = (bv(i, j, k, 1) * nx + bv(i, j, k, 2) * ny + bv(i, j, k, 3) * nz) / c0; Mn < -1.0) {
        density = inflow->density;
        u = inflow->u;
        v = inflow->v;
        w = inflow->w;
        p = inflow->pressure;
        T = inflow->temperature;
        for (int l = 0; l < n_scalar; ++l) {
          sv_b[l] = inflow->sv[l];
        }
      } else {
        // subsonic inflow
        int i1 = i - dir[0], j1 = j - dir[1], k1 = k - dir[2];
        real c1{0};
        if constexpr (mix_model != MixtureModel::Air) {
          c1 = zone->acoustic_speed(i1, j1, k1);
        } else {
          c1 = sqrt(gamma_air * R_air * bv(i1, j1, k1, 5));
        }
        real rho1 = bv(i1, j1, k1, 0), p1 = bv(i1, j1, k1, 4);
        p = 0.5 * (p1 + inflow->pressure - rho1 * c1 * ((inflow->u - bv(i1, j1, k1, 1)) * nx +
                                                        (inflow->v - bv(i1, j1, k1, 2)) * ny +
                                                        (inflow->w - bv(i1, j1, k1, 3)) * nz));
        density = inflow->density + (p - inflow->pressure) / (c1 * c1);
        u = inflow->u - (p - inflow->pressure) / (rho1 * c1) * nx;
        v = inflow->v - (p - inflow->pressure) / (rho1 * c1) * ny;
        w = inflow->w - (p - inflow->pressure) / (rho1 * c1) * nz;
        for (int l = 0; l < n_scalar; ++l) {
          sv_b[l] = inflow->sv[l];
        }
        // compute T
        if constexpr (mix_model != MixtureModel::Air) {
          T = p * inflow->mw / (density * R_u);
        } else {
          T = p / (density * R_air);
        }
      }
    } else {
      density = inflow->density;
      u = inflow->u;
      v = inflow->v;
      w = inflow->w;
      p = inflow->pressure;
      T = inflow->temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = inflow->sv[l];
      }

      vel = inflow->velocity;

      if (inflow->fluctuation_type == 1) {
        // White noise fluctuation
        // We assume it obeying a N(0,rms^2) distribution
        // The fluctuation is added to the velocity
        auto index{0};
        switch (b.face) {
          case 1:
            index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 2:
            index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 0:
          default:
            index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
            break;
        }
        auto &rng_state = rng_states_d_ptr[index];

        real rms = inflow->fluctuation_intensity;

        u += curand_normal_double(&rng_state) * rms * vel;
        vel = sqrt(u * u + v * v + w * w);
      } else if (inflow->fluctuation_type == 3) {
        //S mode waves with wide band frequencies and spanwise wavenumbers are induced
        real t = param->physical_time;

        real x = zone->x(i, j, k);
        for (int m = 0; m <= 198; ++m) {
          constexpr real delta_f = 5000;
          real f = delta_f * m + 10000;
          real alpha = 2.0 * pi * f / (1.0 - 1.0 / param->mach_ref) / param->v_ref;
          real Am = 0;
          if (f <= 40000) {
            constexpr real CL = 3.953e-4;
            Am = sqrt(CL / f * delta_f * 0.5) * param->p_ref;
          } else if (f > 40000) {
            constexpr real CU = 126.5e6;
            Am = sqrt(CU / pow(f, 3.5) * delta_f * 0.5) * param->p_ref;
          }
          const real disturbance = Am * cos(alpha * x - 2.0 * pi * f * t + inflow->random_phase[m]);
          p += disturbance;
          u -= disturbance * param->mach_ref / param->rho_ref / param->v_ref;
          v += 0;
          w += 0;
          T += disturbance * (gamma_air - 1.0) * param->mach_ref * param->mach_ref / param->rho_ref / param->v_ref /
              param->v_ref * param->T_ref;
          density = p * mw_air / (R_u * T);
        }
      }
    }
  }

  // Specify the boundary value as given.
  bv(i, j, k, 0) = density;
  bv(i, j, k, 1) = u;
  bv(i, j, k, 2) = v;
  bv(i, j, k, 3) = w;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = T;
  for (int l = 0; l < n_scalar; ++l) {
    sv(i, j, k, l) = sv_b[l];
  }
  compute_cv_from_bv_1_point<mix_model>(zone, param, i, j, k);
  for (int g = 1; g <= ngg; g++) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (int l = 0; l < n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv_b[l];
    }

    if (inflow->inflow_sub) {
      const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

      real rho2 = 2 * density - bv(ii, ij, ik, 0);
      real p2 = 2 * p - bv(ii, ij, ik, 4);

      if (p2 <= 0.1 * p) p2 = 0.1 * p;                 // avoid negative pressure
      if (rho2 <= 0.1 * density) rho2 = 0.1 * density; // avoid negative density

      bv(gi, gj, gk, 0) = rho2;
      bv(gi, gj, gk, 1) = 2 * u - bv(ii, ij, ik, 1);;
      bv(gi, gj, gk, 2) = 2 * v - bv(ii, ij, ik, 2);
      bv(gi, gj, gk, 3) = 2 * w - bv(ii, ij, ik, 3);
      bv(gi, gj, gk, 4) = p2;
      if constexpr (mix_model != MixtureModel::Air) {
        bv(gi, gj, gk, 5) = p2 * inflow->mw / (rho2 * R_u);
      } else {
        bv(gi, gj, gk, 5) = p2 / (rho2 * R_air);
      }
    } else {
      if (inflow->fluctuation_type == 3) {
        //S mode waves with wide band frequencies and spanwise wavenumbers are induced
        real t = param->physical_time;

        real x = zone->x(gi, gj, gk);
        for (int m = 0; m <= 198; ++m) {
          constexpr real delta_f = 5000;
          real f = delta_f * m + 10000;
          real alpha = 2.0 * pi * f / (1.0 - 1.0 / param->mach_ref) / param->v_ref;
          real Am = 0;
          if (f <= 40000) {
            constexpr real CL = 3.953e-4;
            Am = sqrt(CL / f * delta_f * 0.5) * param->p_ref;
          } else if (f > 40000) {
            constexpr real CU = 126.5e6;
            Am = sqrt(CU / pow(f, 3.5) * delta_f * 0.5) * param->p_ref;
          }
          const real disturbance = Am * cos(alpha * x - 2.0 * pi * f * t + inflow->random_phase[m]);
          p = inflow->pressure + disturbance;
          u = inflow->u - disturbance * param->mach_ref / param->rho_ref / param->v_ref;
          v = inflow->v;
          w = inflow->w;
          T = inflow->temperature +
              disturbance * (gamma_air - 1.0) * param->mach_ref * param->mach_ref / param->rho_ref / param->v_ref /
              param->v_ref * param->T_ref;
          density = p * mw_air / (R_u * T);
        }
      }

      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = p;
      bv(gi, gj, gk, 5) = T;
    }

    compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model> __global__ void apply_inflow_df(DZone *zone, Inflow *inflow, DParameter *param,
  ggxl::VectorField3D<real> *fluctuation_dPtr, int df_iFace) {
  const int ngg = zone->ngg;
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= zone->my + ngg || k >= zone->mz + ngg) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  constexpr int i = 0;
  const real u_upper = inflow->u, u_lower = inflow->u_lower;
  const auto y = zone->y(i, j, k);
  real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / inflow->delta_omega);
  const real u_fluc = fluctuation_dPtr[df_iFace](0, j, k, 0) * param->delta_u;
  const real v_fluc = fluctuation_dPtr[df_iFace](0, j, k, 1) * param->delta_u;
  const real w_fluc = fluctuation_dPtr[df_iFace](0, j, k, 2) * param->delta_u;
  const real mw = y > 0 ? inflow->mw : inflow->mw_lower;
  real T = y > 0 ? inflow->temperature : inflow->t_lower;
  real rho = y > 0 ? inflow->density : inflow->density_lower;
  real p = y > 0 ? inflow->pressure : inflow->p_lower;
  real c_p = 0;
  if constexpr (mix_model != MixtureModel::Air) {
    const real *sv_b = y > 0 ? inflow->sv : inflow->sv_lower;
    real cpl[MAX_SPEC_NUMBER];
    compute_cp(T, cpl, param);
    for (int l = 0; l < param->n_spec; ++l) {
      c_p += cpl[l] * sv_b[l];
    }
  } else {
    c_p = gamma_air * R_u / mw_air / (gamma_air - 1);
  }
  real t_fluc = -u_fluc * u / c_p; // SRA
  t_fluc *= 0.25;                  // StreamS multiplies a parameter "dftscaling=0.25" here
  const real rho_fluc = -t_fluc * rho / T;
  u += u_fluc;
  rho += rho_fluc;
  T += t_fluc;
  p = rho * R_u / mw * T;
  for (int ig = 0; ig <= ngg; ++ig) {
    const int gi{-ig};
    bv(gi, j, k, 0) = rho;
    bv(gi, j, k, 1) = u;
    bv(gi, j, k, 2) = v_fluc;
    bv(gi, j, k, 3) = w_fluc;
    bv(gi, j, k, 4) = p;
    bv(gi, j, k, 5) = T;
    const real *sv_b = y > 0 ? inflow->sv : inflow->sv_lower;
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, j, k, l) = sv_b[l];
    }
    compute_cv_from_bv_1_point<mix_model>(zone, param, gi, j, k);
  }
}

template<MixtureModel mix_model> __global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, int i_face,
  int step = -1) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  int jet_label = -1;
  if (param->problem_type == 2) {
    // jicf problem
    const auto x = zone->x(i, j, k), z = zone->z(i, j, k);
    const auto r = param->jet_radius;
    for (int i_jet = 0; i_jet < param->n_jet; i_jet++) {
      const real xc = param->xc_jet[i_jet];
      const real zc = param->zc_jet[i_jet];
      if (hypot(x - xc, z - zc) <= r) {
        jet_label = i_jet;
        break;
      }
    }
  }
  if (jet_label >= 0) {
    // in jet region, the values are assigned from the jet
    const auto rho_jet = param->jet_rho[jet_label], u_jet = param->jet_u[jet_label],
        v_jet = param->jet_v[jet_label], w_jet = param->jet_w[jet_label];
    const auto p_jet = param->jet_p[jet_label], t_jet = param->jet_T[jet_label];
    bv(i, j, k, 0) = rho_jet;
    bv(i, j, k, 1) = u_jet;
    bv(i, j, k, 2) = v_jet;
    bv(i, j, k, 3) = w_jet;
    bv(i, j, k, 4) = p_jet;
    bv(i, j, k, 5) = t_jet;
    for (int l = 0; l < param->n_spec; l++) {
      sv(i, j, k, l) = param->jet_sv(jet_label, l);
    }
    compute_cv_from_bv_1_point<mix_model>(zone, param, i, j, k);

    // For ghost grids
    for (int g = 1; g <= ngg; ++g) {
      const int i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

      bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_jet;
      bv(i_gh[0], i_gh[1], i_gh[2], 1) = u_jet;
      bv(i_gh[0], i_gh[1], i_gh[2], 2) = v_jet;
      bv(i_gh[0], i_gh[1], i_gh[2], 3) = w_jet;
      bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_jet;
      bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_jet;
      for (int l = 0; l < param->n_spec; l++) {
        sv(i_gh[0], i_gh[1], i_gh[2], l) = param->jet_sv(jet_label, l);
      }
      compute_cv_from_bv_1_point<mix_model>(zone, param, i_gh[0], i_gh[1], i_gh[2]);
    }
    return;
  }

  real t_wall{bv(i, j, k, 5)};

  const int idx[]{i - dir[0], j - dir[1], k - dir[2]};
  if (wall->thermal_type == Wall::ThermalType::isothermal) {
    t_wall = wall->temperature;
  } else if (wall->thermal_type == Wall::ThermalType::adiabatic) {
    t_wall = bv(idx[0], idx[1], idx[2], 5);
  } else if (wall->thermal_type == Wall::ThermalType::equilibrium_radiation) {
    const real t_in{bv(idx[0], idx[1], idx[2], 5)};

    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    const real lambda = Sutherland(t_wall) * cp / param->Pr;

    const real heat_flux{lambda * (t_in - t_wall) / zone->wall_distance(idx[0], idx[1], idx[2])};
    if (heat_flux > 0) {
      const real coeff =
          stefan_boltzmann_constants * wall->emissivity * zone->wall_distance(idx[0], idx[1], idx[2]) * t_wall *
          t_wall * t_wall;
      t_wall -= (coeff * t_wall + lambda * t_wall - lambda * t_in) / (4 * coeff + lambda);
    } else {
      t_wall = wall->temperature;
    }
    if (t_wall < wall->temperature) {
      t_wall = wall->temperature;
    }
  }
  const real p{bv(idx[0], idx[1], idx[2], 4)};

  real R{R_air};
  if constexpr (mix_model != MixtureModel::Air) {
    // Mixture
    const auto Rk = param->gas_const;
    if (wall->catalytic_type == 0) {
      // Non-catalytic
      R = 0;
      for (int l = 0; l < param->n_spec; ++l) {
        sv(i, j, k, l) = sv(idx[0], idx[1], idx[2], l);
        R += sv(i, j, k, l) * Rk[l];
      }
    } else {
      R = 0;
      for (int l = 0; l < param->n_spec; ++l) {
        sv(i, j, k, l) = wall->yk[l];
        R += sv(i, j, k, l) * Rk[l];
      }
    }
  }

  const real rho_wall = p / (t_wall * R);
  bv(i, j, k, 0) = rho_wall;
  bv(i, j, k, 1) = 0;
  bv(i, j, k, 2) = 0;
  bv(i, j, k, 3) = 0;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = t_wall;

  if (wall->if_blow_shock_wave && step >= 0 && step <= 50) {
    gxl::Matrix<real, 3, 3, 1> bdJin;
    real d1 = zone->metric(i, j, k, 3);
    real d2 = zone->metric(i, j, k, 4);
    real d3 = zone->metric(i, j, k, 5);
    real kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(1, 1) = d1 / kk;
    bdJin(1, 2) = d2 / kk;
    bdJin(1, 3) = d3 / kk;

    d1 = bdJin(1, 2) - bdJin(1, 3);
    d2 = bdJin(1, 3) - bdJin(1, 1);
    d3 = bdJin(1, 1) - bdJin(1, 2);
    kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(2, 1) = d1 / kk;
    bdJin(2, 2) = d2 / kk;
    bdJin(2, 3) = d3 / kk;

    d1 = bdJin(1, 2) * bdJin(2, 3) - bdJin(1, 3) * bdJin(2, 2);
    d2 = bdJin(1, 3) * bdJin(2, 1) - bdJin(1, 1) * bdJin(2, 3);
    d3 = bdJin(1, 1) * bdJin(2, 2) - bdJin(1, 2) * bdJin(2, 1);
    kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(3, 1) = d1 / kk;
    bdJin(3, 2) = d2 / kk;
    bdJin(3, 3) = d3 / kk;

    real u = bv(i - dir[0], j - dir[1], k - dir[2], 1),
        v = bv(i - dir[0], j - dir[1], k - dir[2], 2),
        w = bv(i - dir[0], j - dir[1], k - dir[2], 3);
    real vn = bdJin(1, 1) * u + bdJin(1, 2) * v + bdJin(1, 3) * w;
    real vt = bdJin(2, 1) * u + bdJin(2, 2) * v + bdJin(2, 3) * w;
    real vs = bdJin(3, 1) * u + bdJin(3, 2) * v + bdJin(3, 3) * w;

    bv(i, j, k, 1) = bdJin(2, 1) * vt + bdJin(3, 1) * vs - bdJin(1, 1) * vn;
    bv(i, j, k, 2) = bdJin(2, 2) * vt + bdJin(3, 2) * vs - bdJin(1, 2) * vn;
    bv(i, j, k, 3) = bdJin(2, 3) * vt + bdJin(3, 3) * vs - bdJin(1, 3) * vn;
  }

  real v_blow{0};
  int if_fluctuation = wall->fluctuation_type;
  if (if_fluctuation == 1) {
    // Pirozzoli & Li fluctuations
    real phil[10] = {0.03, 0.47, 0.43, 0.73, 0.86, 0.36, 0.96, 0.47, 0.36, 0.61};
    real phim[5] = {0.31, 0.05, 0.03, 0.72, 0.93};
    real A0 = wall->fluctuation_intensity;
    real x0 = wall->fluctuation_x0;
    real x1 = wall->fluctuation_x1;
    real t = param->physical_time;

    real x = zone->x(i, j, k), z = zone->z(i, j, k);
    real zmax = abs(zone->z(0, 0, zone->mz - 1) - zone->z(0, 0, 0));
    real theta = 2 * pi * (x - x0) / (x1 - x0);
    real fx = 4.0 * sin(theta) * (1.0 - cos(theta)) / sqrt(27.0);

    real gz = 0;
    for (int l = 0; l < 10; ++l) {
      gz += wall->Zl[l] * sin(2.0 * pi * (l + 1) * (z / zmax + phil[l]));
    }
    real ht = 0;
    for (int m = 0; m < 5; ++m) {
      ht += wall->Tm[m] * sin(wall->fluctuation_frequency * t + 2.0 * pi * phim[m]);
    }
    if (x > x0 && x < x1) {
      v_blow = A0 * param->v_ref * fx * gz * ht;
      bv(i, j, k, 2) = A0 * param->v_ref * fx * gz * ht;
    }
  } else if (if_fluctuation == 3) {
    real A0 = wall->fluctuation_intensity;
    real omega = 2.0 * pi * wall->fluctuation_frequency;
    real beta = 2.0 * pi / wall->spanwise_wavelength;
    real x0 = wall->fluctuation_x0;
    real x1 = wall->fluctuation_x1;
    real t = param->physical_time;

    real x_middle = (x0 + x1) * 0.5;
    real xi = 0;
    real z = zone->z(i, j, k);
    if (real x = zone->x(i, j, k); x >= x0 && x <= x_middle) {
      xi = (x - x0) / (x_middle - x0);
      const real xi3 = xi * xi * xi;
      bv(i, j, k, 2) = A0 * (15.1875 * xi3 * xi * xi - 35.4375 * xi3 * xi + 20.25 * xi3) * cos(beta * z) *
                       sin(omega * t) / param->v_ref;
    } else if (x >= x_middle && x <= x1) {
      xi = (x1 - x) / (x1 - x_middle);
      const real xi3 = xi * xi * xi;
      bv(i, j, k, 2) = -A0 * (15.1875 * xi3 * xi * xi - 35.4375 * xi3 * xi + 20.25 * xi3) * cos(beta * z) *
                       sin(omega * t) / param->v_ref;
    }
  }

  if (param->n_ps > 0) {
    const int i_ps{param->i_ps};
    for (int l = 0; l < param->n_ps; l++) {
      sv(i, j, k, i_ps + l) = sv(idx[0], idx[1], idx[2], i_ps + l);
    }
  }

  compute_cv_from_bv_1_point<mix_model>(zone, param, i, j, k);

  for (int g = 1; g <= ngg; ++g) {
    const int i_in[]{i - g * dir[0], j - g * dir[1], k - g * dir[2]};
    const int i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

    const real p_i{bv(i_in[0], i_in[1], i_in[2], 4)};
    const real t_i{bv(i_in[0], i_in[1], i_in[2], 5)};

    double t_g{t_i};
    if (wall->thermal_type == Wall::ThermalType::isothermal ||
        wall->thermal_type == Wall::ThermalType::equilibrium_radiation) {
      t_g = 2 * t_wall - t_i;    // 0.5*(t_i+t_g)=t_w
      if (t_g <= 0.1 * t_wall) { // improve stability
        t_g = t_wall;
      }
    }

    if constexpr (mix_model != MixtureModel::Air) {
      const auto Rk = param->gas_const;
      if (wall->catalytic_type == 0) {
        // Non-catalytic
        R = 0;
        for (int l = 0; l < param->n_spec; ++l) {
          // The mass fraction is given by a symmetry condition, is this reasonable?
          sv(i_gh[0], i_gh[1], i_gh[2], l) = sv(i_in[0], i_in[1], i_in[2], l);
          R += sv(i_in[0], i_in[1], i_in[2], l) * Rk[l];
        }
      } else {
        R = 0;
        for (int l = 0; l < param->n_spec; ++l) {
          sv(i_gh[0], i_gh[1], i_gh[2], l) = wall->yk[l];
          R += sv(i, j, k, l) * Rk[l];
        }
      }
    }

    const real rho_g{p_i / (t_g * R)};
    bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    bv(i_gh[0], i_gh[1], i_gh[2], 1) = -bv(i_in[0], i_in[1], i_in[2], 1);
    bv(i_gh[0], i_gh[1], i_gh[2], 2) = v_blow * 2 - bv(i_in[0], i_in[1], i_in[2], 2);
    bv(i_gh[0], i_gh[1], i_gh[2], 3) = -bv(i_in[0], i_in[1], i_in[2], 3);
    bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_g;

    if (param->n_ps > 0) {
      const int i_ps{param->i_ps};
      for (int l = 0; l < param->n_ps; l++) {
        sv(i_gh[0], i_gh[1], i_gh[2], i_ps + l) = sv(i_in[0], i_in[1], i_in[2], i_ps + l);
      }
    }

    compute_cv_from_bv_1_point<mix_model>(zone, param, i_gh[0], i_gh[1], i_gh[2]);
  }
}

template<MixtureModel mix_model> __global__ void apply_periodic(DZone *zone, DParameter *param, int i_face) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  int idx_other[3]{i, j, k};
  switch (b.face) {
    case 0: // i face
      idx_other[0] = b.direction < 0 ? zone->mx - 1 : 0;
      break;
    case 1: // j face
      idx_other[1] = b.direction < 0 ? zone->my - 1 : 0;
      break;
    case 2: // k face
    default:
      idx_other[2] = b.direction < 0 ? zone->mz - 1 : 0;
      break;
  }

  for (int l = 0; l < 6; ++l) {
    const auto ave = 0.5 * (bv(i, j, k, l) + bv(idx_other[0], idx_other[1], idx_other[2], l));
    bv(i, j, k, l) = ave;
    bv(idx_other[0], idx_other[1], idx_other[2], l) = ave;
  }
  for (int l = 0; l < param->n_scalar; ++l) {
    const auto ave = 0.5 * (sv(i, j, k, l) + sv(idx_other[0], idx_other[1], idx_other[2], l));
    sv(i, j, k, l) = ave;
    sv(idx_other[0], idx_other[1], idx_other[2], l) = ave;
  }
  compute_cv_from_bv_1_point<mix_model>(zone, param, i, j, k);
  for (int g = 1; g <= ngg; ++g) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const int ii{idx_other[0] + g * dir[0]}, ij{idx_other[1] + g * dir[1]}, ik{idx_other[2] + g * dir[2]};
    for (int l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(ii, ij, ik, l);
    }
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }
    compute_cv_from_bv_1_point<mix_model>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model> __global__ void apply_outflow_nr(DZone *zone, int i_face, const DParameter *param) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;

  const int face = b.face;
  real kx{zone->metric(i, j, k, face * 3)}, ky{zone->metric(i, j, k, face * 3 + 1)},
      kz{zone->metric(i, j, k, face * 3 + 2)};
  const real gradKInv = 1 / norm3d(kx, ky, kz);

  const real u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3);
  const real U = u * kx + v * ky + w * kz;
  const real a = sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0));
  const real U_face = U * gradKInv;
  if (abs(U_face / a) > 1) {
    // the supersonic outflow does not specify any information
    return;
  }

  kx *= gradKInv;
  ky *= gradKInv;
  kz *= gradKInv;

  // Let us temporarily assume the discretization is 4th-order.
  constexpr real cc[5] = {-25 / 12.0, 48 / 12.0, -36 / 12.0, 16 / 12.0, -3 / 12.0};
  int sgn = 1;
  const bool start_face = b.direction == -1;
  if (!start_face) {
    // The big number face
    sgn = -1;
  }

  real rho_k{0}, u_k{0}, v_k{0}, w_k{0}, p_k{0};
  if (face == 0) {
    // x direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i + sgn * l, j, k, 0);
      u_k += c * bv(i + sgn * l, j, k, 1);
      v_k += c * bv(i + sgn * l, j, k, 2);
      w_k += c * bv(i + sgn * l, j, k, 3);
      p_k += c * bv(i + sgn * l, j, k, 4);
    }
  } else if (face == 1) {
    // y direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i, j + sgn * l, k, 0);
      u_k += c * bv(i, j + sgn * l, k, 1);
      v_k += c * bv(i, j + sgn * l, k, 2);
      w_k += c * bv(i, j + sgn * l, k, 3);
      p_k += c * bv(i, j + sgn * l, k, 4);
    }
  } else if (face == 2) {
    // z direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i, j, k + sgn * l, 0);
      u_k += c * bv(i, j, k + sgn * l, 1);
      v_k += c * bv(i, j, k + sgn * l, 2);
      w_k += c * bv(i, j, k + sgn * l, 3);
      p_k += c * bv(i, j, k + sgn * l, 4);
    }
  }

  const real aGradK = a / gradKInv;
  real Lambda[5]{U - aGradK, U, U, U, U + aGradK};

  // if (param->L1_wave == 0) // zero-reflection
  if (start_face) {
    Lambda[0] = min(Lambda[0], 0.0);
    Lambda[1] = min(Lambda[1], 0.0);
    Lambda[2] = min(Lambda[2], 0.0);
    Lambda[3] = min(Lambda[3], 0.0);
    Lambda[4] = min(Lambda[4], 0.0);
  } else {
    Lambda[0] = max(Lambda[0], 0.0);
    Lambda[1] = max(Lambda[1], 0.0);
    Lambda[2] = max(Lambda[2], 0.0);
    Lambda[3] = max(Lambda[3], 0.0);
    Lambda[4] = max(Lambda[4], 0.0);
  }

  real L[5]{};
  const real a2 = a * a, rho = bv(i, j, k, 0);

  L[0] = Lambda[0] * (-rho * a * (kx * u_k + ky * v_k + kz * w_k) + p_k);
  L[1] = Lambda[1] * (a2 * rho_k - p_k);
  L[4] = Lambda[4] * (rho * a * (kx * u_k + ky * v_k + kz * w_k) + p_k);
  if (face == 0) {
    L[2] = Lambda[2] * (-ky * u_k + kx * v_k);
    L[3] = Lambda[3] * (-kz * u_k + kx * w_k);
  } else if (face == 1) {
    L[2] = Lambda[2] * (-kx * v_k + ky * u_k);
    L[3] = Lambda[3] * (-kz * v_k + ky * w_k);
  } else if (face == 2) {
    L[2] = Lambda[2] * (-kx * w_k + kz * u_k);
    L[3] = Lambda[3] * (-ky * w_k + kz * v_k);
  }

  real d[5]{};
  d[0] = 1 / a2 * (L[1] + 0.5 * (L[0] + L[4]));
  d[4] = 0.5 * (L[0] + L[4]);
  if (face == 0) {
    d[1] = kx * (L[4] - L[0]) / (2 * rho * a) - (ky * L[2] + kz * L[3]) * gradKInv;
    d[2] = ky * (L[4] - L[0]) / (2 * rho * a) + ((kx * kx + kz * kz) * L[2] - ky * kz * L[3]) * gradKInv / kx;
    d[3] = kz * (L[4] - L[0]) / (2 * rho * a) + ((kx * kx + ky * ky) * L[3] - ky * kz * L[2]) * gradKInv / kx;
  } else if (face == 1) {
    d[1] = kx * (L[4] - L[0]) / (2 * rho * a) + ((ky * ky + kz * kz) * L[2] - kx * kz * L[3]) * gradKInv / ky;
    d[2] = ky * (L[4] - L[0]) / (2 * rho * a) - (kx * L[2] + kz * L[3]) * gradKInv;
    d[3] = kz * (L[4] - L[0]) / (2 * rho * a) + ((kx * kx + ky * ky) * L[3] - kx * kz * L[2]) * gradKInv / ky;
  } else if (face == 2) {
    d[1] = kx * (L[4] - L[0]) / (2 * rho * a) + ((ky * ky + kz * kz) * L[2] - kx * ky * L[3]) * gradKInv / kz;
    d[2] = ky * (L[4] - L[0]) / (2 * rho * a) + ((kx * kx + kz * kz) * L[3] - kx * ky * L[2]) * gradKInv / kz;
    d[3] = kz * (L[4] - L[0]) / (2 * rho * a) - (kx * L[2] + ky * L[3]) * gradKInv;
  }

  const real minusJacInv = -zone->jac(i, j, k);
  auto &rhs = zone->dq;
  rhs(i, j, k, 0) += minusJacInv * d[0];
  rhs(i, j, k, 1) += minusJacInv * (u * d[0] + rho * d[1]);
  rhs(i, j, k, 2) += minusJacInv * (v * d[0] + rho * d[2]);
  rhs(i, j, k, 3) += minusJacInv * (w * d[0] + rho * d[3]);
  rhs(i, j, k, 4) += minusJacInv * (0.5 * (u * u + v * v + w * w) * d[0] + rho * u * d[1] + rho * v * d[2] + rho * w *
                                    d[3] + d[4] / (gamma_air - 1));
  // rhs(i, j, k, 0) = minusJacInv * d[0];
  // rhs(i, j, k, 1) = minusJacInv * (u * d[0] + rho * d[1]);
  // rhs(i, j, k, 2) = minusJacInv * (v * d[0] + rho * d[2]);
  // rhs(i, j, k, 3) = minusJacInv * (w * d[0] + rho * d[3]);
  // rhs(i, j, k, 4) = minusJacInv * (0.5 * (u * u + v * v + w * w) * d[0] + rho * u * d[1] + rho * v * d[2] + rho * w *
  //                                  d[3] + d[4] / (gamma_air - 1));
}

template<MixtureModel mix_model> __global__ void apply_outflow_nr_conserv(DZone *zone, int i_face,
  const DParameter *param) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &cv = zone->cv;

  // Debug, we only apply the outflow condition to the i direction
  // if (b.face != 0)
  //   return;

  const int face = b.face;
  // Let us temporarily assume the discretization is 4th-order.
  // constexpr real cc[5] = {-25 / 12.0, 48 / 12.0, -36 / 12.0, 16 / 12.0, -3 / 12.0};
  constexpr real cc[3] = {-1.5, 2, -0.5};
  int sgn = 1;
  const bool start_face = b.direction == -1;
  if (!start_face) {
    // The big number face
    sgn = -1;
  }
  // Compute the spatial gradient in the normal direction
  real dR_dn{0}, dRU_dn{0}, dRV_dn{0}, dRW_dn{0}, dRE_dn{0};
  real dRY_dn[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER]{};
  if (face == 0) {
    // x direction
    for (int l = 0; l <= 2; ++l) {
      const real c = sgn * cc[l];
      dR_dn += c * cv(i + sgn * l, j, k, 0);
      dRU_dn += c * cv(i + sgn * l, j, k, 1);
      dRV_dn += c * cv(i + sgn * l, j, k, 2);
      dRW_dn += c * cv(i + sgn * l, j, k, 3);
      dRE_dn += c * cv(i + sgn * l, j, k, 4);
      for (int l_scalar = 0; l_scalar < param->n_scalar; ++l_scalar) {
        dRY_dn[l_scalar] += c * cv(i + sgn * l, j, k, 5 + l_scalar);
      }
    }
  } else if (face == 1) {
    // y direction
    for (int l = 0; l <= 2; ++l) {
      const real c = sgn * cc[l];
      dR_dn += c * cv(i, j + sgn * l, k, 0);
      dRU_dn += c * cv(i, j + sgn * l, k, 1);
      dRV_dn += c * cv(i, j + sgn * l, k, 2);
      dRW_dn += c * cv(i, j + sgn * l, k, 3);
      dRE_dn += c * cv(i, j + sgn * l, k, 4);
      for (int l_scalar = 0; l_scalar < param->n_scalar; ++l_scalar) {
        dRY_dn[l_scalar] += c * cv(i, j + sgn * l, k, 5 + l_scalar);
      }
    }
  } else if (face == 2) {
    // z direction
    for (int l = 0; l <= 2; ++l) {
      const real c = sgn * cc[l];
      dR_dn += c * cv(i, j, k + sgn * l, 0);
      dRU_dn += c * cv(i, j, k + sgn * l, 1);
      dRV_dn += c * cv(i, j, k + sgn * l, 2);
      dRW_dn += c * cv(i, j, k + sgn * l, 3);
      dRE_dn += c * cv(i, j, k + sgn * l, 4);
      for (int l_scalar = 0; l_scalar < param->n_scalar; ++l_scalar) {
        dRY_dn[l_scalar] += c * cv(i, j, k + sgn * l, 5 + l_scalar);
      }
    }
  }


  // Compute the contravariant speed Uk
  auto &bv = zone->bv;
  const real r = bv(i, j, k, 0), u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3), p = bv(i, j, k, 4);
  real kx{zone->metric(i, j, k, face * 3)}, ky{zone->metric(i, j, k, face * 3 + 1)},
      kz{zone->metric(i, j, k, face * 3 + 2)};
  const real gradKInv = 1 / norm3d(kx, ky, kz);
  kx *= gradKInv;
  ky *= gradKInv;
  kz *= gradKInv;
  const real Uk = u * kx + v * ky + w * kz;
  // Compute the speed of sound
  real gam{gamma_air};
  real h_i[MAX_SPEC_NUMBER];
  const real T{bv(i, j, k, 5)};
  if constexpr (mix_model != MixtureModel::Air) {
    real cp_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(T, h_i, cp_i, param);
    real cp{0}, cv_tot{0};
    const auto &sv = zone->sv;
    for (int l = 0; l < param->n_spec; ++l) {
      cp += sv(i, j, k, l) * cp_i[l];
      cv_tot += sv(i, j, k, l) * (cp_i[l] - param->gas_const[l]);
    }
    gam = cp / cv_tot;
  }
  const real c2 = gam * p / r, c = sqrt(c2), ic2 = 1 / c2;
  const real cGradK = c / gradKInv;
  real Lambda[3]{Uk - cGradK, Uk, Uk + cGradK};
  // Apply the perfect non-reflecting condition. All strengths of incoming waves are set to 0.
  // if (param->L1_wave == 0) // zero-reflection
  if (start_face) {
    Lambda[0] = min(Lambda[0], 0.0);
    Lambda[1] = min(Lambda[1], 0.0);
    Lambda[2] = min(Lambda[2], 0.0);
  } else {
    Lambda[0] = max(Lambda[0], 0.0);
    Lambda[1] = max(Lambda[1], 0.0);
    Lambda[2] = max(Lambda[2], 0.0);
  }

  // Compute the left eigenmatrix EL
  const real ek = 0.5 * (u * u + v * v + w * w);
  const real gm1 = gam - 1;
  const real alpha = gm1 * ek;
  real al_dRYdn{0}, alphaL[MAX_SPEC_NUMBER]{};
  for (int l = 0; l < param->n_spec; ++l) {
    alphaL[l] = gam * param->gas_const[l] * T - gm1 * h_i[l];
    al_dRYdn += alphaL[l] * dRY_dn[l];
  }

  real L[5 + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER]{};
  // Compute the Ls = \Lambda * EL * dQ/d\xi
  L[0] = Lambda[0] * 0.5 * ic2 * ((alpha + Uk * c) * dR_dn
                                  - (gm1 * u + kx * c) * dRU_dn
                                  - (gm1 * v + ky * c) * dRV_dn
                                  - (gm1 * w + kz * c) * dRW_dn
                                  + gm1 * dRE_dn
                                  + al_dRYdn);
  real tmp = gm1 * kx;
  L[1] = Lambda[1] * ic2 * ((kx * (c2 - alpha) - c * (kz * v - ky * w)) * dR_dn
                            + (tmp * u) * dRU_dn
                            + (tmp * v + kz * c) * dRV_dn
                            + (tmp * w - ky * c) * dRW_dn
                            - tmp * dRE_dn
                            - kx * al_dRYdn);
  tmp = gm1 * ky;
  L[2] = Lambda[1] * ic2 * ((ky * (c2 - alpha) - c * (kx * w - kz * u)) * dR_dn
                            + (tmp * u - kz * c) * dRU_dn
                            + (tmp * v) * dRV_dn
                            + (tmp * w + kx * c) * dRW_dn
                            - tmp * dRE_dn
                            - ky * al_dRYdn);
  tmp = gm1 * kz;
  L[3] = Lambda[1] * ic2 * ((kz * (c2 - alpha) - c * (ky * u - kx * v)) * dR_dn
                            + (tmp * u + ky * c) * dRU_dn
                            + (tmp * v - kx * c) * dRV_dn
                            + (tmp * w) * dRW_dn
                            - tmp * dRE_dn
                            - kz * al_dRYdn);
  L[4] = Lambda[2] * 0.5 * ic2 * ((alpha - Uk * c) * dR_dn
                                  - (gm1 * u - kx * c) * dRU_dn
                                  - (gm1 * v - ky * c) * dRV_dn
                                  - (gm1 * w - kz * c) * dRW_dn
                                  + gm1 * dRE_dn
                                  + al_dRYdn);
  for (int l = 0; l < param->n_spec; ++l) {
    L[5 + l] = Lambda[1] * (dRY_dn[l] - zone->sv(i, j, k, l) * dR_dn);
  }

  real d[5]{};
  const real h = (cv(i, j, k, 4) + p) / r, iGm1 = 1 / gm1;
  tmp = L[0] + kx * L[1] + ky * L[2] + kz * L[3] + L[4];
  d[0] = tmp;
  d[1] = u * tmp + c * (kx * (L[4] - L[0]) - kz * L[2] + ky * L[3]);
  d[2] = v * tmp + c * (ky * (L[4] - L[0]) + kz * L[1] - kx * L[3]);
  d[3] = w * tmp + c * (kz * (L[4] - L[0]) - ky * L[1] + kx * L[2]);
  d[4] = h * tmp + c * (Uk * (L[4] - L[0])
                        + (kz * v - ky * w - kx * c * iGm1) * L[1]
                        + (kx * w - kz * u - ky * c * iGm1) * L[2]
                        + (ky * u - kx * v - kz * c * iGm1) * L[3]);
  real add{0};
  for (int l = 0; l < param->n_spec; ++l) {
    d[5 + l] = L[5 + l] + zone->sv(i, j, k, l) * tmp;
    add += alphaL[l] * L[5 + l];
  }
  d[4] -= add * iGm1;

  const real minusJacInv = -zone->jac(i, j, k);
  auto &rhs = zone->dq;
  rhs(i, j, k, 0) += minusJacInv * d[0];
  rhs(i, j, k, 1) += minusJacInv * d[1];
  rhs(i, j, k, 2) += minusJacInv * d[2];
  rhs(i, j, k, 3) += minusJacInv * d[3];
  rhs(i, j, k, 4) += minusJacInv * d[4];
  for (int l = 0; l < param->n_spec; ++l) {
    rhs(i, j, k, 5 + l) += minusJacInv * d[5 + l];
  }

  // real dKxDivJac_dK = 0, dKyDivJac_dK = 0, dKzDivJac_dK = 0;
  // const auto &metric = zone->metric;
  // const auto &jac = zone->jac;
  // if (face == 0) {
  //   dKxDivJac_dK = sgn * (cc[0] * metric(i, j, k)(1, 1) / jac(i, j, k)
  //                         + cc[1] * metric(i + sgn, j, k)(1, 1) / jac(i + sgn, j, k)
  //                         + cc[2] * metric(i + 2 * sgn, j, k)(1, 1) / jac(i + 2 * sgn, j, k));
  //   dKyDivJac_dK = sgn * (cc[0] * metric(i, j, k)(1, 2) / jac(i, j, k)
  //                         + cc[1] * metric(i + sgn, j, k)(1, 2) / jac(i + sgn, j, k)
  //                         + cc[2] * metric(i + 2 * sgn, j, k)(1, 2) / jac(i + 2 * sgn, j, k));
  //   dKzDivJac_dK = sgn * (cc[0] * metric(i, j, k)(1, 3) / jac(i, j, k)
  //                         + cc[1] * metric(i + sgn, j, k)(1, 3) / jac(i + sgn, j, k)
  //                         + cc[2] * metric(i + 2 * sgn, j, k)(1, 3) / jac(i + 2 * sgn, j, k));
  // } else if (face == 1) {
  //   dKxDivJac_dK = sgn * (cc[0] * metric(i, j, k)(2, 1) / jac(i, j, k)
  //                         + cc[1] * metric(i, j + sgn, k)(2, 1) / jac(i, j + sgn, k)
  //                         + cc[2] * metric(i, j + 2 * sgn, k)(2, 1) / jac(i, j + 2 * sgn, k));
  //   dKyDivJac_dK = sgn * (cc[0] * metric(i, j, k)(2, 2) / jac(i, j, k)
  //                         + cc[1] * metric(i, j + sgn, k)(2, 2) / jac(i, j + sgn, k)
  //                         + cc[2] * metric(i, j + 2 * sgn, k)(2, 2) / jac(i, j + 2 * sgn, k));
  //   dKzDivJac_dK = sgn * (cc[0] * metric(i, j, k)(2, 3) / jac(i, j, k)
  //                         + cc[1] * metric(i, j + sgn, k)(2, 3) / jac(i, j + sgn, k)
  //                         + cc[2] * metric(i, j + 2 * sgn, k)(2, 3) / jac(i, j + 2 * sgn, k));
  // } else if (face == 2) {
  //   dKxDivJac_dK = sgn * (cc[0] * metric(i, j, k)(3, 1) / jac(i, j, k)
  //                         + cc[1] * metric(i, j, k + sgn)(3, 1) / jac(i, j, k + sgn)
  //                         + cc[2] * metric(i, j, k + 2 * sgn)(3, 1) / jac(i, j, k + 2 * sgn));
  //   dKyDivJac_dK = sgn * (cc[0] * metric(i, j, k)(3, 2) / jac(i, j, k))
  //                  + cc[1] * metric(i, j, k + sgn)(3, 2) / jac(i, j, k + sgn)
  //                  + cc[2] * metric(i, j, k + 2 * sgn)(3, 2) / jac(i, j, k + 2 * sgn);
  //   dKzDivJac_dK = sgn * (cc[0] * metric(i, j, k)(3, 3) / jac(i, j, k)
  //                         + cc[1] * metric(i, j, k + sgn)(3, 3) / jac(i, j, k + sgn)
  //                         + cc[2] * metric(i, j, k + 2 * sgn)(3, 3) / jac(i, j, k + 2 * sgn));
  // }
  //
  // rhs(i, j, k, 0) += minusJacInv * (r * u * dKxDivJac_dK + r * v * dKyDivJac_dK + r * w * dKzDivJac_dK);
  // rhs(i, j, k, 1) += minusJacInv * ((r * u * u + p) * dKxDivJac_dK + r * u * v * dKyDivJac_dK + r * u * w *
  //                                   dKzDivJac_dK);
  // rhs(i, j, k, 2) += minusJacInv * (r * v * u * dKxDivJac_dK + (r * v * v + p) * dKyDivJac_dK + r * v * w *
  //                                   dKzDivJac_dK);
  // rhs(i, j, k, 3) += minusJacInv * (r * w * u * dKxDivJac_dK + r * w * v * dKyDivJac_dK + (r * w * w + p) *
  //                                   dKzDivJac_dK);
  // const real rH = cv(i, j, k, 4) + p;
  // rhs(i, j, k, 4) += minusJacInv * (rH * u * dKxDivJac_dK + rH * v * dKyDivJac_dK + rH * w * dKzDivJac_dK);
  // for (int l = 0; l < param->n_spec; ++l) {
  //   rhs(i, j, k, 5 + l) += minusJacInv * (cv(i, j, k, l + 5) * u * dKxDivJac_dK + cv(i, j, k, l + 5) * v * dKyDivJac_dK
  //                                         + cv(i, j, k, l + 5) * w * dKzDivJac_dK);
  // }
}

template<MixtureModel mix_model> void DBoundCond::nonReflectingBoundary(const Block &block, Field &field,
  DParameter *param) const {
  // Non-reflecting boundary condition
  // First, for outflow boundaries.
  for (size_t l = 0; l < n_outflow; l++) {
    const int label = outflow_info[l].label;
    if (!gxl::exists(nonReflectingBCs, label))
      continue;
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        const auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow_nr_conserv<mix_model> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }
}

template<MixtureModel mix_model> void DBoundCond::apply_boundary_conditions(const Block &block, Field &field,
  DParameter *param, int step) const {
  // Boundary conditions are applied in the order of priority, which with higher priority is applied later.
  // Finally, the communication between faces will be carried out after these bc applied
  // Priority: (-1 - inner faces >) 2-wall > 3-symmetry > 5-inflow = 7-subsonic inflow > 6-outflow = 9-back pressure > 4-farfield

  // 6-outflow
  for (size_t l = 0; l < n_outflow; l++) {
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow<mix_model> <<<BPG, TPB>>>(field.d_ptr, &outflow[l], i_face, param);
    }
  }

  // 5-inflow
  for (size_t l = 0; l < n_inflow; l++) {
    const auto nb = inflow_info[l].n_boundary;
    if (df_label[l] > -1) {
      // Here we assume only one face corresponds to the inflow boundary
      // nb should be 1, and the number of points should be my, mz of the corresponding block
      for (size_t i = 0; i < nb; i++) {
        auto [i_zone, i_face] = inflow_info[l].boundary[i];
        if (i_zone != block.block_id) {
          continue;
        }
        int my = block.my, mz = block.mz, ngg = block.ngg;
        generate_random_numbers(df_label[l], my, mz, ngg);
        apply_convolution(df_label[l], my, mz, ngg);
        compute_fluctuations(param, field.d_ptr, &inflow[l], df_label[l], my, mz, ngg);
        dim3 TPB{32, 8};
        dim3 BPG{(my + 2 * ngg - 1) / TPB.x + 1, (mz + 2 * ngg - 1) / TPB.y + 1};
        apply_inflow_df<mix_model> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], param, fluctuation_dPtr,
                                                  df_label[l]);
      }
    } else {
      for (size_t i = 0; i < nb; i++) {
        auto [i_zone, i_face] = inflow_info[l].boundary[i];
        if (i_zone != block.block_id) {
          continue;
        }
        const auto &hf = block.boundary[i_face];
        const auto ngg = block.ngg;
        uint tpb[3], bpg[3], n_point[3];
        for (size_t j = 0; j < 3; j++) {
          n_point[j] = hf.range_end[j] - hf.range_start[j] + 1;
          tpb[j] = n_point[j] <= 2 * ngg + 1 ? 1 : 16;
          bpg[j] = (n_point[j] - 1) / tpb[j] + 1;
        }
        dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
        apply_inflow<mix_model> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], i_face, param,
                                               profile_dPtr_withGhost, rng_d_ptr, fluctuation_dPtr);
      }
    }
  }

  // 3-symmetry
  for (size_t l = 0; l < n_symmetry; l++) {
    const auto nb = symmetry_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = symmetry_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_symmetry<mix_model> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }

  // 2 - wall
  for (size_t l = 0; l < n_wall; l++) {
    const auto nb = wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = wall_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_wall<mix_model><<<BPG, TPB>>>(field.d_ptr, &wall[l], param, i_face, step);
    }
  }

  // 8 - periodic
  for (size_t l = 0; l < n_periodic; l++) {
    const auto nb = periodic_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = periodic_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_periodic<mix_model><<<BPG, TPB>>>(field.d_ptr, param, i_face);
    }
  }
}

template void DBoundCond::nonReflectingBoundary<MixtureModel::Air>(const Block &block, Field &field,
  DParameter *param) const;

template void DBoundCond::nonReflectingBoundary<MixtureModel::Mixture>(const Block &block, Field &field,
  DParameter *param) const;

template void DBoundCond::apply_boundary_conditions<MixtureModel::Air>(const Block &block, Field &field,
  DParameter *param, int step) const;

template void DBoundCond::apply_boundary_conditions<MixtureModel::Mixture>(const Block &block, Field &field,
  DParameter *param, int step) const;
}

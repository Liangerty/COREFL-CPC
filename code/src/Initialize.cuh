#pragma once

#include "Parameter.h"
#include "Field.h"
#include "ChemData.h"
#include "BoundCond.h"
#include <filesystem>
#include <mpi.h>
#include "gxl_lib/MyString.h"
#include "TurbMethod.hpp"
#include <vector>
#include <fstream>

namespace cfd {
template<MixtureModel mix_model>
void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

template<MixtureModel mix_model>
void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

template<MixtureModel mix_model>
void read_flowfield_with_same_block(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species,
                                    const std::vector<int> &blk_order, MPI_Offset offset_data,
                                    const std::vector<int> &mx, const std::vector<int> &my, const std::vector<int> &mz,
                                    int n_var_old, const std::vector<int> &index_order, MPI_File &fp,
                                    std::array<int, 2> &old_data_info);

template<MixtureModel mix_model>
void read_flowfield_by_0Order_interpolation(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                                            Species &species, const std::vector<int> &blk_order, MPI_Offset offset_data,
                                            const std::vector<int> &mx, const std::vector<int> &my,
                                            const std::vector<int> &mz, int n_var_old,
                                            const std::vector<int> &index_order, MPI_File &fp,
                                            std::array<int, 2> &old_data_info);

template<MixtureModel mix_model>
void read_2D_for_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

void initialize_mixing_layer(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, const Species &species);

__global__ void
initialize_mixing_layer_with_profile(ggxl::VectorField3D<real> *profile_dPtr, int profile_idx, DZone *zone,
                                     int n_scalar);

__global__ void
initialize_mixing_layer_with_info(DZone *zone, const real *var_info, int n_spec, real delta_omega, int n_turb, int n_fl,
                                  int n_ps);

/**
 * @brief To relate the order of variables from the flowfield files to bv, yk, turbulent arrays
 * @param parameter the parameter object
 * @param var_name the array which contains all variables from the flowfield files
 * @param species information about species
 * @param old_data_info the information about the previous simulation, the first one tells if species info exists, the second one tells if turbulent var exists
 * @return an array of orders. 0~5 means density, u, v, w, p, T; 6~5+ns means the species order, 6+ns~... means other variables such as mut...
 */
template<MixtureModel mix_model>
std::vector<int>
identify_variable_labels(Parameter &parameter, std::vector<std::string> &var_name, Species &species,
                         std::array<int, 2> &old_data_info);

void initialize_spec_from_inflow(Parameter &parameter, const Mesh &mesh,
                                 std::vector<Field> &field, Species &species);

void initialize_turb_from_inflow(Parameter &parameter, const Mesh &mesh,
                                 std::vector<Field> &field, Species &species);

void initialize_mixture_fraction_from_species(Parameter &parameter, const Mesh &mesh,
                                              std::vector<Field> &field, Species &species);

void expand_2D_to_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field);

// Implementations
template<MixtureModel mix_model>
void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  const int init_method = parameter.get_int("initial");
  // No matter which method is used to initialize the flowfield,
  // the default inflow is first read and initialize the inf parameters.
  // Otherwise, for simulations that begin from previous simulations,
  // processes other than the one containing the inflow plane would have no info about inf parameters.
  const std::string default_init = parameter.get_string("default_init");
  [[maybe_unused]] Inflow default_inflow(default_init, species, parameter);

  switch (init_method) {
    case 0:
      initialize_from_start(parameter, mesh, field, species);
      break;
    case 1:
      read_flowfield<mix_model>(parameter, mesh, field, species);
      break;
    case 2: // Read a 2D profile, all span-wise layers are initialized with the same profile.
      read_2D_for_3D<mix_model>(parameter, mesh, field, species);
      break;
    default:
      printf("\tThe initialization method is unknown, use freestream value to initialize by default.\n");
      initialize_from_start(parameter, mesh, field, species);
  }
}

template<MixtureModel mix_model>
void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  const std::filesystem::path out_dir("output");
  if (!exists(out_dir)) {
    printf("The directory to flowfield files does not exist!\n");
  }
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  MPI_Offset offset{0};
  MPI_Status status;

  // Magic number, 8 bytes + byte order + file type
  offset += 16;
  // "solution file"
  gxl::read_str_from_binary_MPI_ver(fp, offset);
  int n_var_old{5};
  MPI_File_read_at(fp, offset, &n_var_old, 1, MPI_INT, &status);
  offset += 4;
  std::vector<std::string> var_name;
  var_name.resize(n_var_old);
  for (size_t i = 0; i < n_var_old; ++i) {
    var_name[i] = gxl::read_str_from_binary_MPI_ver(fp, offset);
  }
  // The first one tells if species info exists, if exists (1), from air, (0); in flamelet model, if we have species field, but not mixture fraction field, then (1); if we have all of them, then (2).
  // That is, if the first value is not 0, we have species info.
  // The 2nd one tells if turbulent var exists, if 0 (compute from laminar), 1(From SA), 2(From SST)
  std::array old_data_info{0, 0};
  auto index_order = cfd::identify_variable_labels<mix_model>(parameter, var_name, species,
                                                                    old_data_info);

  // auto index_or = parameter.identify_variable_labels(var_name, species);

  const int n_spec{species.n_spec};
  const int n_turb{parameter.get_int("n_turb")};

  std::vector<int> mx, my, mz;
  int n_zone{0};
  real solution_time{0};
  float marker{299.0f};
  MPI_File_read_at(fp, offset, &marker, 1, MPI_FLOAT, &status);
  offset += 4;
  while (abs(marker - 299.0f) < 1e-3) {
    ++n_zone;
    // 2. Zone name.
    gxl::read_str_from_binary_MPI_ver(fp, offset);
    // Jump through the following info which is not relevant to the current process.
    offset += 8;
    // Read the solution time
    MPI_File_read_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
    offset += 8;
    // Jump through the following info which is not relevant to the current process.
    offset += 20;
    // For ordered zone, specify IMax, JMax, KMax
    int mx1, my1, mz1;
    MPI_File_read_at(fp, offset, &mx1, 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &my1, 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &mz1, 1, MPI_INT, &status);
    offset += 4;
    mx.emplace_back(mx1);
    my.emplace_back(my1);
    mz.emplace_back(mz1);
    // 11. For all zone types (repeat for each Auxiliary data name/value pair), no more data
    offset += 4;
    // Next zone marker
    MPI_File_read_at(fp, offset, &marker, 1, MPI_FLOAT, &status);
    offset += 4;
  }
  parameter.update_parameter("solution_time", solution_time);

  // Next, see if the blocks are exactly the same as the current mesh
  bool same_mesh{false};
  if (n_zone == mesh.n_block_total) {
    same_mesh = true;
    for (size_t i = 0; i < n_zone; ++i) {
      if (mx[i] != mesh.mx_blk[i] || my[i] != mesh.my_blk[i] || mz[i] != mesh.mz_blk[i]) {
        same_mesh = false;
        break;
      }
    }
  }
  if (!same_mesh) {
    // Next, see if the blocks are exactly the same as the current mesh, while only the block order is different
    bool same_block = true;
    std::vector<int> blk_order(n_zone, -1);
    if (n_zone == mesh.n_block_total) {
      for (int i = 0; i < n_zone; ++i) {
        int find{0};
        for (int j = 0; j < n_zone; ++j) {
          if (mesh.mx_blk[i] == mx[j] && mesh.my_blk[i] == my[j] && mesh.mz_blk[i] == mz[j]) {
            ++find;
            blk_order[i] = j;
            if (find > 1) {
              same_block = false;
              break;
            }
          }
        }
        if (blk_order[i] == -1) {
          same_block = false;
          break;
        }
      }
    } else {
      same_block = false;
    }

    if (same_block) {
      // Read the data in the order of the current mesh
      read_flowfield_with_same_block<mix_model>(parameter, mesh, field, species, blk_order, offset,
                                                      mx, my, mz, n_var_old, index_order, fp, old_data_info);
    } else {
      // The mesh is different, we need to interpolate the data to the current mesh
    }
  } else {
    std::vector<std::string> zone_name;
    // Next, data section
    // Jump the front part for process 0 ~ myid-1
    int n_jump_blk{0};
    for (int i = 0; i < parameter.get_int("myid"); ++i) {
      n_jump_blk += mesh.nblk[i];
    }
    int i_blk{0};
    for (int b = 0; b < n_jump_blk; ++b) {
      offset += 16 + 20 * n_var_old;
      const int64_t N = mx[b] * my[b] * mz[b];
      // We always write double precision out
      offset += n_var_old * N * 8;
      ++i_blk;
    }
    // Read data of the current process
    int n_ps = parameter.get_int("n_ps");
    for (size_t blk = 0; blk < mesh.n_block; ++blk) {
      // 1. Zone marker. Value = 299.0, indicates a V112 header.
      offset += 4;
      // 2. Variable data format, 2 for double by default
      offset += 4 * n_var_old;
      // 3. Has passive variables: 0 = no, 1 = yes.
      offset += 4;
      // 4. Has variable sharing 0 = no, 1 = yes.
      offset += 4;
      // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
      offset += 4;
      // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
      offset += 8 * 2 * n_var_old;
      // zone data
      // First, the coordinates x, y and z.
      const int64_t N = mx[i_blk] * my[i_blk] * mz[i_blk];
      offset += 3 * N * 8;
      // Other variables
      const auto &b = mesh[blk];
      MPI_Datatype ty;
      int lsize[3]{mx[i_blk], my[i_blk], mz[i_blk]};
      const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
      int memsize[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
      const int ngg_file{(mx[i_blk] - b.mx) / 2};
      int start_idx[3]{b.ngg - ngg_file, b.ngg - ngg_file, b.ngg - ngg_file};
      MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);
      for (size_t l = 3; l < n_var_old; ++l) {
        auto index = index_order[l];
        // auto index = index_or[l];
        if (index < 6) {
          // basic variables
          auto bv = field[blk].bv[index];
          MPI_File_read_at(fp, offset, bv, 1, ty, &status);
          offset += memsz;
        } /*else if (index >= 1000 && index < 1000 + n_spec) {
          // species variables
          index -= 1000;
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        }*/
        else if (index < 6 + n_spec) {
          // If air, n_spec=0;
          // species variables
          index -= 6;
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else if ((parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) &&
                   index < 6 + n_spec + n_turb) {
          // RANS/DDES
          // If laminar, n_turb=0
          // turbulent variables
          index -= 6;
          if (n_turb == old_data_info[1]) {
            // SA from SA or SST from SST
            auto sv = field[blk].sv[index];
            MPI_File_read_at(fp, offset, sv, 1, ty, &status);
            offset += memsz;
          } else if (n_turb == 1 && old_data_info[1] == 2) {
            // SA from SST. Currently, just use freestream value to initialize. Modify this later when I write SA
            old_data_info[1] = 0;
          } else if (n_turb == 2 && old_data_info[1] == 1) {
            // SST from SA. As ACANS has done, the turbulent variables are initialized from freestream value
            old_data_info[1] = 0;
          }
        } else if ((parameter.get_int("species") == 2 || parameter.get_int("reaction") == 2) &&
                   index < 6 + n_spec + n_turb + 2) {
          // If flamelet model is used, we need to read the flamelet info
          index -= 6;
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else if (n_ps > 0 && index < 6 + parameter.get_int("i_ps") + n_ps) {
          // Passive scalar
          index -= 6;
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else {
          // No matched label, just ignore
          offset += memsz;
        }
      }
      ++i_blk;
    }
    MPI_File_close(&fp);

    // Next, if the previous simulation does not contain some of the variables used in the current simulation,
    // then we initialize them here
    if constexpr (mix_model != MixtureModel::Air) {
      if (old_data_info[0] == 0) {
        initialize_spec_from_inflow(parameter, mesh, field, species);
        old_data_info[0] = 1;
      }
    }

    for (auto &f: field) {
      cudaMemcpy(f.h_ptr->bv.data(), f.bv.data(), f.h_ptr->bv.size() * sizeof(real) * 6, cudaMemcpyHostToDevice);
      cudaMemcpy(f.h_ptr->sv.data(), f.sv.data(), f.h_ptr->sv.size() * sizeof(real) * parameter.get_int("n_scalar"),
                 cudaMemcpyHostToDevice);
    }

    std::ifstream step_file{"output/message/step.txt"};
    int step{0};
    step_file >> step;
    step_file.close();
    parameter.update_parameter("step", step);

    if (parameter.get_int("myid") == 0) {
      printf("\t->-> %-25s : initialization method.\n", "From previous results");
    }
  }
}

template<MixtureModel mix_model>
std::vector<int>
identify_variable_labels(Parameter &parameter, std::vector<std::string> &var_name, Species &species,
                         std::array<int, 2> &old_data_info) {
  std::vector<int> labels;
  const int n_spec = species.n_spec;
  const int n_turb = parameter.get_int("n_turb");
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0 and no match would be found.
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
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5;
    } else {
      if constexpr (mix_model != MixtureModel::Air) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label;
            old_data_info[0] = 1;
            break;
          }
        }
        if (n == "MIXTUREFRACTION") {
          // Mixture fraction
          l = 6 + n_spec + n_turb;
          old_data_info[0] = 2;
        } else if (n == "MIXTUREFRACTIONVARIANCE") {
          // Mixture fraction variance
          l = 6 + n_spec + n_turb + 1;
          old_data_info[0] = 2;
        }
      }
      if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
        const int i_ps = parameter.get_int("i_ps");
        for (int i = 0; i < n_ps; ++i) {
          if (n == "PS" + std::to_string(i + 1)) {
            l = i_ps + i + 6;
            break;
          }
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

template<MixtureModel mix_model>
void read_2D_for_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  const std::filesystem::path out_dir(parameter.get_string("result_file"));
  if (!exists(out_dir)) {
    printf("The flowfield file does not exist!\n");
  }
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, out_dir.string().c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  MPI_Offset offset{0};
  MPI_Status status;

  // Magic number, 8 bytes + byte order + file type
  offset += 16;
  // "solution file"
  gxl::read_str_from_binary_MPI_ver(fp, offset);
  int n_var_old{5};
  MPI_File_read_at(fp, offset, &n_var_old, 1, MPI_INT, &status);
  offset += 4;
  std::vector<std::string> var_name;
  var_name.resize(n_var_old);
  for (size_t i = 0; i < n_var_old; ++i) {
    var_name[i] = gxl::read_str_from_binary_MPI_ver(fp, offset);
  }
  // The first one tells if species info exists, if exists (1), from air, (0); in flamelet model, if we have species field, but not mixture fraction field, then (1); if we have all of them, then (2).
  // That is, if the first value is not 0, we have species info.
  // The 2nd one tells if turbulent var exists, if 0 (compute from laminar), 1(From SA), 2(From SST)
  std::array old_data_info{0, 0};
  auto index_order = cfd::identify_variable_labels<mix_model>(parameter, var_name, species,
                                                                    old_data_info);
  const int n_spec{species.n_spec};
  const int n_turb{parameter.get_int("n_turb")};
  auto *mx = new int[mesh.n_block_total], *my = new int[mesh.n_block_total], *mz = new int[mesh.n_block_total];
  real solution_time{0};
  for (int b = 0; b < mesh.n_block_total; ++b) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    offset += 4;
    // 2. Zone name.
    gxl::read_str_from_binary_MPI_ver(fp, offset);
    // Jump through the following info which is not relevant to the current process.
    offset += 8;
    // Read the solution time
    MPI_File_read_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
    offset += 8;
    // Jump through the following info which is not relevant to the current process.
    offset += 20;
    // For ordered zone, specify IMax, JMax, KMax
    MPI_File_read_at(fp, offset, &mx[b], 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &my[b], 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &mz[b], 1, MPI_INT, &status);
    offset += 4;
    // 11. For all zone types (repeat for each Auxiliary data name/value pair), no more data
    offset += 4;
  }
  parameter.update_parameter("solution_time", solution_time);
  // Read the EOHMARKER
  offset += 4;

  std::vector<std::string> zone_name;
  // Next, data section
  // Jump the front part for process 0 ~ myid-1
  int n_jump_blk{0};
  for (int i = 0; i < parameter.get_int("myid"); ++i) {
    n_jump_blk += mesh.nblk[i];
  }
  int i_blk{0};
  for (int b = 0; b < n_jump_blk; ++b) {
    offset += 16 + 20 * n_var_old;
    const int N = mx[b] * my[b] * mz[b];
    // We always write double precision out
    offset += n_var_old * N * 8;
    ++i_blk;
  }
  // Read data of the current process
  for (size_t blk = 0; blk < mesh.n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    offset += 4;
    // 2. Variable data format, 2 for double by default
    offset += 4 * n_var_old;
    // 3. Has passive variables: 0 = no, 1 = yes.
    offset += 4;
    // 4. Has variable sharing 0 = no, 1 = yes.
    offset += 4;
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    offset += 8 * 2 * n_var_old;
    // zone data
    // First, the coordinates x, y and z.
    const int N = mx[i_blk] * my[i_blk] * mz[i_blk];
    offset += 3 * N * 8;
    // Other variables
    const auto &b = mesh[blk];
    MPI_Datatype ty;
    // The mz here will be 1.
    int lsize[3]{mx[i_blk], my[i_blk], mz[i_blk]};
    const auto memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const int ngg_file{(mx[i_blk] - b.mx) / 2};
    int start_idx[3]{b.ngg - ngg_file, b.ngg - ngg_file, b.ngg - ngg_file};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    for (size_t l = 3; l < n_var_old; ++l) {
      auto index = index_order[l];
      if (index < 6) {
        // basic variables
        auto bv = field[blk].bv[index];
        MPI_File_read_at(fp, offset, bv, 1, ty, &status);
        offset += memsz;
      } else if (index < 6 + n_spec) {
        // If air, n_spec=0;
        // species variables
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else if (index < 6 + n_spec + n_turb) {
        // If laminar, n_turb=0
        // turbulent variables
        index -= 6;
        if (n_turb == old_data_info[1]) {
          // SA from SA or SST from SST
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else if (n_turb == 1 && old_data_info[1] == 2) {
          // SA from SST. Currently, just use freestream value to initialize. Modify this later when I write SA
          old_data_info[1] = 0;
        } else if (n_turb == 2 && old_data_info[1] == 1) {
          // SST from SA. As ACANS has done, the turbulent variables are initialized from freestream value
          old_data_info[1] = 0;
        }
      } else if (index < 6 + n_spec + n_turb + 2) {
        // Flamelet info
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else {
        // No matched label, just ignore
        offset += memsz;
      }
    }
    ++i_blk;
  }
  MPI_File_close(&fp);

  // Next, if the previous simulation does not contain some of the variables used in the current simulation,
  // then we initialize them here
  if constexpr (mix_model != MixtureModel::Air) {
    if (old_data_info[0] == 0) {
      initialize_spec_from_inflow(parameter, mesh, field, species);
      old_data_info[0] = 1;
    }
  }

  // Next, we expand the 2D profile to 3D.
  expand_2D_to_3D(parameter, mesh, field);

  for (auto &f: field) {
    cudaMemcpy(f.h_ptr->bv.data(), f.bv.data(), f.h_ptr->bv.size() * sizeof(real) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(f.h_ptr->sv.data(), f.sv.data(), f.h_ptr->sv.size() * sizeof(real) * parameter.get_int("n_scalar"),
               cudaMemcpyHostToDevice);
  }

  std::ifstream step_file{"output/message/step.txt"};
  int step{0};
  step_file >> step;
  step_file.close();
  parameter.update_parameter("step", step);

  if (parameter.get_int("myid") == 0) {
    printf(
      "Flowfield is initialized from previous 2D simulation results, the span-wise is copied from the 2D profile.\n");
  }
}

template<MixtureModel mix_model>
void
read_flowfield_with_same_block(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species,
                               const std::vector<int> &blk_order, MPI_Offset offset_data,
                               const std::vector<int> &mx, const std::vector<int> &my, const std::vector<int> &mz,
                               int n_var_old, const std::vector<int> &index_order, MPI_File &fp,
                               std::array<int, 2> &old_data_info) {
  int blk_start = 0;
  const int myid = parameter.get_int("myid");
  for (int i = 0; i < myid; ++i) {
    blk_start += mesh.nblk[i];
  }

  std::vector<std::string> zone_name;
  // Next, data section
  int n_spec = species.n_spec;
  int n_turb = parameter.get_int("n_turb");
  int n_ps = parameter.get_int("n_ps");
  MPI_Status status;
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // Get the offset of the corresponding block
    MPI_Offset offset = offset_data;
    const int i_blk_read{blk_order[blk_start + blk]};
    printf("\tProcess %d, block %d is read from block %d in the previous simulation.\n", myid, blk, i_blk_read);
    for (int counter = 0; counter < i_blk_read; ++counter) {
      offset += 16 + 20 * n_var_old;
      const int64_t N = mx[counter] * my[counter] * mz[counter];
      // We always write double precision out
      offset += n_var_old * N * 8;
    }
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    offset += 4;
    // 2. Variable data format, 2 for double by default
    offset += 4 * n_var_old;
    // 3. Has passive variables: 0 = no, 1 = yes.
    offset += 4;
    // 4. Has variable sharing 0 = no, 1 = yes.
    offset += 4;
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    offset += 8 * 2 * n_var_old;
    // zone data
    // First, the coordinates x, y and z.
    const int64_t N = mx[i_blk_read] * my[i_blk_read] * mz[i_blk_read];
    offset += 3 * N * 8;
    // Other variables
    const auto &b = mesh[blk];
    MPI_Datatype ty;
    int lsize[3]{mx[i_blk_read], my[i_blk_read], mz[i_blk_read]};
    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const int ngg_file{(mx[i_blk_read] - b.mx) / 2};
    int start_idx[3]{b.ngg - ngg_file, b.ngg - ngg_file, b.ngg - ngg_file};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    for (size_t l = 3; l < n_var_old; ++l) {
      auto index = index_order[l];
      if (index < 6) {
        // basic variables
        auto bv = field[blk].bv[index];
        MPI_File_read_at(fp, offset, bv, 1, ty, &status);
        offset += memsz;
      } else if (index < 6 + n_spec) {
        // If air, n_spec=0;
        // species variables
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else if ((parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) &&
                 index < 6 + n_spec + n_turb) {
        // RANS/DDES
        // If laminar, n_turb=0
        // turbulent variables
        index -= 6;
        if (n_turb == old_data_info[1]) {
          // SA from SA or SST from SST
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else if (n_turb == 1 && old_data_info[1] == 2) {
          // SA from SST. Currently, just use freestream value to initialize. Modify this later when I write SA
          old_data_info[1] = 0;
        } else if (n_turb == 2 && old_data_info[1] == 1) {
          // SST from SA. As ACANS has done, the turbulent variables are initialized from freestream value
          old_data_info[1] = 0;
        }
      } else if ((parameter.get_int("species") == 2 || parameter.get_int("reaction") == 2) &&
                 index < 6 + n_spec + n_turb + 2) {
        // If flamelet model is used, we need to read the flamelet info
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else if (n_ps > 0 && index < 6 + parameter.get_int("i_ps") + n_ps) {
        // Passive scalar
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else {
        // No matched label, just ignore
        offset += memsz;
      }
    }
  }
  MPI_File_close(&fp);

  // Next, if the previous simulation does not contain some of the variables used in the current simulation,
  // then we initialize them here
  if constexpr (mix_model != MixtureModel::Air) {
    if (old_data_info[0] == 0) {
      initialize_spec_from_inflow(parameter, mesh, field, species);
      old_data_info[0] = 1;
    }
  }

  for (auto &f: field) {
    cudaMemcpy(f.h_ptr->bv.data(), f.bv.data(), f.h_ptr->bv.size() * sizeof(real) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(f.h_ptr->sv.data(), f.sv.data(), f.h_ptr->sv.size() * sizeof(real) * parameter.get_int("n_scalar"),
               cudaMemcpyHostToDevice);
  }

  std::ifstream step_file{"output/message/step.txt"};
  int step{0};
  step_file >> step;
  step_file.close();
  parameter.update_parameter("step", step);

  if (parameter.get_int("myid") == 0) {
    printf("\t->-> %-25s : initialization method.\n", "From previous results");
  }
}


template<MixtureModel mix_model>
void read_flowfield_by_0Order_interpolation(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                                            Species &species, const std::vector<int> &blk_order, MPI_Offset offset_data,
                                            const std::vector<int> &mx, const std::vector<int> &my,
                                            const std::vector<int> &mz, int n_var_old,
                                            const std::vector<int> &index_order, MPI_File &fp,
                                            std::array<int, 2> &old_data_info) {
  // First, get the max and min coordinates of the current mesh
  const int n_block = mesh.n_block;
  std::vector<real> x_min(n_block, 1e10), x_max(n_block, -1e10), y_min(n_block, 1e10), y_max(n_block, -1e10),
      z_min(n_block, 1e10), z_max(n_block, -1e10);
  for (int blk = 0; blk < n_block; ++blk) {
    const auto &b = mesh[blk];
    int mx1 = b.mx, my1 = b.my, mz1 = b.mz;
    for (int k = 0; k < mz1; ++k) {
      for (int j = 0; j < my1; ++j) {
        for (int i = 0; i < mx1; ++i) {
          x_min[blk] = std::min(x_min[blk], b.x(i, j, k));
          x_max[blk] = std::max(x_max[blk], b.x(i, j, k));
          y_min[blk] = std::min(y_min[blk], b.y(i, j, k));
          y_max[blk] = std::max(y_max[blk], b.y(i, j, k));
          z_min[blk] = std::min(z_min[blk], b.z(i, j, k));
          z_max[blk] = std::max(z_max[blk], b.z(i, j, k));
        }
      }
    }
  }

  // Next, find the max block of previous simulation
  int max_blk_idx{0};
  const int n_block_read{(int) (mx.size())};
  int64_t max_N{mx[1] * my[1] * mz[1]};
  for (int b = 1; b < n_block_read; ++b) {
    const int64_t N = mx[b] * my[b] * mz[b];
    if (N > max_N) {
      max_N = N;
      max_blk_idx = b;
    }
  }
}
}

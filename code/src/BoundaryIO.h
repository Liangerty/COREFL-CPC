#pragma once

#include <filesystem>
#include "BoundCond.cuh"
#include "Define.h"
#include "gxl_lib/MyString.h"
#include "gxl_lib/Array.hpp"

namespace cfd {
template<MixtureModel mix_model, OutputTimeChoice output_time_choice = OutputTimeChoice::Instance>
struct BoundaryIO {
  const Parameter &parameter;
  const Mesh &mesh;
  const Species &species;
  const std::vector<Field> &field;
  // The labels of the boundary conditions to be outputted
  std::vector<int> labels_to_output{};
  // For every boundary condition to be outputted, record its bc name
  std::vector<std::string> name_of_boundary{};
  std::vector<int> type{};
  std::vector<MPI_Offset> offset_header{};
  std::vector<std::vector<MPI_Offset>> offset_minmax_var{};
  std::vector<std::vector<MPI_Offset>> offset_var{};
  std::vector<int> n_var{};

  std::vector<std::vector<const Boundary *>> boundaries{};
  std::vector<std::vector<int>> mx{}, my{}, mz{};
  std::vector<std::vector<int>> xs{}, xe{}, ys{}, ye{}, zs{}, ze{};
  std::vector<std::vector<int>> block_ids{};

  explicit BoundaryIO(const Parameter &parameter_, const Mesh &mesh_, const Species &species_,
                      std::vector<Field> &_field);

  void print_boundary();

private:
  void write_header(const std::vector<Field> &_field);
};

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
BoundaryIO<mix_model, output_time_choice>::BoundaryIO(const Parameter &parameter_, const Mesh &mesh_,
                                                                   const Species &species_, std::vector<Field> &_field)
  : parameter(parameter_),
  mesh(mesh_),
  species(species_),
  field(_field),
  labels_to_output(parameter.get_int_array("output_bc")) {
  const auto n_labels{labels_to_output.size()};
  name_of_boundary.resize(n_labels);
  boundaries.resize(n_labels);
  type.resize(n_labels);
  xs.resize(n_labels);
  ys.resize(n_labels);
  zs.resize(n_labels);
  xe.resize(n_labels);
  ye.resize(n_labels);
  ze.resize(n_labels);
  mx.resize(n_labels);
  my.resize(n_labels);
  mz.resize(n_labels);
  offset_header.resize(n_labels);
  offset_minmax_var.resize(n_labels);
  offset_var.resize(n_labels);
  n_var.resize(n_labels);
  block_ids.resize(n_labels);

  const auto bc_names{parameter.get_string_array("boundary_conditions")};
  for (auto &name: bc_names) {
    auto &bc = parameter.get_struct(name);
    for (int i = 0; i < n_labels; ++i) {
      if (std::get<int>(bc.at("label")) == labels_to_output[i]) {
        name_of_boundary[i] = name;
        auto type_name{std::get<std::string>(bc.at("type"))};
        if (type_name == "wall") {
          type[i] = 2;
        } else if (type_name == "symmetry") {
          type[i] = 3;
        } else if (type_name == "farfield") {
          type[i] = 4;
        } else if (type_name == "inflow") {
          type[i] = 5;
        } else if (type_name == "outflow") {
          type[i] = 6;
        }
        break;
      }
    }
  }

  const int ngg{parameter.get_int("ngg")};
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const auto &b = mesh[blk];
    for (const auto &f: b.boundary) {
      const auto label{f.type_label};
      for (int l = 0; l < n_labels; ++l) {
        if (label == labels_to_output[l]) {
          boundaries[l].push_back(&f);
          const auto face = f.face;
          int range_start[3]{f.range_start[0] + ngg, f.range_start[1] + ngg, f.range_start[2] + ngg};
          int range_end[3]{f.range_end[0] - ngg, f.range_end[1] - ngg, f.range_end[2] - ngg};
          range_start[face] -= ngg;
          range_end[face] += ngg;
          if (type[l] == 2) {
            range_start[0] = f.range_start[0];
            range_start[1] = f.range_start[1];
            range_start[2] = f.range_start[2];
            range_end[0] = f.range_end[0];
            range_end[1] = f.range_end[1];
            range_end[2] = f.range_end[2];
          }
          xs[l].emplace_back(range_start[0]);
          xe[l].emplace_back(range_end[0]);
          ys[l].emplace_back(range_start[1]);
          ye[l].emplace_back(range_end[1]);
          zs[l].emplace_back(range_start[2]);
          ze[l].emplace_back(range_end[2]);
          mx[l].emplace_back(std::abs(range_end[0] - range_start[0]) + 1);
          my[l].emplace_back(std::abs(range_end[1] - range_start[1]) + 1);
          mz[l].emplace_back(std::abs(range_end[2] - range_start[2]) + 1);
          block_ids[l].push_back(blk);
        }
      }
    }
  }

  const std::filesystem::path out_dir("output");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }

  write_header(field);
}

template<MixtureModel mix_model, int BoundaryType>
int32_t
acquire_boundary_variable_names(std::vector<std::string> &var_name, const Parameter &parameter,
                                const Species &species) {
  int32_t nv = 3 + 6; // x,y,z + rho,u,v,w,p,T
  if constexpr (BoundaryType != 2) {
    // Not wall, add Mach number
    ++nv;
    var_name.emplace_back("Mach");
  }
  if constexpr (mix_model != MixtureModel::Air) {
    nv += parameter.get_int("n_spec"); // Y_k
    var_name.resize(nv);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      if constexpr (BoundaryType != 2) {
        var_name[ind + 10] = name;
      } else {
        var_name[ind + 9] = name;
      }
    }
  }
  return nv;
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void
BoundaryIO<mix_model, output_time_choice>::write_header(const std::vector<Field> &_field) {
  const std::filesystem::path out_dir("output");
  const int n_proc{parameter.get_int("n_proc")};
  for (int l = 0; l < labels_to_output.size(); ++l) {
    std::string file_name{out_dir.string() + "/" + name_of_boundary[l] + ".plt"};
    MPI_File fp;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fp);
    MPI_Status status;

    // I. Header section

    // Each file should have only one header; thus we let process 0 to write it.

    MPI_Offset offset{0};
    const int myid{parameter.get_int("myid")};
    n_var[l] = 3 + 6;
    std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature"};
    if (type[l] == 2) {
      n_var[l] = acquire_boundary_variable_names<mix_model, 2>(var_name, parameter, species);
    } else {
      n_var[l] = acquire_boundary_variable_names<mix_model, 1>(var_name, parameter, species);
    }
    if (myid == 0) {
      // i. Magic number, Version number
      // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
      // difference is related to us. For common use, we use V112.
      constexpr auto magic_number{"#!TDV112"};
      gxl::write_str_without_null(magic_number, fp, offset);

      // ii. Integer value of 1
      constexpr int32_t byte_order{1};
      MPI_File_write_at(fp, offset, &byte_order, 1, MPI_INT32_T, &status);
      offset += 4;

      // iii. Title and variable names.
      // 1. FileType: 0=full, 1=grid, 2=solution
      constexpr int32_t file_type{0};
      MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
      offset += 4;
      // 2. Title
      gxl::write_str("Solution file", fp, offset);
      // 3. Number of variables in the datafile, common data include, n_var = 3(x,y,z)+6(rho,u,v,w,p,t)+n_spec+n_scalar
      MPI_File_write_at(fp, offset, &n_var[l], 1, MPI_INT32_T, &status);
      offset += 4;
      // 4. Variable names.
      for (auto &name: var_name) {
        gxl::write_str(name.c_str(), fp, offset);
      }
    }

    // Next part, we need the info about other processes. Thus, we communicate with other processes to get the info first.
    int n_face_this{static_cast<int>(boundaries[l].size())};
    auto *n_face = new int[n_proc];
    MPI_Allgather(&n_face_this, 1, MPI_INT, n_face, 1, MPI_INT, MPI_COMM_WORLD);
    auto *disp = new int[n_proc];
    disp[0] = 0;
    int n_face_total{n_face[0]};
    for (int i = 1; i < n_proc; ++i) {
      disp[i] = disp[i - 1] + n_face[i - 1];
      n_face_total += n_face[i];
    }
    // Collect all length info to the root process
    auto *len_x = new int[n_face_total];
    auto *len_y = new int[n_face_total];
    auto *len_z = new int[n_face_total];
    MPI_Allgatherv(mx[l].data(), n_face[myid], MPI_INT, len_x, n_face, disp, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(my[l].data(), n_face[myid], MPI_INT, len_y, n_face, disp, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(mz[l].data(), n_face[myid], MPI_INT, len_z, n_face, disp, MPI_INT, MPI_COMM_WORLD);

    if (myid == 0) {
      // iv. Zones
      for (int i = 0; i < n_face_total; ++i) {
        // 1. Zone marker. Value = 299.0, indicates a V112 header.
        constexpr float zone_marker{299.0f};
        MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
        offset += 4;
        // 2. Zone name.
        gxl::write_str(("zone " + std::to_string(i)).c_str(), fp, offset);
        // 3. Parent zone. No longer used
        constexpr int32_t parent_zone{-1};
        MPI_File_write_at(fp, offset, &parent_zone, 1, MPI_INT32_T, &status);
        offset += 4;
        // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
        constexpr int32_t strand_id{-1};
        MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
        offset += 4;
        // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
        constexpr double solution_time{0};
        MPI_File_write_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
        offset += 8;
        // 6. Default Zone Color. Seldom used. Set to -1.
        constexpr int32_t zone_color{-1};
        MPI_File_write_at(fp, offset, &zone_color, 1, MPI_INT32_T, &status);
        offset += 4;
        // 7. ZoneType 0=ORDERED
        constexpr int32_t zone_type{0};
        MPI_File_write_at(fp, offset, &zone_type, 1, MPI_INT32_T, &status);
        offset += 4;
        // 8. Specify Var Location. 0 = All data is located at the nodes
        constexpr int32_t var_location{0};
        MPI_File_write_at(fp, offset, &var_location, 1, MPI_INT32_T, &status);
        offset += 4;
        // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
        // raw face neighbors are not defined for these zone types.
        constexpr int32_t raw_face_neighbor{0};
        MPI_File_write_at(fp, offset, &raw_face_neighbor, 1, MPI_INT32_T, &status);
        offset += 4;
        // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
        constexpr int32_t miscellaneous_face{0};
        MPI_File_write_at(fp, offset, &miscellaneous_face, 1, MPI_INT32_T, &status);
        offset += 4;
        // For ordered zone, specify IMax, JMax, KMax
        MPI_File_write_at(fp, offset, &len_x[i], 1, MPI_INT32_T, &status);
        offset += 4;
        MPI_File_write_at(fp, offset, &len_y[i], 1, MPI_INT32_T, &status);
        offset += 4;
        MPI_File_write_at(fp, offset, &len_z[i], 1, MPI_INT32_T, &status);
        offset += 4;

        // 11. For all zone types (repeat for each Auxiliary data name/value pair)
        // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
        // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
        // No more data
        constexpr int32_t no_more_auxi_data{0};
        MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
        offset += 4;
      }

      // End of Header
      constexpr float EOHMARKER{357.0f};
      MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
      offset += 4;

      offset_header[l] = offset;
    }
    MPI_Bcast(&offset_header[l], 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    // Compute the offsets for each process
    MPI_Offset new_offset{0};
    int i_face{0};
    for (int p = 0; p < myid; ++p) {
      for (int f = 0; f < n_face[p]; ++f) {
        new_offset += 16 + 20 * n_var[l];
        const int N = len_x[i_face] * len_y[i_face] * len_z[i_face];
        // We always write double precision out
        new_offset += n_var[l] * N * 8;
        ++i_face;
      }
    }
    offset_header[l] += new_offset;

    // write the common data section
    offset = offset_header[l];
    for (int f = 0; f < n_face_this; ++f) {
      // 1. Zone marker. Value = 299.0, indicates a V112 header.
      constexpr float zone_marker{299.0f};
      MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
      offset += 4;
      // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
      constexpr int32_t data_format{2};
      for (int m = 0; m < n_var[l]; ++m) {
        MPI_File_write_at(fp, offset, &data_format, 1, MPI_INT32_T, &status);
        offset += 4;
      }
      // 3. Has passive variables: 0 = no, 1 = yes.
      constexpr int32_t passive_var{0};
      MPI_File_write_at(fp, offset, &passive_var, 1, MPI_INT32_T, &status);
      offset += 4;
      // 4. Has variable sharing 0 = no, 1 = yes.
      constexpr int32_t shared_var{0};
      MPI_File_write_at(fp, offset, &shared_var, 1, MPI_INT32_T, &status);
      offset += 4;
      // 5. Zero-based zone number to share connectivity list with (-1 = no sharing).
      constexpr int32_t shared_connect{-1};
      MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
      offset += 4;
      // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
      // For each non-shared and non-passive variable (as specified above):
      const auto blk = block_ids[l][f];
      auto &b{mesh[blk]};

      double min_val{b.x(xs[l][f], ys[l][f], zs[l][f])}, max_val{b.x(xs[l][f], ys[l][f], zs[l][f])};
      for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
        for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
          for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
            min_val = std::min(min_val, b.x(i, j, k));
            max_val = std::max(max_val, b.x(i, j, k));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      min_val = b.y(xs[l][f], ys[l][f], zs[l][f]);
      max_val = b.y(xs[l][f], ys[l][f], zs[l][f]);
      for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
        for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
          for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
            min_val = std::min(min_val, b.y(i, j, k));
            max_val = std::max(max_val, b.y(i, j, k));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      min_val = b.z(xs[l][f], ys[l][f], zs[l][f]);
      max_val = b.z(xs[l][f], ys[l][f], zs[l][f]);
      for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
        for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
          for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
            min_val = std::min(min_val, b.z(i, j, k));
            max_val = std::max(max_val, b.z(i, j, k));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      const auto &v{field[blk]};
      // Later, the max/min values of flow variables are computed.
      offset_minmax_var[l].emplace_back(offset);
      for (int m = 0; m < 6; ++m) {
        min_val = v.bv(xs[l][f], ys[l][f], zs[l][f], m);
        max_val = v.bv(xs[l][f], ys[l][f], zs[l][f], m);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.bv(i, j, k, m));
              max_val = std::max(max_val, v.bv(i, j, k, m));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      if (type[l] != 2) {
        min_val = v.ov(xs[l][f], ys[l][f], zs[l][f], 0);
        max_val = v.ov(xs[l][f], ys[l][f], zs[l][f], 0);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.ov(i, j, k, 0));
              max_val = std::max(max_val, v.ov(i, j, k, 0));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
      const int n_scalar{parameter.get_int("n_scalar")};
      for (int m = 0; m < n_scalar; ++m) {
        min_val = v.sv(xs[l][f], ys[l][f], zs[l][f], m);
        max_val = v.sv(xs[l][f], ys[l][f], zs[l][f], m);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.sv(i, j, k, m));
              max_val = std::max(max_val, v.sv(i, j, k, m));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }

      // 7. Zone Data.
      MPI_Datatype ty;
      const int ngg{parameter.get_int("ngg")};
      int big_size[3]{b.mx + 2 * ngg, b.my + 2 * ngg, b.mz + 2 * ngg};
      int sub_size[3]{mx[l][f], my[l][f], mz[l][f]};
      const auto mem_sz = sub_size[0] * sub_size[1] * sub_size[2] * 8;
      int start_idx[3]{xs[l][f] + ngg, ys[l][f] + ngg, zs[l][f] + ngg};
      MPI_Type_create_subarray(3, big_size, sub_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);
      MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
      offset += mem_sz;
      MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
      offset += mem_sz;
      MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
      offset += mem_sz;
      // Later, the variables are outputted.
      offset_var[l].emplace_back(offset);
      for (int m = 0; m < 6; ++m) {
        auto var = v.bv[m];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
      if (type[l] != 2) {
        auto var = v.ov[0];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
      for (int m = 0; m < n_scalar; ++m) {
        auto var = v.sv[m];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
    }

    MPI_File_close(&fp);
    delete[]len_x;
    delete[]len_y;
    delete[]len_z;
    delete[]n_face;
    delete[]disp;
  }
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void BoundaryIO<mix_model, output_time_choice>::print_boundary() {
  const std::filesystem::path out_dir("output");
  for (int l = 0; l < labels_to_output.size(); ++l) {
    std::string file_name{out_dir.string() + "/" + name_of_boundary[l] + ".plt"};
    MPI_File fp;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fp);
    MPI_Status status;

    // II. Data Section
    // First, modify the new min/max values of the variables
    const int n_face{static_cast<int>(boundaries[l].size())};
    for (int f = 0; f < n_face; ++f) {
      MPI_Offset offset{offset_minmax_var[l][f]};
      const auto blk = block_ids[l][f];

      double min_val{0}, max_val{1};
      auto &v{field[blk]};
      for (int m = 0; m < 6; ++m) {
        min_val = v.bv(xs[l][f], ys[l][f], zs[l][f], m);
        max_val = v.bv(xs[l][f], ys[l][f], zs[l][f], m);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.bv(i, j, k, m));
              max_val = std::max(max_val, v.bv(i, j, k, m));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      if (type[l] != 2) {
        min_val = v.ov(xs[l][f], ys[l][f], zs[l][f], 0);
        max_val = v.ov(xs[l][f], ys[l][f], zs[l][f], 0);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.ov(i, j, k, 0));
              max_val = std::max(max_val, v.ov(i, j, k, 0));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
      const int n_scalar{parameter.get_int("n_scalar")};
      for (int m = 0; m < n_scalar; ++m) {
        min_val = v.sv(xs[l][f], ys[l][f], zs[l][f], m);
        max_val = v.sv(xs[l][f], ys[l][f], zs[l][f], m);
        for (int k = zs[l][f]; k <= ze[l][f]; ++k) {
          for (int j = ys[l][f]; j <= ye[l][f]; ++j) {
            for (int i = xs[l][f]; i <= xe[l][f]; ++i) {
              min_val = std::min(min_val, v.sv(i, j, k, m));
              max_val = std::max(max_val, v.sv(i, j, k, m));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      // 7. Zone marker. Value = 299.0, indicates a V112 header.
      // 7. Zone Data.
      MPI_Datatype ty;
      const int ngg{parameter.get_int("ngg")};
      const auto &b = mesh[blk];
      int big_size[3]{b.mx + 2 * ngg, b.my + 2 * ngg, b.mz + 2 * ngg};
      int sub_size[3]{mx[l][f], my[l][f], mz[l][f]};
      const auto mem_sz = sub_size[0] * sub_size[1] * sub_size[2] * 8;
      int start_idx[3]{xs[l][f] + ngg, ys[l][f] + ngg, zs[l][f] + ngg};
      MPI_Type_create_subarray(3, big_size, sub_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);

      offset = offset_var[l][f];
      for (int m = 0; m < 6; ++m) {
        auto var = v.bv[m];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
      if (type[l] != 2) {
        auto var = v.ov[0];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
      for (int m = 0; m < n_scalar; ++m) {
        auto var = v.sv[m];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += mem_sz;
      }
    }
    MPI_File_close(&fp);
  }
}
}

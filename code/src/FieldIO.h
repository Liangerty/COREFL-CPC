#pragma once

#include "Define.h"
#include "mpi.h"
#include "Field.h"
#include "ChemData.h"
#include <filesystem>
#include <fstream>
#include "gxl_lib/MyString.h"
#include "TurbMethod.hpp"

namespace cfd {
int add_other_variable_name(std::vector<std::string> &var_name, const Parameter &parameter);

MPI_Offset write_static_max_min(MPI_Offset offset, const Field &field, int ngg, MPI_File &fp);

MPI_Offset write_dynamic_max_min_first_step(MPI_Offset offset, const Field &field, int ngg, MPI_File &fp);

MPI_Offset
write_dynamic_max_min(MPI_Offset offset, const Field &field, int ngg, MPI_File &fp);

MPI_Offset
write_static_array(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, long long int mem_sz);

MPI_Offset
write_dynamic_array(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, long long int mem_sz);

template<MixtureModel mix_model, OutputTimeChoice output_time_choice = OutputTimeChoice::Instance>
class FieldIO {
  const int myid{0};
  const Mesh &mesh;
  std::vector<Field> &field;
  const Parameter &parameter;
  const Species &species;
  int32_t n_var = 10;
  int ngg_output = 0;
  MPI_Offset offset_header = 0;
  MPI_Offset *offset_minmax_var = nullptr;
  MPI_Offset *offset_var = nullptr;
  MPI_Offset *offset_sol_time = nullptr;

public:
  explicit FieldIO(int _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
    const Species &spec, int ngg_out);

  void print_field(int step, real time = 0) const;

private:
  void write_header();

  void compute_offset_header();

  void write_common_data_section();

  int32_t acquire_variable_names(std::vector<std::string> &var_name) const;
};

template<MixtureModel mix_model, OutputTimeChoice output_time_choice> FieldIO<
  mix_model, output_time_choice>::FieldIO(int _myid, const Mesh &_mesh, std::vector<Field> &_field,
  const Parameter &_parameter, const Species &spec, int ngg_out):
  myid{_myid}, mesh{_mesh}, field(_field), parameter{_parameter}, species{spec}, ngg_output{ngg_out} {
  const std::filesystem::path out_dir("output");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  write_header();
  compute_offset_header();
  write_common_data_section();
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void FieldIO<mix_model, output_time_choice>::write_header() {
  const std::filesystem::path out_dir("output");
  MPI_File fp;
  // Question: Should I use MPI_MODE_CREATE only here?
  // If a previous simulation has a larger file size than the current one, the way we write to the file is offset,
  // then the larger part would not be accessed anymore, which may result in a waste of memory.
  // In order to avoid this, we should delete the original file if we are conducting a brand-new simulation, and backup the original ones if needed.
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header; thus we let process 0 to write it.

  auto *offset_solution_time = new MPI_Offset[mesh.n_block_total];

  MPI_Offset offset{0};
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
    // 1. FileType: 0=full, 1=grid, 2=solution.
    constexpr int32_t file_type{0};
    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
    offset += 4;
    // 2. Title
    gxl::write_str("Solution file", fp, offset);
    // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
    auto varName = parameter.get_string_array("var_name");
    n_var = static_cast<int>(varName.size());
    // std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
    // n_var = acquire_variable_names(var_name);
    n_var = add_other_variable_name(varName, parameter);
    MPI_File_write_at(fp, offset, &n_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: varName) {
      gxl::write_str(name.c_str(), fp, offset);
    }
    // for (auto &name: var_name) {
    //   gxl::write_str(name.c_str(), fp, offset);
    // }

    // iv. Zones
    for (int i = 0; i < mesh.n_block_total; ++i) {
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
      constexpr int32_t strand_id{-2};
      MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
      offset += 4;
      // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
      offset_solution_time[i] = offset;
      const double solution_time{parameter.get_real("solution_time")};
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
      const auto mx{mesh.mx_blk[i] + 2 * ngg_output}, my{mesh.my_blk[i] + 2 * ngg_output},
          mz{mesh.mz_blk[i] + 2 * ngg_output};
      MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
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

    offset_header = offset;
  }
  offset_sol_time = new MPI_Offset[mesh.n_block];
  auto *disp = new int[mesh.n_proc];
  disp[0] = 0;
  for (int i = 1; i < mesh.n_proc; ++i) {
    disp[i] = disp[i - 1] + mesh.nblk[i - 1];
  }
  MPI_Scatterv(offset_solution_time, mesh.nblk, disp, MPI_OFFSET, offset_sol_time, mesh.n_block, MPI_OFFSET, 0,
               MPI_COMM_WORLD);
  MPI_Bcast(&offset_header, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_var, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void FieldIO<mix_model, output_time_choice>::compute_offset_header() {
  MPI_Offset new_offset{0};
  int i_blk{0};
  for (int p = 0; p < myid; ++p) {
    const int n_blk = mesh.nblk[p];
    for (int b = 0; b < n_blk; ++b) {
      new_offset += 16 + 20 * n_var;
      const int64_t mx{mesh.mx_blk[i_blk] + 2 * ngg_output}, my{mesh.my_blk[i_blk] + 2 * ngg_output},
          mz{mesh.mz_blk[i_blk] + 2 * ngg_output};
      const int64_t N = mx * my * mz;
      // We always write double precision out
      new_offset += n_var * N * 8;
      ++i_blk;
    }
  }
  offset_header += new_offset;
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void FieldIO<mix_model, output_time_choice>::write_common_data_section() {
  const std::filesystem::path out_dir("output");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  const auto n_block{mesh.n_block};
  offset_minmax_var = new MPI_Offset[n_block];
  offset_var = new MPI_Offset[n_block];

  auto offset{offset_header};
  const auto ngg{ngg_output};
  for (int blk = 0; blk < n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_var; ++l) {
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
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};

    double min_val{b.x(-ngg, -ngg, -ngg)}, max_val{b.x(-ngg, -ngg, -ngg)};
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.x(i, j, k));
          max_val = std::max(max_val, b.x(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.y(-ngg, -ngg, -ngg);
    max_val = b.y(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.y(i, j, k));
          max_val = std::max(max_val, b.y(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.z(-ngg, -ngg, -ngg);
    max_val = b.z(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.z(i, j, k));
          max_val = std::max(max_val, b.z(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;

    // Write static variables, which would not change during the simulation, such as, wall distance
    offset = write_static_max_min(offset, v, ngg, fp);

    // Later, the max/min values of flow variables are computed.
    offset_minmax_var[blk] = offset;
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // min_val = v.ov(-ngg, -ngg, -ngg, 0);
    // max_val = v.ov(-ngg, -ngg, -ngg, 0);
    // for (int k = -ngg; k < mz + ngg; ++k) {
    //   for (int j = -ngg; j < my + ngg; ++j) {
    //     for (int i = -ngg; i < mx + ngg; ++i) {
    //       min_val = std::min(min_val, v.ov(i, j, k, 0));
    //       max_val = std::max(max_val, v.ov(i, j, k, 0));
    //     }
    //   }
    // }
    // MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const int n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   min_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   max_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   for (int k = -ngg; k < mz + ngg; ++k) {
    //     for (int j = -ngg; j < my + ngg; ++j) {
    //       for (int i = -ngg; i < mx + ngg; ++i) {
    //         min_val = std::min(min_val, v.ov(i, j, k, 1));
    //         max_val = std::max(max_val, v.ov(i, j, k, 1));
    //       }
    //     }
    //   }
    //   MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    //   MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    // }
    int n_other_var = parameter.get_int("n_other_var");
    for (int l = 0; l < n_other_var; ++l) {
      min_val = v.ov(-ngg, -ngg, -ngg, l);
      max_val = v.ov(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, l));
            max_val = std::max(max_val, v.ov(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // Output the auxiliary variables max/min, which are not computed at the beggining.
    offset = write_dynamic_max_min_first_step(offset, v, ngg, fp);

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const long long memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    int start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
    offset += memsz;
    // Write static variables out. E.g., wall distance
    offset = write_static_array(offset, v, fp, ty, memsz);

    // Later, the variables are outputted.
    offset_var[blk] = offset;
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // auto var = v.ov[0];
    // MPI_File_write_at(fp, offset, var, 1, ty, &status);
    // offset += memsz;
    for (int l = 0; l < n_scalar; ++l) {
      auto var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   var = v.ov[1];
    //   MPI_File_write_at(fp, offset, var, 1, ty, &status);
    //   offset += memsz;
    // }
    for (int l = 0; l < n_other_var; ++l) {
      auto var = v.ov[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // Write the dynamic part of auxiliary variables out.
    offset = write_dynamic_array(offset, v, fp, ty, memsz);

    MPI_Type_free(&ty);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
int32_t
FieldIO<mix_model, output_time_choice>::acquire_variable_names(std::vector<std::string> &var_name) const {
  int32_t nv = 3 + 7; // x,y,z + rho,u,v,w,p,T,Mach
  if constexpr (mix_model != MixtureModel::Air) {
    nv += parameter.get_int("n_spec"); // Y_k
    var_name.resize(nv);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      var_name[ind + 10] = name;
    }
  }
  if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
    nv += n_ps;
    for (int i = 0; i < n_ps; ++i) {
      var_name.emplace_back("PS" + std::to_string(i + 1));
    }
  }
  return nv;
}

template<MixtureModel mix_model, OutputTimeChoice output_time_choice>
void FieldIO<mix_model, output_time_choice>::print_field(int step, real time) const {
  if (myid == 0) {
    std::ofstream file("output/message/step.txt");
    file << step;
    file.close();
  }

  // Copy data from GPU to CPU
  // for (auto &f: field) {
  //   f.copy_data_from_device(parameter);
  // }

  const std::filesystem::path out_dir("output");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // II. Data Section
  // First, modify the new min/max values of the variables
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    MPI_Offset offset{offset_sol_time[blk]};
    MPI_File_write_at(fp, offset, &time, 1, MPI_DOUBLE, &status);

    offset = offset_minmax_var[blk];

    double min_val{0}, max_val{1};
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};
    const auto ngg{ngg_output};
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // min_val = v.ov(-ngg, -ngg, -ngg, 0);
    // max_val = v.ov(-ngg, -ngg, -ngg, 0);
    // for (int k = -ngg; k < mz + ngg; ++k) {
    //   for (int j = -ngg; j < my + ngg; ++j) {
    //     for (int i = -ngg; i < mx + ngg; ++i) {
    //       min_val = std::min(min_val, v.ov(i, j, k, 0));
    //       max_val = std::max(max_val, v.ov(i, j, k, 0));
    //     }
    //   }
    // }
    // MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const int n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   min_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   max_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   for (int k = -ngg; k < mz + ngg; ++k) {
    //     for (int j = -ngg; j < my + ngg; ++j) {
    //       for (int i = -ngg; i < mx + ngg; ++i) {
    //         min_val = std::min(min_val, v.ov(i, j, k, 1));
    //         max_val = std::max(max_val, v.ov(i, j, k, 1));
    //       }
    //     }
    //   }
    //   MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    //   MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    // }
    const int n_other_var = parameter.get_int("n_other_var");
    for (int l = 0; l < n_other_var; ++l) {
      min_val = v.ov(-ngg, -ngg, -ngg, l);
      max_val = v.ov(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, l));
            max_val = std::max(max_val, v.ov(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // Write the max/min pair of additional variables
    write_dynamic_max_min(offset, v, ngg, fp);

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const long long memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    int start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    offset = offset_var[blk];
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // auto var = v.ov[0];
    // MPI_File_write_at(fp, offset, var, 1, ty, &status);
    // offset += memsz;
    for (int l = 0; l < n_scalar; ++l) {
      auto var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   var = v.ov[1];
    //   MPI_File_write_at(fp, offset, var, 1, ty, &status);
    //   offset += memsz;
    // }
    for (int l = 0; l < n_other_var; ++l) {
      auto var = v.ov[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // Write the dynamic part of auxiliary variables out.
    /*offset = */
    write_dynamic_array(offset, v, fp, ty, memsz);
    MPI_Type_free(&ty);
  }
  MPI_File_close(&fp);
}

template<MixtureModel mix_model>
class FieldIO<mix_model, OutputTimeChoice::TimeSeries> {
  const int myid{0};
  const Mesh &mesh;
  std::vector<Field> &field;
  const Parameter &parameter;
  const Species &species;
  int32_t n_var = 10;
  std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
  int ngg_output = 0;
  MPI_Offset offset_zones = 0;
  int zone_id = 0;
  double *x_min = nullptr, *x_max = nullptr, *y_min = nullptr, *y_max = nullptr, *z_min = nullptr, *z_max = nullptr;
  MPI_Offset offset_header = 0;
  MPI_Offset *offset_minmax_var = nullptr;
  MPI_Offset *offset_var = nullptr;

public:
  explicit FieldIO(int _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
    const Species &spec, int ngg_out);

  void print_field(int step, real time) const;

private:
  void write_header();

  void compute_offset_header();

  void write_common_data_section();

  int32_t acquire_variable_names();
};

template<MixtureModel mix_model>
int32_t FieldIO<mix_model, OutputTimeChoice::TimeSeries>::acquire_variable_names() {
  int32_t nv = 3 + 7; // x,y,z + rho,u,v,w,p,T,Mach
  if constexpr (mix_model != MixtureModel::Air) {
    nv += parameter.get_int("n_spec"); // Y_k
    var_name.resize(nv);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      var_name[ind + 10] = name;
    }
  }
  if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
    nv += n_ps;
    for (int i = 0; i < n_ps; ++i) {
      var_name.emplace_back("PS" + std::to_string(i + 1));
    }
  }
  return nv;
}

template<MixtureModel mix_model> FieldIO<mix_model, OutputTimeChoice::TimeSeries>::FieldIO(int _myid,
  const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter, const Species &spec, int ngg_out):
  myid{_myid}, mesh{_mesh}, field(_field), parameter{_parameter}, species{spec}, ngg_output{ngg_out} {
  if (parameter.get_int("output_time_series") == 0) {
    return;
  }
  const std::filesystem::path out_dir("output/time_series");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  write_header();
  compute_offset_header();
  write_common_data_section();
}

template<MixtureModel mix_model>
void FieldIO<mix_model, OutputTimeChoice::TimeSeries>::write_header() {
  const std::filesystem::path out_dir("output/time_series");
  MPI_File fp;
  // Question: Should I use MPI_MODE_CREATE only here?
  // If a previous simulation has a larger file size than the current one, the way we write to the file is offset,
  // then the larger part would not be accessed anymore, which may result in a waste of memory.
  // In order to avoid this, we should delete the original file if we are conducting a brand-new simulation, and backup the original ones if needed.
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield_0.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header; thus we let process 0 to write it.

  MPI_Offset offset{0};
  auto *offset_process = new MPI_Offset[mesh.n_proc];
  auto *zone_number_process = new int[mesh.n_proc];
  zone_number_process[0] = 0;
  var_name = parameter.get_string_array("var_name");
  n_var = static_cast<int>(var_name.size());
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
    // 1. FileType: 0=full, 1=grid, 2=solution.
    constexpr int32_t file_type{0};
    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
    offset += 4;
    // 2. Title
    gxl::write_str("Solution file", fp, offset);
    // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
    //    std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
    MPI_File_write_at(fp, offset, &n_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: var_name) {
      gxl::write_str(name.c_str(), fp, offset);
    }
    offset_process[0] = offset;

    // iv. Zones
    const auto n_blk = mesh.nblk;
    auto *disp = new int[mesh.n_proc];
    disp[0] = n_blk[0];
    for (int i = 1; i < mesh.n_proc; ++i) {
      disp[i] = disp[i - 1] + n_blk[i];
    }
    int pid{0};
    for (int i = 0; i < mesh.n_block_total; ++i) {
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
      constexpr int32_t strand_id{-2};
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
      const auto mx{mesh.mx_blk[i] + 2 * ngg_output}, my{mesh.my_blk[i] + 2 * ngg_output},
          mz{mesh.mz_blk[i] + 2 * ngg_output};
      MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
      offset += 4;

      // 11. For all zone types (repeat for each Auxiliary data name/value pair)
      // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
      // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
      // No more data
      constexpr int32_t no_more_auxi_data{0};
      MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
      offset += 4;

      // See which process this block belongs to.
      if (i == disp[pid] - 1) {
        pid++;
        offset_process[pid] = offset;
        if (pid < mesh.n_proc - 1) {
          zone_number_process[pid] = i + 1;
        }
      }
    }


    // End of Header
    constexpr float EOHMARKER{357.0f};
    MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
    offset += 4;

    offset_header = offset;
  }
  MPI_Scatter(offset_process, 1, MPI_OFFSET, &offset_zones, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
  MPI_Scatter(zone_number_process, 1, MPI_INT, &zone_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&offset_header, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&n_var, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model>
void FieldIO<mix_model, OutputTimeChoice::TimeSeries>::compute_offset_header() {
  MPI_Offset new_offset{0};
  int i_blk{0};
  for (int p = 0; p < myid; ++p) {
    const int n_blk = mesh.nblk[p];
    for (int b = 0; b < n_blk; ++b) {
      new_offset += 16 + 20 * n_var;
      const int64_t mx{mesh.mx_blk[i_blk] + 2 * ngg_output}, my{mesh.my_blk[i_blk] + 2 * ngg_output},
          mz{mesh.mz_blk[i_blk] + 2 * ngg_output};
      const int64_t N = mx * my * mz;
      // We always write double precision out
      new_offset += n_var * N * 8;
      ++i_blk;
    }
  }
  offset_header += new_offset;
}

template<MixtureModel mix_model>
void FieldIO<mix_model, OutputTimeChoice::TimeSeries>::write_common_data_section() {
  const std::filesystem::path out_dir("output/time_series");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield_0.plt").c_str(), MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  const auto n_block{mesh.n_block};
  offset_minmax_var = new MPI_Offset[n_block];
  offset_var = new MPI_Offset[n_block];
  x_min = new double[n_block], x_max = new double[n_block], y_min = new double[n_block], y_max = new double[n_block],
      z_min = new double[n_block], z_max = new double[n_block];

  auto offset{offset_header};
  const auto ngg{ngg_output};
  for (int blk = 0; blk < n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_var; ++l) {
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
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
    offset += 4;

    offset_minmax_var[blk] = offset;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};

    double min_val{b.x(-ngg, -ngg, -ngg)}, max_val{b.x(-ngg, -ngg, -ngg)};
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.x(i, j, k));
          max_val = std::max(max_val, b.x(i, j, k));
        }
      }
    }
    x_min[blk] = min_val;
    x_max[blk] = max_val;
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.y(-ngg, -ngg, -ngg);
    max_val = b.y(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.y(i, j, k));
          max_val = std::max(max_val, b.y(i, j, k));
        }
      }
    }
    y_min[blk] = min_val;
    y_max[blk] = max_val;
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.z(-ngg, -ngg, -ngg);
    max_val = b.z(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.z(i, j, k));
          max_val = std::max(max_val, b.z(i, j, k));
        }
      }
    }
    z_min[blk] = min_val;
    z_max[blk] = max_val;
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    // Later, the max/min values of flow variables are computed.
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // min_val = v.ov(-ngg, -ngg, -ngg, 0);
    // max_val = v.ov(-ngg, -ngg, -ngg, 0);
    // for (int k = -ngg; k < mz + ngg; ++k) {
    //   for (int j = -ngg; j < my + ngg; ++j) {
    //     for (int i = -ngg; i < mx + ngg; ++i) {
    //       min_val = std::min(min_val, v.ov(i, j, k, 0));
    //       max_val = std::max(max_val, v.ov(i, j, k, 0));
    //     }
    //   }
    // }
    // MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const int n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   min_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   max_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   for (int k = -ngg; k < mz + ngg; ++k) {
    //     for (int j = -ngg; j < my + ngg; ++j) {
    //       for (int i = -ngg; i < mx + ngg; ++i) {
    //         min_val = std::min(min_val, v.ov(i, j, k, 1));
    //         max_val = std::max(max_val, v.ov(i, j, k, 1));
    //       }
    //     }
    //   }
    //   MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    //   MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    // }
    int n_other_var = parameter.get_int("n_other_var");
    for (int l = 0; l < n_other_var; ++l) {
      min_val = v.ov(-ngg, -ngg, -ngg, l);
      max_val = v.ov(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, l));
            max_val = std::max(max_val, v.ov(i, j, k, l));
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
    int lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const long long memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    int start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
    offset += memsz;
    // Later, the variables are outputted.
    offset_var[blk] = offset;
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // auto var = v.ov[0];
    // MPI_File_write_at(fp, offset, var, 1, ty, &status);
    // offset += memsz;
    for (int l = 0; l < n_scalar; ++l) {
      auto var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   var = v.ov[1];
    //   MPI_File_write_at(fp, offset, var, 1, ty, &status);
    //   offset += memsz;
    // }
    for (int l = 0; l < n_other_var; ++l) {
      auto var = v.ov[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model>
void FieldIO<mix_model, OutputTimeChoice::TimeSeries>::print_field(int step, real time) const {
  // if (step % parameter.get_int("output_file") != 0) {
  //   // The data has not been updated.
  //   // Copy data from GPU to CPU
  //   for (auto &f: field) {
  //     f.copy_data_from_device(parameter);
  //   }
  // }

  const std::filesystem::path out_dir("output/time_series");
  MPI_File fp;
  char time_char[11];
  sprintf(time_char, "%9.4e", time);
  std::string time_str{time_char};
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield_" + time_str + "s.plt").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header; thus we let process 0 to write it.

  MPI_Offset offset{0};
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
    // 1. FileType: 0=full, 1=grid, 2=solution.
    constexpr int32_t file_type{0};
    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
    offset += 4;
    // 2. Title
    //    gxl::write_str(("t=" + std::to_string(time) + "s").c_str(), fp, offset);
    gxl::write_str("Solution file", fp, offset); // If the file name is changed, the offset will also be changed.
    // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
    MPI_File_write_at(fp, offset, &n_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: var_name) {
      gxl::write_str(name.c_str(), fp, offset);
    }
  }

  // iv. Zones
  offset = offset_zones;
  for (int i = 0; i < mesh.n_block; ++i) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Zone name.
    gxl::write_str(("zone " + std::to_string(i + zone_id)).c_str(), fp, offset);
    // 3. Parent zone. No longer used
    constexpr int32_t parent_zone{-1};
    MPI_File_write_at(fp, offset, &parent_zone, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
    constexpr int32_t strand_id{-2};
    MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
    offset += 4;
    // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
    //    constexpr double solution_time{0};
    MPI_File_write_at(fp, offset, &time, 1, MPI_DOUBLE, &status);
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
    const auto mx{mesh[i].mx + 2 * ngg_output}, my{mesh[i].my + 2 * ngg_output},
        mz{mesh[i].mz + 2 * ngg_output};
    MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
    offset += 4;

    // 11. For all zone types (repeat for each Auxiliary data name/value pair)
    // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
    // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
    // No more data
    constexpr int32_t no_more_auxi_data{0};
    MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
    offset += 4;
  }

  if (myid == 0) {
    // End of Header
    constexpr float EOHMARKER{357.0f};
    offset = offset_header - 4;
    MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
    offset += 4;
  }

  // II. Data Section
  // First, modify the new min/max values of the variables
  offset = offset_header;
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_var; ++l) {
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
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
    offset += 4;
    //    offset = offset_minmax_var[blk];
    // x min and x max
    MPI_File_write_at(fp, offset, &x_min[blk], 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &x_max[blk], 1, MPI_DOUBLE, &status);
    offset += 8;
    // y min and y max
    MPI_File_write_at(fp, offset, &y_min[blk], 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &y_max[blk], 1, MPI_DOUBLE, &status);
    offset += 8;
    // z min and z max
    MPI_File_write_at(fp, offset, &z_min[blk], 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &z_max[blk], 1, MPI_DOUBLE, &status);
    offset += 8;

    double min_val{0}, max_val{1};
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};
    const auto ngg{ngg_output};
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // min_val = v.ov(-ngg, -ngg, -ngg, 0);
    // max_val = v.ov(-ngg, -ngg, -ngg, 0);
    // for (int k = -ngg; k < mz + ngg; ++k) {
    //   for (int j = -ngg; j < my + ngg; ++j) {
    //     for (int i = -ngg; i < mx + ngg; ++i) {
    //       min_val = std::min(min_val, v.ov(i, j, k, 0));
    //       max_val = std::max(max_val, v.ov(i, j, k, 0));
    //     }
    //   }
    // }
    // MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    // offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const int n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   min_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   max_val = v.ov(-ngg, -ngg, -ngg, 1);
    //   for (int k = -ngg; k < mz + ngg; ++k) {
    //     for (int j = -ngg; j < my + ngg; ++j) {
    //       for (int i = -ngg; i < mx + ngg; ++i) {
    //         min_val = std::min(min_val, v.ov(i, j, k, 1));
    //         max_val = std::max(max_val, v.ov(i, j, k, 1));
    //       }
    //     }
    //   }
    //   MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    //   MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    //   offset += 8;
    // }
    const int n_other_var = parameter.get_int("n_other_var");
    for (int l = 0; l < n_other_var; ++l) {
      min_val = v.ov(-ngg, -ngg, -ngg, l);
      max_val = v.ov(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, l));
            max_val = std::max(max_val, v.ov(i, j, k, l));
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
    int lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const long long memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    int start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    //    offset = offset_var[blk];
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
    offset += memsz;
    // Later, the variables are outputted.
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // auto var = v.ov[0];
    // MPI_File_write_at(fp, offset, var, 1, ty, &status);
    // offset += memsz;
    for (int l = 0; l < n_scalar; ++l) {
      auto var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    // if constexpr (TurbMethod<turb>::hasMut) {
    //   var = v.ov[1];
    //   MPI_File_write_at(fp, offset, var, 1, ty, &status);
    //   offset += memsz;
    // }
    for (int l = 0; l < n_other_var; ++l) {
      auto var = v.ov[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);
}
} // cfd

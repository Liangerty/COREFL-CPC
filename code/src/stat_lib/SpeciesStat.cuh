#pragma once

#include <array>
#include <vector>
#include "../Parameter.h"
#include "../Mesh.h"
#include "../Field.h"
#include <mpi.h>
#include <filesystem>
#include "../gxl_lib/MyString.h"

namespace cfd {
template<typename T> std::vector<int> read_species_collect_file(Parameter &parameter, const Mesh &mesh,
  int n_block_ahead, std::vector<Field> &field);

template<typename T>
MPI_Offset
create_species_collect_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead);

template<typename T>
void export_species_collect_file(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  MPI_Offset offset_ahead, std::vector<int> &counter, const std::vector<MPI_Datatype> &tys);

struct ScalarDissipationRate {
  constexpr static int n_collect = 5;
  static constexpr std::array<std::string_view, n_collect> collect_name = {
    "rhoDdYDxdYDx", "rhoDdYDx", "rhoDdYDy", "rhoDdYDz", "rhoD"
  };
  constexpr static int ngg = 0;
  constexpr static std::string_view file_name = "species_dissipation_rate";

  static void read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count, MPI_Datatype ty,
    MPI_Status *status);

  static void copy_to_device(Field &zone, int nv, long long sz);

  static void copy_to_host(Field &zone, int nv, long long sz);

  static void
  write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty, MPI_Status *status);
};

__device__ void collect_species_dissipation_rate(DZone *zone, const DParameter *param, int i, int j, int k);

struct ScalarFlucBudget {
  constexpr static int n_collect = 29;
  static constexpr std::array<std::string_view, n_collect> collect_name = {
    "rhoUZ", "rhoVZ", "rhoWZ", "rhoUZZ", "rhoVZZ", "rhoWZZ", "rhoDGradZGradZ", "rhoDdZDx", "rhoDdZDy", "rhoDdZDz",
    "rhoD", "rhoDZdZdx", "rhoDZdZdy", "rhoDZdZdz", "rhoUUZ", "rhoUVZ", "rhoUWZ", "rhoVVZ", "rhoVWZ", "rhoWWZ", "ZdPdx",
    "ZdPdy", "ZdPdz", "ZGradTau1j", "ZGradTau2j", "ZGradTau3j", "uDiffuseFlux", "vDiffuseFlux", "wDiffuseFlux"
  };
  constexpr static int ngg = 1;
  constexpr static std::string_view file_name = "scalar_fluc_budget";

  static void read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count, MPI_Datatype ty,
    MPI_Status *status);

  static void copy_to_device(Field &zone, int nv, long long sz);

  static void copy_to_host(Field &zone, int nv, long long sz);

  static void
  write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty, MPI_Status *status);
};

// Please make sure that tke budget is collected before this function is called
__device__ void collect_scalar_fluc_budget(DZone *zone, const DParameter *param, int i, int j, int k);

struct ScalarVelocityCorrelation {
  constexpr static int n_collect = 3;
  static constexpr std::array<std::string_view, n_collect> collect_name = {"rhoUY", "rhoVY", "rhoWY"};
  constexpr static int ngg = 0;
  constexpr static std::string_view file_name = "species_velocity_correlation";

  static void read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count, MPI_Datatype ty,
    MPI_Status *status);

  static void copy_to_device(Field &zone, int nv, long long sz);

  static void copy_to_host(Field &zone, int nv, long long sz);

  static void
  write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty, MPI_Status *status);
};

__device__ void collect_species_velocity_correlation(DZone *zone, const DParameter *param, int i, int j, int k);
}

template<typename T> std::vector<int> cfd::read_species_collect_file(Parameter &parameter, const Mesh &mesh,
  int n_block_ahead, std::vector<Field> &field) {
  const std::filesystem::path out_dir("output/stat/");
  std::string fileName{T::file_name};
  std::string file_name = out_dir.string() + fileName + ".bin";

  const int n_species_stat = parameter.get_int("n_species_stat");
  const int n_ps = parameter.get_int("n_ps");
  const int n_var_stat = n_species_stat + n_ps;

  std::vector<int> counter_read(T::n_collect * n_var_stat, 0);
  if (!std::filesystem::exists(file_name)) {
    printf("File %s does not exist. The data are collected from now\n", file_name.c_str());
  } else {
    MPI_File fp;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
    MPI_Status status;

    constexpr int n_collect = T::n_collect;
    constexpr int ngg = T::ngg;
    std::vector<std::string> collectName(n_collect * n_var_stat);
    for (int i = 0; i < n_species_stat; ++i) {
      auto name = parameter.get_string_array("stat_species")[i];
      for (int j = 0; j < n_collect; ++j) {
        std::string collNameJ{T::collect_name[j]};
        collectName.emplace_back(collNameJ + name);
      }
    }
    for (int i = 0; i < n_ps; ++i) {
      auto name = "PS" + std::to_string(i + 1);
      for (int j = 0; j < n_collect; ++j) {
        std::string collNameJ{T::collect_name[j]};
        collectName.emplace_back(collNameJ + name);
      }
    }

    MPI_Offset offset_read;
    int nBlock = 0;
    MPI_File_read_at(fp, 0, &nBlock, 1, MPI_INT32_T, &status);
    if (nBlock != mesh.n_block_total) {
      printf("Error: The number of blocks in %s is not consistent with the current mesh.\n", file_name.c_str());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nv_read = 0;
    MPI_File_read_at(fp, 4, &nv_read, 1, MPI_INT32_T, &status);
    int ngg_read = 0;
    MPI_File_read_at(fp, 8, &ngg_read, 1, MPI_INT32_T, &status);
    if (ngg_read != T::ngg) {
      printf("Error: The number of ghost cells in %s is not %d.\n", file_name.c_str(), T::ngg);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<std::string> varNameRead(nv_read);
    offset_read = 4 * 3;
    for (int l = 0; l < nv_read; ++l) {
      varNameRead[l] = gxl::read_str_from_binary_MPI_ver(fp, offset_read);
    }
    std::vector<int> counter_read_temp(nv_read);
    MPI_File_read_at(fp, offset_read, counter_read_temp.data(), nv_read, MPI_INT32_T, &status);
    offset_read += 4 * nv_read;
    std::vector<int> readVarIndex(nv_read, -1);
    for (int l = 0; l < nv_read; ++l) {
      for (int i = 0; i < n_collect; ++i) {
        if (varNameRead[l] == collectName[i]) {
          readVarIndex[l] = i;
          counter_read[i] = counter_read_temp[l];
          break;
        }
      }
    }

    for (int b = 0; b < n_block_ahead; ++b) {
      MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
      offset_read += sz * nv_read + 4 * 3;
    }

    for (int b = 0; b < mesh.n_block; ++b) {
      int mx, my, mz;
      MPI_File_read_at(fp, offset_read, &mx, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp, offset_read, &my, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp, offset_read, &mz, 1, MPI_INT32_T, &status);
      offset_read += 4;
      if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
        printf("Error: The mesh size in %s is not consistent with the current mesh.\n", file_name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;
      MPI_Datatype ty;
      int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
      int start_idx[3]{0, 0, 0};
      MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);

      for (int l = 0; l < nv_read; ++l) {
        int i = readVarIndex[l];
        if (i != -1) {
          T::read(fp, offset_read, field[b], i, 1, ty, &status);
        }
        offset_read += sz;
      }
      T::copy_to_device(field[b], nv_read, sz);
    }
  }

  return counter_read;
}

template<typename T>
MPI_Offset cfd::create_species_collect_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead) {
  const std::filesystem::path out_dir("output/stat/");
  MPI_File fp;
  const std::string fileName{T::file_name};
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + fileName + ".bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
  MPI_Status status;
  MPI_Offset offset{0};

  const int myid = parameter.get_int("myid");
  const int n_scalar_stat = parameter.get_int("n_species_stat") + parameter.get_int("n_ps");
  const int n_collect = T::n_collect * n_scalar_stat;
  constexpr int ngg = T::ngg;
  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    // collect reynolds 1st order statistics
    MPI_File_write_at(fp, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 4, &n_collect, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 8, &ngg, 1, MPI_INT32_T, &status);
    offset = 4 * 3;
    if (parameter.get_int("n_species_stat") > 0) {
      for (auto &n: parameter.get_string_array("stat_species")) {
        for (auto var: T::collect_name) {
          std::string name{var};
          name += n;
          gxl::write_str(name.data(), fp, offset);
        }
      }
    }
    for (int i = 0; i < parameter.get_int("n_ps"); ++i) {
      for (auto var: T::collect_name) {
        std::string name{var};
        name += "PS" + std::to_string(i + 1);
        gxl::write_str(name.data(), fp, offset);
      }
    }
  }
  MPI_Bcast(&offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

  if (myid != 0) {
    offset += 4 * n_collect; // counter
  }
  for (int b = 0; b < n_block_ahead; ++b) {
    const MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
    offset += sz * n_collect + 4 * 3;
  }

  return offset;
}

template<typename T> void cfd::export_species_collect_file(Parameter &parameter, const Mesh &mesh,
  std::vector<Field> &field, MPI_Offset offset_ahead, std::vector<int> &counter, const std::vector<MPI_Datatype> &tys) {
  const std::filesystem::path out_dir("output/stat/");
  const std::string fileName{T::file_name};
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + fileName + ".bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
  MPI_Status status;

  MPI_Offset offset = offset_ahead;
  const int n_species_stat = parameter.get_int("n_species_stat") + parameter.get_int("n_ps");
  const int n_collect = T::n_collect * n_species_stat;
  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    MPI_File_write_at(fp, offset, counter.data(), n_collect, MPI_INT32_T, &status);
    offset += 4 * n_collect;
  }
  constexpr int ngg = T::ngg;
  for (int b = 0; b < mesh.n_block; ++b) {
    const int mx = mesh[b].mx, my = mesh[b].my, mz = mesh[b].mz;
    const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;

    T::copy_to_host(field[b], n_collect, sz);

    // We create this datatype because in the original MPI_File_write_at, the number of elements is a 64-bit integer.
    // However, the nx*ny*nz may be larger than 2^31, so we need to use MPI_Type_create_subarray to create a datatype
    MPI_Datatype ty = tys[b];

    MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
    offset += 4;
    T::write(fp, offset, field[b], n_collect, ty, &status);
    offset += sz * n_collect;
  }
}

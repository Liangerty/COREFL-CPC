#include "TkeBudget.cuh"
#include <filesystem>
#include <mpi.h>
#include "../DParameter.cuh"
#include "../gxl_lib/MyString.h"

MPI_Offset cfd::create_tke_budget_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/tke_budget.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
  MPI_Status status;
  MPI_Offset offset{0};

  const int myid = parameter.get_int("myid");
  constexpr int n_collect = TkeBudget::n_collect;
  constexpr int ngg = TkeBudget::ngg;
  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    // collect reynolds 1st order statistics
    MPI_File_write_at(fp, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 4, &n_collect, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 8, &ngg, 1, MPI_INT32_T, &status);
    offset = 4 * 3;
    for (auto var: TkeBudget::collect_name) {
      gxl::write_str(var.data(), fp, offset);
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

std::vector<int> cfd::read_tke_budget_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead,
  std::vector<Field> &field) {
  const std::filesystem::path out_dir("output/stat");
  const std::string file_name = out_dir.string() + "/tke_budget.bin";

  std::vector<int> counter_read(TkeBudget::n_collect, 0);
  if (!std::filesystem::exists(file_name)) {
    printf("File %s does not exist. The Tke budgets are collected from now\n", file_name.c_str());
  } else {
    MPI_File fp_rey1;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey1);
    MPI_Status status;

    constexpr int n_collect = TkeBudget::n_collect;
    constexpr int ngg = TkeBudget::ngg;
    const auto &reyAveVar = TkeBudget::collect_name;

    int nBlock = 0;
    MPI_File_read_at(fp_rey1, 0, &nBlock, 1, MPI_INT32_T, &status);
    if (nBlock != mesh.n_block_total) {
      printf("Error: The number of blocks in %s is not consistent with the current mesh.\n", file_name.c_str());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n_read_rey1 = 0;
    MPI_File_read_at(fp_rey1, 4, &n_read_rey1, 1, MPI_INT32_T, &status);
    int ngg_read = 0;
    MPI_File_read_at(fp_rey1, 8, &ngg_read, 1, MPI_INT32_T, &status);
    if (ngg_read != TkeBudget::ngg) {
      printf("Error: The number of ghost cells in %s is not %d.\n", file_name.c_str(), TkeBudget::ngg);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<std::string> rey1Var(n_read_rey1);
    MPI_Offset offset_read = 4 * 3;
    for (int l = 0; l < n_read_rey1; ++l) {
      rey1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey1, offset_read);
    }
    std::vector<int> counter_read_temp(n_read_rey1);
    MPI_File_read_at(fp_rey1, offset_read, counter_read_temp.data(), n_read_rey1, MPI_INT32_T, &status);
    offset_read += 4 * n_read_rey1;
    std::vector<int> read_rey1_index(n_read_rey1, -1);
    for (int l = 0; l < n_read_rey1; ++l) {
      for (int i = 0; i < n_collect; ++i) {
        if (rey1Var[l] == reyAveVar[i]) {
          read_rey1_index[l] = i;
          counter_read[i] = counter_read_temp[l];
          break;
        }
      }
    }

    for (int b = 0; b < n_block_ahead; ++b) {
      const MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
      offset_read += sz * n_read_rey1 + 4 * 3;
    }

    for (int b = 0; b < mesh.n_block; ++b) {
      int mx, my, mz;
      MPI_File_read_at(fp_rey1, offset_read, &mx, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp_rey1, offset_read, &my, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp_rey1, offset_read, &mz, 1, MPI_INT32_T, &status);
      offset_read += 4;
      if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
        printf("Error: The mesh size in %s is not consistent with the current mesh.\n", file_name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;
      MPI_Datatype ty;
      const int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
      constexpr int start_idx[3]{0, 0, 0};
      MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);

      MPI_File_read_at(fp_rey1, offset_read, field[b].collect_tke_budget.data(), n_collect, ty, &status);
      cudaMemcpy(field[b].h_ptr->collect_tke_budget.data(), field[b].collect_tke_budget.data(), sz * n_collect,
                 cudaMemcpyHostToDevice);
    }
  }

  return counter_read;
}

__device__ void cfd::collect_tke_budget(DZone *zone, DParameter *param, int i, int j, int k) {
  auto &coll = zone->collect_tke_budget;
  auto &bv = zone->bv;

  const auto &metric = zone->metric;

  const real xi_x = metric(i, j, k, 0);
  const real xi_y = metric(i, j, k, 1);
  const real xi_z = metric(i, j, k, 2);
  const real eta_x = metric(i, j, k, 3);
  const real eta_y = metric(i, j, k, 4);
  const real eta_z = metric(i, j, k, 5);
  const real zeta_x = metric(i, j, k, 6);
  const real zeta_y = metric(i, j, k, 7);
  const real zeta_z = metric(i, j, k, 8);

  const real u_x = 0.5 * (xi_x * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_x * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_x * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_y = 0.5 * (xi_y * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_y * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_y * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_z = 0.5 * (xi_z * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_z * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_z * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));

  const real mu = zone->mul(i, j, k);
  const real tau11 = mu * (4.0 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau12 = mu * (u_y + v_x);
  const real tau13 = mu * (u_z + w_x);
  const real tau22 = mu * (4.0 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau23 = mu * (v_z + w_y);
  const real tau33 = mu * (4.0 * w_z - 2 * u_x - 2 * v_y) / 3.0;

  // Here, the viscous stresses are directly averaged instead of collecting the sum and then averaging in post-processing.
  const int count = param->statViscousStressAddCount;
  const real oneDivCount1 = 1.0 / (count + 1);
  coll(i, j, k, 0) = (coll(i, j, k, 0) * count + tau11) * oneDivCount1;
  coll(i, j, k, 1) = (coll(i, j, k, 1) * count + tau12) * oneDivCount1;
  coll(i, j, k, 2) = (coll(i, j, k, 2) * count + tau13) * oneDivCount1;
  coll(i, j, k, 3) = (coll(i, j, k, 3) * count + tau22) * oneDivCount1;
  coll(i, j, k, 4) = (coll(i, j, k, 4) * count + tau23) * oneDivCount1;
  coll(i, j, k, 5) = (coll(i, j, k, 5) * count + tau33) * oneDivCount1;
  param->statViscousStressAddCount++;

  coll(i, j, k, 6) += tau11 * u_x + tau12 * u_y + tau13 * u_z
      + tau12 * v_x + tau22 * v_y + tau23 * v_z
      + tau13 * w_x + tau23 * w_y + tau33 * w_z;
}

void cfd::export_tke_budget_file(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  MPI_Offset offset_ahead, const std::vector<int> &counter, const std::vector<MPI_Datatype> &tys) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/tke_budget.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
  MPI_Status status;

  MPI_Offset offset = offset_ahead;
  constexpr int n_collect = TkeBudget::n_collect;
  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    MPI_File_write_at(fp, offset, counter.data(), n_collect, MPI_INT32_T, &status);
    offset += 4 * n_collect;
  }
  constexpr int ngg = TkeBudget::ngg;
  for (int b = 0; b < mesh.n_block; ++b) {
    const auto &zone = field[b].h_ptr;
    const int mx = mesh[b].mx, my = mesh[b].my, mz = mesh[b].mz;
    const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;

    cudaMemcpy(field[b].collect_tke_budget.data(), zone->collect_tke_budget.data(), sz * n_collect,
               cudaMemcpyDeviceToHost);

    // We create this datatype because in the original MPI_File_write_at, the number of elements is a 64-bit integer.
    // However, the nx*ny*nz may be larger than 2^31, so we need to use MPI_Type_create_subarray to create a datatype
    const MPI_Datatype ty = tys[b];

    MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp, offset, field[b].collect_tke_budget.data(), n_collect, ty, &status);
    offset += sz * n_collect;
  }
  MPI_File_close(&fp);
}

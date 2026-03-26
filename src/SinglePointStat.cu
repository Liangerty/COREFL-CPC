#include "SinglePointStat.cuh"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include "ChemData.h"
#include "MixingLayer.cuh"
#include "gxl_lib/MyAlgorithm.h"
#include <numeric>
#include "stat_lib/TkeBudget.cuh"
#include "stat_lib/SpeciesStat.cuh"
#include "DParameter.cuh"

namespace cfd {
SinglePointStat::SinglePointStat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field,
  const Species &_species) :
  parameter(_parameter), mesh(_mesh), field(_field), species(_species) {
  const bool collect_this_time{_parameter.get_bool("if_collect_statistics")};
  myid = _parameter.get_int("myid");
  if_collect_2nd_moments = parameter.get_bool("if_collect_2nd_moments");
  if (!if_collect_2nd_moments) {
    n_rey2nd = 0;
    n_fav2nd = 0;
    rey2ndVar = {};
    fav2ndVar = {};
    parameter.update_parameter("rho_p_correlation", false);
  }
  if_collect_spec_favreAvg = parameter.get_bool("if_collect_spec_favreAvg");
  perform_spanwise_average = parameter.get_bool("perform_spanwise_average");
  if (!perform_spanwise_average) {
    scalar_fluc_budget = parameter.get_bool("stat_scalar_fluc_budget");
    species_velocity_correlation = parameter.get_bool("stat_species_velocity_correlation");
    species_dissipation_rate = parameter.get_bool("stat_species_dissipation_rate");
    tke_budget = parameter.get_bool("stat_tke_budget");
  }

  if (collect_this_time) {
    const std::filesystem::path out_dir("output/stat");
    if (!exists(out_dir)) {
      create_directories(out_dir);
    }

    ty_1gg.resize(mesh.n_block);
    ty_0gg.resize(mesh.n_block);
  }

  // First, identify which species are to be statistically analyzed.
  if (species.n_spec > 0) {
    auto &array = _parameter.get_string_array("stat_species");
    if (gxl::exists<std::string>(array, "all")) {
      species_stat_index.resize(species.n_spec);
      std::iota(species_stat_index.begin(), species_stat_index.end(), 0);
      parameter.update_parameter("stat_species", species.spec_name);
      if (collect_this_time && myid == 0)printf("\tAll species are used to collect statistical data.\n");
    } else {
      if (array.empty()) {
        if (collect_this_time && myid == 0)
          printf("\tInfo: No species is specified for statistical analysis.\n");
      } else {
        for (const auto &spec: array) {
          bool found = false;
          for (const auto &[n, i]: species.spec_list) {
            if (spec == n) {
              species_stat_index.push_back(i);
              found = true;
              break;
            }
          }
          if (!found) {
            if (collect_this_time && myid == 0)
              printf("\tWarning: Species %s is not found in the species list, whose data will not be collected.\n",
                     spec.c_str());
          }
        }
        if (collect_this_time && myid == 0) {
          printf("\tThe following species are used to collect statistical data:\n");
          int counter_spec{0};
          for (const int i: species_stat_index) {
            printf("\t%s\t", species.spec_name[i].c_str());
            ++counter_spec;
            if (counter_spec % 10 == 0) {
              printf("\n");
            }
          }
        }
      }
    }
    n_species_stat = static_cast<int>(species_stat_index.size());
  }
  parameter.update_parameter("n_species_stat", n_species_stat);
  parameter.update_parameter("species_stat_index", species_stat_index);
  n_ps = parameter.get_int("n_ps");

  init_stat_name();
  // update the variables in parameter, these are used to initialize the memory in the field.
  counter_rey1st.resize(n_reyAve, 0);
  parameter.update_parameter("n_stat_reynolds_1st", n_reyAve);
  counter_rey1stScalar.resize(n_reyAveScalar, 0);
  parameter.update_parameter("n_stat_reynolds_1st_scalar", n_reyAveScalar);
  counter_fav1st.resize(n_favAve, 0);
  parameter.update_parameter("n_stat_favre_1st", n_favAve);
  if (if_collect_2nd_moments) {
    counter_rey2nd.resize(n_rey2nd, 0);
    counter_fav2nd.resize(n_fav2nd, 0);
  }
  parameter.update_parameter("n_stat_reynolds_2nd", n_rey2nd);
  parameter.update_parameter("n_stat_favre_2nd", n_fav2nd);
  parameter.update_parameter("reyAveVarIndex", reyAveVarIndex);
  parameter.update_parameter("reyAveScalarIndex", reyAveScalarIndex);

  // other statistics
  if (!perform_spanwise_average) {
    if (n_species_stat > 0 || n_ps > 0) {
      if (scalar_fluc_budget) {
        tke_budget = true; // There are quantities needed here in tke_budget.
        species_velocity_correlation = false;
        species_dissipation_rate = false;
        counter_scalar_fluc_budget.resize(ScalarFlucBudget::n_collect * (n_species_stat + n_ps), 0);
        parameter.update_parameter("stat_tke_budget", true);
        parameter.update_parameter("stat_species_velocity_correlation", false);
        parameter.update_parameter("stat_species_dissipation_rate", false);
      }
      if (species_velocity_correlation) {
        counter_species_velocity_correlation.resize(ScalarVelocityCorrelation::n_collect * (n_species_stat + n_ps), 0);
      }
      if (species_dissipation_rate) {
        counter_species_dissipation_rate.resize(ScalarDissipationRate::n_collect * (n_species_stat + n_ps), 0);
      }
    }
    tke_budget = parameter.get_bool("stat_tke_budget");
  }
  if (tke_budget) {
    counter_tke_budget.resize(TkeBudget::n_collect, 0);
  } else {
    counter_tke_budget.resize(1, 0);
  }
}

void SinglePointStat::init_stat_name() {
  // The default average is the favre one.
  // All variables computed will be Favre averaged.
  const int n_spec = species.n_spec;
  if (n_spec > 0) {
    if (if_collect_spec_favreAvg)
      n_favAve += n_spec;
    if (if_collect_2nd_moments)
      n_fav2nd += n_spec;
    for (int l = 0; l < n_spec; ++l) {
      if (if_collect_spec_favreAvg)
        favAveVar.push_back("rho" + species.spec_name[l]);
      if (if_collect_2nd_moments)
        fav2ndVar.push_back("rho" + species.spec_name[l] + species.spec_name[l]);
    }
  }
  if (n_ps > 0) {
    n_favAve += n_ps;
    if (if_collect_2nd_moments)
      n_fav2nd += n_ps;
    for (int l = 0; l < n_ps; ++l) {
      favAveVar.push_back("rhoPs" + std::to_string(l + 1));
      if (if_collect_2nd_moments)
        fav2ndVar.push_back("rhoPs" + std::to_string(l + 1) + "Ps" + std::to_string(l + 1));
    }
  }

  // Next, see if there are some basic variables except rho, p are to be averaged.
  if (auto &stat_rey_1st = parameter.get_string_array("stat_rey_1st"); !stat_rey_1st.empty()) {
    const int n_rey_1st = static_cast<int>(stat_rey_1st.size());
    for (int i = 0; i < n_rey_1st; ++i) {
      auto var = gxl::to_upper(stat_rey_1st[i]);
      if (var == "U") {
        reyAveVar.emplace_back("u");
        reyAveVarIndex.push_back(1);
        ++n_reyAve;
      } else if (var == "V") {
        reyAveVar.emplace_back("v");
        reyAveVarIndex.push_back(2);
        ++n_reyAve;
      } else if (var == "W") {
        reyAveVar.emplace_back("w");
        reyAveVarIndex.push_back(3);
        ++n_reyAve;
      } else if (var == "T") {
        reyAveVar.emplace_back("T");
        reyAveVarIndex.push_back(5);
        ++n_reyAve;
      } else if (var == "PS") {
        // Here, we assume the ps is not used in RAS models.
        for (int l = 0; l < n_ps; ++l) {
          reyAveScalar.emplace_back("Ps" + std::to_string(l + 1));
          reyAveScalarIndex.push_back(n_spec + l);
          ++n_reyAveScalar;
        }
      } else if (n_spec > 0) {
        bool found = false;
        for (int l = 0; l < n_spec; ++l) {
          if (var == species.spec_name[l]) {
            reyAveScalar.emplace_back(species.spec_name[l]);
            reyAveScalarIndex.push_back(l);
            ++n_reyAveScalar;
            found = true;
            break;
          }
        }
        if (!found) {
          printf("Error: Variable %s for Reynolds average is not supported.\n", var.c_str());
        }
      } else {
        printf("Error: Variable %s for Reynolds average is not supported.\n", var.c_str());
      }
    }
  }
  if (if_collect_2nd_moments && parameter.get_bool("rho_p_correlation")) {
    ++n_rey2nd;
    rey2ndVar.emplace_back("rhoP");
  }
}

void SinglePointStat::initialize_statistics_collector() {
  if (parameter.get_bool("if_continue_collect_statistics")) {
    read_previous_statistical_data();
  }

  compute_offset_for_export_data();

  if (perform_spanwise_average && parameter.get_bool("output_statistics_plt"))
    prepare_for_statistical_data_plot(species);
  //  cudaMalloc(&counter_ud_device, sizeof(int) * UserDefineStat::n_collect);
}

void SinglePointStat::compute_offset_for_export_data() {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav1);
  if (if_collect_2nd_moments) {
    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey2);
    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav2);
  }
  MPI_Status status;
  MPI_Offset offset[4]{0, 0, 0, 0};

  const int n_stat_reynolds_1st = parameter.get_int("n_stat_reynolds_1st") + parameter.get_int(
                                    "n_stat_reynolds_1st_scalar");
  const int n_stat_favre_1st = parameter.get_int("n_stat_favre_1st");
  const int n_stat_reynolds_2nd = parameter.get_int("n_stat_reynolds_2nd");
  const int n_stat_favre_2nd = parameter.get_int("n_stat_favre_2nd");
  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    // collect reynolds 1st order statistics
    MPI_File_write_at(fp_rey1, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey1, 4, &n_stat_reynolds_1st, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_rey1, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[0] = 4 * 3;
    for (const auto &var: reyAveVar) {
      gxl::write_str(var.c_str(), fp_rey1, offset[0]);
    }
    for (const auto &var: reyAveScalar) {
      gxl::write_str(var.c_str(), fp_rey1, offset[0]);
    }

    // collect favre 1st order statistics
    MPI_File_write_at(fp_fav1, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav1, 4, &n_stat_favre_1st, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp_fav1, 8, &ngg, 1, MPI_INT32_T, &status);
    offset[1] = 4 * 3;
    for (const auto &var: favAveVar) {
      gxl::write_str(var.c_str(), fp_fav1, offset[1]);
    }

    // collect reynolds 2nd order statistics
    if (if_collect_2nd_moments) {
      MPI_File_write_at(fp_rey2, 0, &n_block, 1, MPI_INT32_T, &status);
      MPI_File_write_at(fp_rey2, 4, &n_stat_reynolds_2nd, 1, MPI_INT32_T, &status);
      MPI_File_write_at(fp_rey2, 8, &ngg, 1, MPI_INT32_T, &status);
      offset[2] = 4 * 3;
      for (const auto &var: rey2ndVar) {
        gxl::write_str(var.c_str(), fp_rey2, offset[2]);
      }

      // collect favre 2nd order statistics
      MPI_File_write_at(fp_fav2, 0, &n_block, 1, MPI_INT32_T, &status);
      MPI_File_write_at(fp_fav2, 4, &n_stat_favre_2nd, 1, MPI_INT32_T, &status);
      MPI_File_write_at(fp_fav2, 8, &ngg, 1, MPI_INT32_T, &status);
      offset[3] = 4 * 3;
      for (const auto &var: fav2ndVar) {
        gxl::write_str(var.c_str(), fp_fav2, offset[3]);
      }
    }
  }
  MPI_Bcast(offset, 4, MPI_OFFSET, 0, MPI_COMM_WORLD);

  if (myid != 0) {
    offset_unit[0] = offset[0] + 4 * n_stat_reynolds_1st;
    offset_unit[1] = offset[1] + 4 * n_stat_favre_1st;
    if (if_collect_2nd_moments) {
      offset_unit[2] = offset[2] + 4 * n_stat_reynolds_2nd;
      offset_unit[3] = offset[3] + 4 * n_stat_favre_2nd;
    }
  } else {
    // Process 0 needs to write the counter of every variable.
    offset_unit[0] = offset[0];
    offset_unit[1] = offset[1];
    if (if_collect_2nd_moments) {
      offset_unit[2] = offset[2];
      offset_unit[3] = offset[3];
    }
  }

  int n_block_ahead = 0;
  for (int p = 0; p < parameter.get_int("myid"); ++p) {
    n_block_ahead += mesh.nblk[p];
  }
  for (int b = 0; b < n_block_ahead; ++b) {
    int nz = 1 + 2 * ngg;
    if (!perform_spanwise_average) {
      nz = mesh.mz_blk[b] + 2 * ngg;
    }
    const MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * nz * 8;
    offset_unit[0] += sz * n_stat_reynolds_1st + 4 * 3;
    offset_unit[1] += sz * n_stat_favre_1st + 4 * 3;
    if (if_collect_2nd_moments) {
      offset_unit[2] += sz * n_stat_reynolds_2nd + 4 * 3;
      offset_unit[3] += sz * n_stat_favre_2nd + 4 * 3;
    }
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    const int mx = mesh[b].mx, my = mesh[b].my;
    int mz = 1;
    if (!perform_spanwise_average) {
      mz = mesh[b].mz;
    }
    const int nx = mx + 2 * ngg, ny = my + 2 * ngg, nz = mz + 2 * ngg;

    MPI_Datatype ty1, ty0;
    const int lSize[3]{nx, ny, nz};
    constexpr int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty1);
    MPI_Type_commit(&ty1);
    ty_1gg[b] = ty1;

    const int sSize[3]{mx, my, mz};
    MPI_Type_create_subarray(3, sSize, sSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty0);
    MPI_Type_commit(&ty0);
    ty_0gg[b] = ty0;
  }

  if (!perform_spanwise_average) {
    if (tke_budget) {
      offset_tke_budget = create_tke_budget_file(parameter, mesh, n_block_ahead);
    }
    if (n_species_stat > 0 || n_ps > 0) {
      if (scalar_fluc_budget) {
        offset_scalar_fluc_budget = create_species_collect_file<ScalarFlucBudget>(parameter, mesh, n_block_ahead);
      }
      if (species_velocity_correlation) {
        offset_species_velocity_correlation =
            cfd::create_species_collect_file<ScalarVelocityCorrelation>(parameter, mesh, n_block_ahead);
      }
      if (species_dissipation_rate) {
        offset_species_dissipation_rate =
            cfd::create_species_collect_file<ScalarDissipationRate>(parameter, mesh, n_block_ahead);
      }
    }
  }
}

void SinglePointStat::read_previous_statistical_data() {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  // see if the file exists
  if (!(std::filesystem::exists(out_dir.string() + "/coll_rey_1st.bin") &&
        std::filesystem::exists(out_dir.string() + "/coll_fav_1st.bin"))) {
    printf("Previous stat files do not exist, please change the parameter [[if_continue_collect_statistics]] to 0 or "
      "provide previous stat files!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_fav1);
  MPI_Status status;

  MPI_Offset offset_read[4]{0, 0, 0, 0};
  // Reynolds 1st order statistics
  int nBlock = 0;
  MPI_File_read_at(fp_rey1, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_rey_1st.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_rey1 = 0;
  MPI_File_read_at(fp_rey1, 4, &n_read_rey1, 1, MPI_INT32_T, &status);
  int ngg_read = 0;
  MPI_File_read_at(fp_rey1, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_rey_1st.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> rey1Var(n_read_rey1);
  offset_read[0] = 4 * 3;
  for (int l = 0; l < n_read_rey1; ++l) {
    rey1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey1, offset_read[0]);
  }
  std::vector<int> counter_read(n_read_rey1);
  MPI_File_read_at(fp_rey1, offset_read[0], counter_read.data(), n_read_rey1, MPI_INT32_T, &status);
  offset_read[0] += 4 * n_read_rey1;
  std::vector<int> read_rey1_index(n_read_rey1, -1);
  std::vector<int> read_rey1_scalar_index(n_read_rey1, -1);
  for (int l = 0; l < n_read_rey1; ++l) {
    for (int i = 0; i < n_reyAve; ++i) {
      if (rey1Var[l] == reyAveVar[i]) {
        read_rey1_index[l] = i;
        counter_rey1st[i] = counter_read[l];
        break;
      }
    }
    for (int i = 0; i < n_reyAveScalar; ++i) {
      if (rey1Var[l] == reyAveScalar[i]) {
        read_rey1_index[l] = i + n_reyAve;
        counter_rey1stScalar[i] = counter_read[l];
        break;
      }
    }
  }
  // Favre 1st order statistics
  nBlock = 0;
  MPI_File_read_at(fp_fav1, 0, &nBlock, 1, MPI_INT32_T, &status);
  if (nBlock != mesh.n_block_total) {
    printf("Error: The number of blocks in coll_fav_1st.bin is not consistent with the current mesh.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int n_read_fav1 = 0;
  MPI_File_read_at(fp_fav1, 4, &n_read_fav1, 1, MPI_INT32_T, &status);
  ngg_read = 0;
  MPI_File_read_at(fp_fav1, 8, &ngg_read, 1, MPI_INT32_T, &status);
  if (ngg_read != ngg) {
    printf("Error: The number of ghost cells in coll_fav_1st.bin is not consistent with the current simulation.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::vector<std::string> fav1Var(n_read_fav1);
  offset_read[1] = 4 * 3;
  for (int l = 0; l < n_read_fav1; ++l) {
    fav1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_fav1, offset_read[1]);
  }
  counter_read.resize(n_read_fav1);
  MPI_File_read_at(fp_fav1, offset_read[1], counter_read.data(), n_read_fav1, MPI_INT32_T, &status);
  offset_read[1] += 4 * n_read_fav1;
  std::vector<int> read_fav1_index(n_read_fav1, -1);
  for (int l = 0; l < n_read_fav1; ++l) {
    for (int i = 0; i < n_favAve; ++i) {
      if (fav1Var[l] == favAveVar[i]) {
        read_fav1_index[l] = i;
        counter_fav1st[i] = counter_read[l];
        break;
      }
    }
  }

  int n_read_rey2 = 0, n_read_fav2 = 0;
  std::vector<int> read_rey2_index;
  std::vector<int> read_fav2_index;
  if (if_collect_2nd_moments) {
    if (std::filesystem::exists(out_dir.string() + "/coll_rey_2nd.bin") &&
        std::filesystem::exists(out_dir.string() + "/coll_fav_2nd.bin")) {
      MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                    MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey2);
      MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                    MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_fav2);
      // Reynolds 2nd order statistics
      nBlock = 0;
      MPI_File_read_at(fp_rey2, 0, &nBlock, 1, MPI_INT32_T, &status);
      if (nBlock != mesh.n_block_total) {
        printf("Error: The number of blocks in coll_rey_2nd.bin is not consistent with the current mesh.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      MPI_File_read_at(fp_rey2, 4, &n_read_rey2, 1, MPI_INT32_T, &status);
      ngg_read = 0;
      MPI_File_read_at(fp_rey2, 8, &ngg_read, 1, MPI_INT32_T, &status);
      if (ngg_read != ngg) {
        printf("Error: The number of ghost cells in coll_rey_2nd.bin is not consistent with the current simulation.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::vector<std::string> rey2Var(n_read_rey2);
      read_rey2_index.assign(n_read_rey2, -1);
      offset_read[2] = 4 * 3;
      for (int l = 0; l < n_read_rey2; ++l) {
        rey2Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey2, offset_read[2]);
      }
      counter_read.resize(n_read_rey2);
      MPI_File_read_at(fp_rey2, offset_read[2], counter_read.data(), n_read_rey2, MPI_INT32_T, &status);
      offset_read[2] += 4 * n_read_rey2;
      for (int l = 0; l < n_read_rey2; ++l) {
        for (int i = 0; i < n_rey2nd; ++i) {
          if (rey2Var[l] == rey2ndVar[i]) {
            read_rey2_index[l] = i;
            counter_rey2nd[i] = counter_read[l];
            break;
          }
        }
      }
      // Favre 2nd order statistics
      nBlock = 0;
      MPI_File_read_at(fp_fav2, 0, &nBlock, 1, MPI_INT32_T, &status);
      if (nBlock != mesh.n_block_total) {
        printf("Error: The number of blocks in coll_fav_2nd.bin is not consistent with the current mesh.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      MPI_File_read_at(fp_fav2, 4, &n_read_fav2, 1, MPI_INT32_T, &status);
      ngg_read = 0;
      MPI_File_read_at(fp_fav2, 8, &ngg_read, 1, MPI_INT32_T, &status);
      if (ngg_read != ngg) {
        printf("Error: The number of ghost cells in coll_fav_2nd.bin is not consistent with the current simulation.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::vector<std::string> fav2Var(n_read_fav2);
      read_fav2_index.assign(n_read_fav2, -1);
      offset_read[3] = 4 * 3;
      for (int l = 0; l < n_read_fav2; ++l) {
        fav2Var[l] = gxl::read_str_from_binary_MPI_ver(fp_fav2, offset_read[3]);
      }
      counter_read.resize(n_read_fav2);
      MPI_File_read_at(fp_fav2, offset_read[3], counter_read.data(), n_read_fav2, MPI_INT32_T, &status);
      offset_read[3] += 4 * n_read_fav2;
      for (int l = 0; l < n_read_fav2; ++l) {
        for (int i = 0; i < n_fav2nd; ++i) {
          if (fav2Var[l] == fav2ndVar[i]) {
            read_fav2_index[l] = i;
            counter_fav2nd[i] = counter_read[l];
            break;
          }
        }
      }
    }
  }

  int n_block_ahead = 0;
  for (int p = 0; p < parameter.get_int("myid"); ++p) {
    n_block_ahead += mesh.nblk[p];
  }
  for (int b = 0; b < n_block_ahead; ++b) {
    int nz = 1 + 2 * ngg;
    if (!perform_spanwise_average) {
      nz = mesh.mz_blk[b] + 2 * ngg;
    }
    MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * nz * 8;
    offset_read[0] += sz * n_read_rey1 + 4 * 3;
    offset_read[1] += sz * n_read_fav1 + 4 * 3;
    if (if_collect_2nd_moments) {
      offset_read[2] += sz * n_read_rey2 + 4 * 3;
      offset_read[3] += sz * n_read_fav2 + 4 * 3;
    }
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    int mx, my, mz;
    MPI_File_read_at(fp_rey1, offset_read[0], &mx, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    MPI_File_read_at(fp_rey1, offset_read[0], &my, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    MPI_File_read_at(fp_rey1, offset_read[0], &mz, 1, MPI_INT32_T, &status);
    offset_read[0] += 4;
    if (perform_spanwise_average) {
      if (mx != mesh[b].mx || my != mesh[b].my || mz != 1) {
        printf("Error: The mesh size in the previous statistical data is not consistent with the current mesh.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    } else {
      if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
        printf("Error: The mesh size in the previous statistical data is not consistent with the current mesh.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }

    const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;
    MPI_Datatype ty;
    int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    for (int l = 0; l < n_read_rey1; ++l) {
      int i = read_rey1_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_rey1, offset_read[0], field[b].collect_reynolds_1st[i], 1, ty, &status);
      }
      offset_read[0] += sz;
    }
    cudaMemcpy(field[b].h_ptr->collect_reynolds_1st.data(), field[b].collect_reynolds_1st.data(),
               sz * (n_reyAve + n_reyAveScalar), cudaMemcpyHostToDevice);

    offset_read[1] += 4 * 3;
    for (int l = 0; l < n_read_fav1; ++l) {
      int i = read_fav1_index[l];
      if (i != -1) {
        MPI_File_read_at(fp_fav1, offset_read[1], field[b].collect_favre_1st[i], 1, ty, &status);
      }
      offset_read[1] += sz;
    }
    cudaMemcpy(field[b].h_ptr->collect_favre_1st.data(), field[b].collect_favre_1st.data(), sz * n_favAve,
               cudaMemcpyHostToDevice);

    if (if_collect_2nd_moments) {
      offset_read[2] += 4 * 3;
      for (int l = 0; l < n_read_rey2; ++l) {
        int i = read_rey2_index[l];
        if (i != -1) {
          MPI_File_read_at(fp_rey2, offset_read[2], field[b].collect_reynolds_2nd[i], 1, ty, &status);
        }
        offset_read[2] += sz;
      }
      cudaMemcpy(field[b].h_ptr->collect_reynolds_2nd.data(), field[b].collect_reynolds_2nd.data(), sz * n_rey2nd,
                 cudaMemcpyHostToDevice);

      offset_read[3] += 4 * 3;
      for (int l = 0; l < n_read_fav2; ++l) {
        int i = read_fav2_index[l];
        if (i != -1) {
          MPI_File_read_at(fp_fav2, offset_read[3], field[b].collect_favre_2nd[i], 1, ty, &status);
        }
        offset_read[3] += sz;
      }
      cudaMemcpy(field[b].h_ptr->collect_favre_2nd.data(), field[b].collect_favre_2nd.data(), sz * n_fav2nd,
                 cudaMemcpyHostToDevice);
    }
  }

  if (!perform_spanwise_average) {
    if (tke_budget) {
      counter_tke_budget = read_tke_budget_file(parameter, mesh, n_block_ahead, field);
    }
    if (scalar_fluc_budget) {
      counter_scalar_fluc_budget =
          read_species_collect_file<ScalarFlucBudget>(parameter, mesh, n_block_ahead, field);
    }
    if (species_velocity_correlation) {
      counter_species_velocity_correlation =
          cfd::read_species_collect_file<ScalarVelocityCorrelation>(parameter, mesh, n_block_ahead, field);
    }
    if (species_dissipation_rate) {
      counter_species_dissipation_rate =
          read_species_collect_file<ScalarDissipationRate>(parameter, mesh, n_block_ahead, field);
    }
  }
}

void SinglePointStat::prepare_for_statistical_data_plot(const Species &species) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header; thus we let process 0 to write it.

  //  auto *offset_solution_time = new MPI_Offset[mesh.n_block_total];

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
    // 3. Number of variables in the datafile
    std::vector<std::string> var_name{
      "x", "y", "z", "<rho>", "<p>", "{u}", "{v}", "{w}", "{T}", "{u''u''}", "{v''v''}",
      "{w''w''}", "{u''v''}", "{u''w''}", "{v''w''}"
    };
    if (perform_spanwise_average) {
      var_name = {
        "x", "y", "<rho>", "<p>", "{u}", "{v}", "{w}", "{T}"
      };
      n_plot = static_cast<int>(var_name.size());
      auto nv_old = n_plot;
      if (if_collect_spec_favreAvg) {
        n_plot += parameter.get_int("n_spec"); // Y_k
        var_name.resize(n_plot);
        auto &names = species.spec_list;
        for (auto &[name, ind]: names) {
          var_name[ind + nv_old] = "{" + name + "}";
        }
        nv_old = n_plot;
      }
      if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
        n_plot += n_ps; // Y_k'' dissipation rate
        var_name.resize(n_plot);
        for (int i = 0; i < n_ps; ++i) {
          var_name[nv_old + i] = "{Ps" + std::to_string(i + 1) + "}";
        }
        nv_old = n_plot;
      }
      if (if_collect_2nd_moments) {
        var_name.emplace_back("<rho'rho'>");
        var_name.emplace_back("<p'p'>");
        n_plot += 2;
        if (parameter.get_bool("rho_p_correlation")) {
          var_name.emplace_back("<rho'p'>");
          ++n_plot;
        }
        nv_old = n_plot;
        var_name.emplace_back("{u''u''}");
        var_name.emplace_back("{v''v''}");
        var_name.emplace_back("{w''w''}");
        var_name.emplace_back("{u''v''}");
        var_name.emplace_back("{u''w''}");
        var_name.emplace_back("{v''w''}");
        var_name.emplace_back("{T''T''}");
        n_plot += 7;
        nv_old = n_plot;
        if (if_collect_spec_favreAvg) {
          n_plot += parameter.get_int("n_spec"); // Y_k
          var_name.resize(n_plot);
          auto &names = species.spec_list;
          for (auto &[name, ind]: names) {
            var_name[ind + nv_old] = "{" + name + "''" + name + "''}";
          }
          nv_old = n_plot;
        }
        if (const int n_ps = parameter.get_int("n_ps"); n_ps > 0) {
          n_plot += n_ps; // Y_k'' dissipation rate
          var_name.resize(n_plot);
          for (int i = 0; i < n_ps; ++i) {
            var_name[nv_old + i] = "{Ps" + std::to_string(i + 1) + "''" + "Ps" + std::to_string(i + 1) + "''}";
          }
        }
      }
    }

    MPI_File_write_at(fp, offset, &n_plot, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: var_name) {
      gxl::write_str(name.c_str(), fp, offset);
    }

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
      //      offset_solution_time[i] = offset;
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
      const auto mx{mesh.mx_blk[i]}, my{mesh.my_blk[i]};
      auto mz{mesh.mz_blk[i]};
      if (perform_spanwise_average)
        mz = 1;
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
  MPI_Bcast(&offset_header, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_plot, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  MPI_Offset new_offset{0};
  int i_blk{0};
  for (int p = 0; p < myid; ++p) {
    const int n_blk = mesh.nblk[p];
    for (int b = 0; b < n_blk; ++b) {
      new_offset += 16 + 20 * n_plot;
      const int mx{mesh.mx_blk[i_blk]}, my{mesh.my_blk[i_blk]};
      int mz{mesh.mz_blk[i_blk]};
      if (perform_spanwise_average)
        mz = 1;
      const int64_t N = mx * my * mz;
      // We always write double precision out
      new_offset += n_plot * N * 8;
      ++i_blk;
    }
  }
  offset_header += new_offset;

  const auto n_block{mesh.n_block};
  offset_minmax_var = new MPI_Offset[n_block];
  offset_var = new MPI_Offset[n_block];

  offset = offset_header;
  for (int blk = 0; blk < n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_plot; ++l) {
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
    const auto mx{b.mx}, my{b.my};
    auto mz{b.mz};
    if (parameter.get_bool("perform_spanwise_average"))
      mz = 1;

    double min_val{b.x(0, 0, 0)}, max_val{b.x(0, 0, 0)};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          min_val = std::min(min_val, b.x(i, j, k));
          max_val = std::max(max_val, b.x(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.y(0, 0, 0);
    max_val = b.y(0, 0, 0);
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          min_val = std::min(min_val, b.y(i, j, k));
          max_val = std::max(max_val, b.y(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    if (!perform_spanwise_average) {
      min_val = b.z(0, 0, 0);
      max_val = b.z(0, 0, 0);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, b.z(i, j, k));
            max_val = std::max(max_val, b.z(i, j, k));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }

    // Then, the max/min values of variables should be printed.
    offset_minmax_var[blk] = offset;
    if (perform_spanwise_average)
      offset += 16 * (n_plot - 2);
    else
      offset += 16 * (n_plot - 3);

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx, my, mz};

    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, b.mz + 2 * b.ngg};
    int start_idx[3]{b.ngg, b.ngg, b.ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    if (!perform_spanwise_average) {
      MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
      offset += memsz;
    }

    // Then, the variables are outputted.
    offset_var[blk] = offset;
    if (perform_spanwise_average)
      offset += memsz * (n_plot - 2);
    else
      offset += memsz * (n_plot - 3);

    MPI_Type_free(&ty);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_File_close(&fp);
}

void SinglePointStat::plot_statistical_data(DParameter *param) const {
  // First, compute the statistical data.
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }

  if (perform_spanwise_average) {
    for (int b = 0; b < mesh.n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, 1};
      compute_statistical_data_spanwise_average<<<bpg, tpb>>>(field[b].d_ptr, param, counter_fav1st[0]);
      auto sz = mx * my * sizeof(real);
      cudaDeviceSynchronize();
      cudaMemcpy(field[b].stat_reynolds_1st.data(), field[b].h_ptr->stat_reynolds_1st.data(),
                 sz * (parameter.get_int("n_stat_reynolds_1st") + parameter.get_int("n_stat_reynolds_1st_scalar")),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].stat_favre_1st.data(), field[b].h_ptr->stat_favre_1st.data(),
                 sz * parameter.get_int("n_stat_favre_1st"), cudaMemcpyDeviceToHost);
      if (if_collect_2nd_moments) {
        cudaMemcpy(field[b].stat_reynolds_2nd.data(), field[b].h_ptr->stat_reynolds_2nd.data(),
                   sz * parameter.get_int("n_stat_reynolds_2nd"), cudaMemcpyDeviceToHost);
        cudaMemcpy(field[b].stat_favre_2nd.data(), field[b].h_ptr->stat_favre_2nd.data(),
                   sz * parameter.get_int("n_stat_favre_2nd"), cudaMemcpyDeviceToHost);
      }
    }
  } else {
    // for (int b = 0; b < mesh.n_block; ++b) {
    //   const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    //   dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    //   compute_statistical_data<<<bpg, tpb>>>(field[b].d_ptr, param, counter, counter_ud_device);
    //   auto sz = mx * my * mz * sizeof(real);
    //   cudaDeviceSynchronize();
    //   cudaMemcpy(field[b].mean_value.data(), field[b].h_ptr->mean_value.data(), sz * (6 + n_scalar),
    //              cudaMemcpyDeviceToHost);
    //   cudaMemcpy(field[b].reynolds_stress_tensor_and_rms.data(), field[b].h_ptr->reynolds_stress_tensor.data(), sz * 6,
    //              cudaMemcpyDeviceToHost);
    //   cudaMemcpy(field[b].user_defined_statistical_data.data(), field[b].h_ptr->user_defined_statistical_data.data(),
    //              sz * UserDefineStat::n_stat, cudaMemcpyDeviceToHost);
    // }
  }

  if (perform_spanwise_average && parameter.get_int("problem_type") == 1) {
    std::vector<ThicknessRecord> local_thickness;
    int local_x_count = 0;
    for (int blk = 0; blk < mesh.n_block; ++blk) local_x_count += mesh[blk].mx;
    local_thickness.reserve(local_x_count);
    const auto thickness_ref = get_thickness_reference_state();
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      compute_thickness_for_block(blk, thickness_ref, local_thickness);
    }
    write_thickness_file(local_thickness);
  }

  // Next, output them.
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/stat_data.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // II. Data Section
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // First, modify the new min/max values of the variables
    MPI_Offset offset = offset_minmax_var[blk];

    double min_val{0}, max_val{1};
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my};
    auto mz{b.mz};
    if (perform_spanwise_average)
      mz = 1;
    const int ns_plot = if_collect_spec_favreAvg ? species.n_spec : 0;
    const int n_ps = parameter.get_int("n_ps");
    for (int l = 0; l < 2; ++l) {
      min_val = v.stat_reynolds_1st(0, 0, 0, l);
      max_val = v.stat_reynolds_1st(0, 0, 0, l);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.stat_reynolds_1st(i, j, k, l));
            max_val = std::max(max_val, v.stat_reynolds_1st(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    for (int l = 0; l < 4; ++l) {
      min_val = v.stat_favre_1st(0, 0, 0, l);
      max_val = v.stat_favre_1st(0, 0, 0, l);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.stat_favre_1st(i, j, k, l));
            max_val = std::max(max_val, v.stat_favre_1st(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    if (ns_plot > 0) {
      for (int l = 0; l < ns_plot; ++l) {
        min_val = v.stat_favre_1st(0, 0, 0, l + 4);
        max_val = v.stat_favre_1st(0, 0, 0, l + 4);
        for (int k = 0; k < mz; ++k) {
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              min_val = std::min(min_val, v.stat_favre_1st(i, j, k, l + 4));
              max_val = std::max(max_val, v.stat_favre_1st(i, j, k, l + 4));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
    }
    for (int l = 0; l < n_ps; ++l) {
      min_val = v.stat_favre_1st(0, 0, 0, l + 4 + ns_plot);
      max_val = v.stat_favre_1st(0, 0, 0, l + 4 + ns_plot);
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            min_val = std::min(min_val, v.stat_favre_1st(i, j, k, l + 4 + ns_plot));
            max_val = std::max(max_val, v.stat_favre_1st(i, j, k, l + 4 + ns_plot));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    if (if_collect_2nd_moments) {
      for (int l = 0; l < 2; ++l) {
        min_val = v.stat_reynolds_2nd(0, 0, 0, l);
        max_val = v.stat_reynolds_2nd(0, 0, 0, l);
        for (int k = 0; k < mz; ++k) {
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              min_val = std::min(min_val, v.stat_reynolds_2nd(i, j, k, l));
              max_val = std::max(max_val, v.stat_reynolds_2nd(i, j, k, l));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      if (parameter.get_bool("rho_p_correlation")) {
        min_val = v.stat_reynolds_2nd(0, 0, 0, 2);
        max_val = v.stat_reynolds_2nd(0, 0, 0, 2);
        for (int k = 0; k < mz; ++k) {
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              min_val = std::min(min_val, v.stat_reynolds_2nd(i, j, k, 2));
              max_val = std::max(max_val, v.stat_reynolds_2nd(i, j, k, 2));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      for (int l = 0; l < 7; ++l) {
        min_val = v.stat_favre_2nd(0, 0, 0, l);
        max_val = v.stat_favre_2nd(0, 0, 0, l);
        for (int k = 0; k < mz; ++k) {
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              min_val = std::min(min_val, v.stat_favre_2nd(i, j, k, l));
              max_val = std::max(max_val, v.stat_favre_2nd(i, j, k, l));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
      if (ns_plot > 0) {
        for (int l = 0; l < species.n_spec; ++l) {
          min_val = v.stat_favre_2nd(0, 0, 0, l + 7);
          max_val = v.stat_favre_2nd(0, 0, 0, l + 7);
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                min_val = std::min(min_val, v.stat_favre_2nd(i, j, k, l + 7));
                max_val = std::max(max_val, v.stat_favre_2nd(i, j, k, l + 7));
              }
            }
          }
          MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
          offset += 8;
          MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
          offset += 8;
        }
      }
      for (int l = 0; l < n_ps; ++l) {
        min_val = v.stat_favre_2nd(0, 0, 0, l + 7 + ns_plot);
        max_val = v.stat_favre_2nd(0, 0, 0, l + 7 + ns_plot);
        for (int k = 0; k < mz; ++k) {
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              min_val = std::min(min_val, v.stat_favre_2nd(i, j, k, l + 7 + ns_plot));
              max_val = std::max(max_val, v.stat_favre_2nd(i, j, k, l + 7 + ns_plot));
            }
          }
        }
        MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
        offset += 8;
        MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
        offset += 8;
      }
    }

    // 7. Zone Data.
    MPI_Datatype ty;
    int lsize[3]{mx, my, mz};
    const int64_t memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    int start_idx[3]{0, 0, 0};
    MPI_Type_create_subarray(3, lsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    offset = offset_var[blk];
    for (int l = 0; l < 2; ++l) {
      auto var = v.stat_reynolds_1st[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    for (int l = 0; l < 4; ++l) {
      auto var = v.stat_favre_1st[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    if (ns_plot > 0) {
      for (int l = 0; l < ns_plot; ++l) {
        auto var = v.stat_favre_1st[l + 4];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += memsz;
      }
    }
    for (int l = 0; l < n_ps; ++l) {
      auto var = v.stat_favre_1st[l + 4 + ns_plot];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    if (if_collect_2nd_moments) {
      for (int l = 0; l < 2; ++l) {
        auto var = v.stat_reynolds_2nd[l];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += memsz;
      }
      if (parameter.get_bool("rho_p_correlation")) {
        auto var = v.stat_reynolds_2nd[2];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += memsz;
      }
      for (int l = 0; l < 7; ++l) {
        auto var = v.stat_favre_2nd[l];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += memsz;
      }
      if (ns_plot > 0) {
        for (int l = 0; l < ns_plot; ++l) {
          auto var = v.stat_favre_2nd[l + 7];
          MPI_File_write_at(fp, offset, var, 1, ty, &status);
          offset += memsz;
        }
      }
      for (int l = 0; l < n_ps; ++l) {
        auto var = v.stat_favre_2nd[l + 7 + ns_plot];
        MPI_File_write_at(fp, offset, var, 1, ty, &status);
        offset += memsz;
      }
    }
    MPI_Type_free(&ty);
  }
  MPI_File_close(&fp);
}

SinglePointStat::ThicknessReferenceState SinglePointStat::get_thickness_reference_state() const {
  ThicknessReferenceState ref{};
  std::vector<real> var_info;
  get_mixing_layer_info(parameter, species, var_info);

  const int ns = species.n_spec;
  ref.u1 = std::sqrt(var_info[1] * var_info[1] + var_info[2] * var_info[2] + var_info[3] * var_info[3]);
  ref.u2 = std::sqrt(var_info[8 + ns] * var_info[8 + ns] + var_info[9 + ns] * var_info[9 + ns] +
                     var_info[10 + ns] * var_info[10 + ns]);
  ref.rho_ref = 0.5 * (var_info[0] + var_info[7 + ns]);
  ref.convective_velocity = parameter.get_real("convective_velocity");
  ref.delta_u = std::abs(ref.u1 - ref.u2);
  ref.upper_u_faster = ref.u1 >= ref.u2;
  if (ref.upper_u_faster) {
    ref.u11 = ref.u1 - 0.1 * ref.delta_u;
    ref.u22 = ref.u2 + 0.1 * ref.delta_u;
  } else {
    ref.u11 = ref.u1 + 0.1 * ref.delta_u;
    ref.u22 = ref.u2 - 0.1 * ref.delta_u;
  }
  return ref;
}

void SinglePointStat::compute_thickness_for_block(int blk, const ThicknessReferenceState &ref,
                                                  std::vector<ThicknessRecord> &records) const {
  const auto &block = mesh[blk];
  const auto &stat = field[blk];
  const int mx = block.mx;
  const int my = block.my;
  const real nan = std::numeric_limits<real>::quiet_NaN();

  for (int i = 0; i < mx; ++i) {
    ThicknessRecord rec{};
    rec.x = block.x(i, 0, 0);
    rec.yc = nan;
    rec.delta_theta = nan;
    rec.delta_omega = nan;
    rec.delta_vis = nan;

    if (my < 3 || ref.delta_u <= 0 || ref.rho_ref <= 0) {
      records.push_back(rec);
      continue;
    }

    real sum1 = 0;
    real max1 = 0;
    int yc_1 = -1, yc_2 = -1;
    real y_u1 = -1e6, y_u2 = -1e6;

    for (int j = 1; j < my - 1; ++j) {
      const real y_last = block.y(i, j - 1, 0);
      const real y_next = block.y(i, j + 1, 0);
      const real y_this = block.y(i, j, 0);
      const real dy = std::abs(0.5 * (y_next - y_last));
      const real rho = stat.stat_reynolds_1st(i, j, 0, 0);
      const real u_loc = stat.stat_favre_1st(i, j, 0, 0);
      const real u_prev = stat.stat_favre_1st(i, j - 1, 0, 0);
      const real u_next = stat.stat_favre_1st(i, j + 1, 0, 0);

      sum1 += rho * (ref.u1 - u_loc) * (u_loc - ref.u2) * dy;

      const real denom = y_next - y_last;
      if (std::abs(denom) > 0) {
        const real dudy = std::abs((u_next - u_prev) / denom);
        if (dudy > max1) max1 = dudy;
      }

      if (ref.upper_u_faster) {
        if (u_loc > ref.convective_velocity && yc_1 == -1) {
          yc_1 = j - 1;
          yc_2 = j;
        }
        if (u_loc > ref.u11 && y_u1 < -1e5) {
          y_u1 = (ref.u11 - u_prev) / (u_loc - u_prev) * (y_this - y_last) + y_last;
        }
        if (u_loc > ref.u22 && y_u2 < -1e5) {
          y_u2 = (ref.u22 - u_prev) / (u_loc - u_prev) * (y_this - y_last) + y_last;
        }
      } else {
        if (u_loc < ref.convective_velocity && yc_1 == -1) {
          yc_1 = j - 1;
          yc_2 = j;
        }
        if (u_loc < ref.u11 && y_u1 < -1e5) {
          y_u1 = (ref.u11 - u_prev) / (u_loc - u_prev) * (y_this - y_last) + y_last;
        }
        if (u_loc < ref.u22 && y_u2 < -1e5) {
          y_u2 = (ref.u22 - u_prev) / (u_loc - u_prev) * (y_this - y_last) + y_last;
        }
      }
    }

    rec.delta_theta = sum1 / (ref.rho_ref * ref.delta_u * ref.delta_u);
    if (max1 > 0) rec.delta_omega = ref.delta_u / max1;
    if (yc_1 != -1 && yc_2 != -1) {
      const real y1 = block.y(i, yc_1, 0);
      const real y2 = block.y(i, yc_2, 0);
      const real u1_loc = stat.stat_favre_1st(i, yc_1, 0, 0);
      const real u2_loc = stat.stat_favre_1st(i, yc_2, 0, 0);
      rec.yc = (ref.convective_velocity - u1_loc) / (u2_loc - u1_loc) * (y2 - y1) + y1;
    }
    rec.delta_vis = std::abs(y_u2 - y_u1);
    records.push_back(rec);
  }
}

void SinglePointStat::write_thickness_file(const std::vector<ThicknessRecord> &local_records) const {
  constexpr int n_value = 5;
  std::vector<real> local_buffer;
  local_buffer.reserve(local_records.size() * n_value);
  for (const auto &rec: local_records) {
    local_buffer.push_back(rec.x);
    local_buffer.push_back(rec.yc);
    local_buffer.push_back(rec.delta_theta);
    local_buffer.push_back(rec.delta_omega);
    local_buffer.push_back(rec.delta_vis);
  }

  const int send_count = static_cast<int>(local_buffer.size());
  std::vector<int> recv_counts;
  if (myid == 0) recv_counts.resize(parameter.get_int("n_proc"));
  MPI_Gather(&send_count, 1, MPI_INT, myid == 0 ? recv_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs;
  int total_count = 0;
  if (myid == 0) {
    displs.resize(recv_counts.size(), 0);
    for (size_t i = 1; i < recv_counts.size(); ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    total_count = displs.empty() ? 0 : displs.back() + recv_counts.back();
  }

  std::vector<real> recv_buffer;
  if (myid == 0) recv_buffer.resize(total_count);
  MPI_Gatherv(local_buffer.data(), send_count, MPI_DOUBLE, myid == 0 ? recv_buffer.data() : nullptr,
              myid == 0 ? recv_counts.data() : nullptr, myid == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (myid != 0) return;

  std::vector<ThicknessRecord> records;
  records.reserve(recv_buffer.size() / n_value);
  for (size_t i = 0; i + n_value - 1 < recv_buffer.size(); i += n_value) {
    records.push_back({recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2], recv_buffer[i + 3], recv_buffer[i + 4]});
  }

  std::stable_sort(records.begin(), records.end(),
                   [](const ThicknessRecord &lhs, const ThicknessRecord &rhs) { return lhs.x < rhs.x; });

  const std::filesystem::path out_file("output/stat/stat_thickness.txt");
  std::ofstream out(out_file, std::ios::trunc);
  out << "x yc delta_theta delta_omega delta_vis\n";
  for (const auto &rec: records) {
    out << rec.x << ' ' << rec.yc << ' ' << rec.delta_theta << ' ' << rec.delta_omega << ' ' << rec.delta_vis << '\n';
  }
}

void SinglePointStat::collect_data(DParameter *param) {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2 || perform_spanwise_average) {
    tpb = {16, 16, 1};
  }

  for (int b = 0; b < mesh.n_block; ++b) {
    if (perform_spanwise_average) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 bpg = {(mx + 2 - 1) / tpb.x + 1, (my + 2 - 1) / tpb.y + 1, 1};
      collect_singlePointStat_spanAvg_Step1<<<bpg, tpb>>>(field[b].d_ptr, param);
    } else {
      const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 bpg = {(mx + 2 - 1) / tpb.x + 1, (my + 2 - 1) / tpb.y + 1, (mz + 2 - 1) / tpb.z + 1};
      collect_singlePointStat_1ghostLayer_Step1<<<bpg, tpb>>>(field[b].d_ptr, param);
      if (scalar_fluc_budget)
        collect_singlePointStat_1ghostLayer_Step2<<<bpg, tpb>>>(field[b].d_ptr, param);
      if (species_dissipation_rate || species_velocity_correlation) {
        dim3 bpg2 = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
        collect_single_point_additional_statistics<<<bpg2, tpb>>>(field[b].d_ptr, param);
      }
    }
  }

  // Update the counters
  for (auto &c: counter_rey1st) {
    ++c;
  }
  for (auto &c: counter_rey1stScalar) {
    ++c;
  }
  for (auto &c: counter_fav1st) {
    ++c;
  }
  for (auto &c: counter_rey2nd) {
    ++c;
  }
  for (auto &c: counter_fav2nd) {
    ++c;
  }
  for (auto &c: counter_tke_budget) {
    ++c;
  }
  for (auto &c: counter_scalar_fluc_budget) {
    ++c;
  }
  for (auto &c: counter_species_dissipation_rate) {
    ++c;
  }
  for (auto &c: counter_species_velocity_correlation) {
    ++c;
  }
}

void SinglePointStat::export_statistical_data(DParameter *param) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1, fp_fav1, fp_rey2, fp_fav2;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_1st.bin").c_str(),
                MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_1st.bin").c_str(),
                MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav1);
  if (if_collect_2nd_moments) {
    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_rey_2nd.bin").c_str(),
                  MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey2);
    MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/coll_fav_2nd.bin").c_str(),
                  MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_fav2);
  }
  MPI_Status status;

  MPI_Offset offset[4]{0, 0, 0, 0};
  memcpy(offset, offset_unit, sizeof(offset_unit));
  if (myid == 0) {
    MPI_File_write_at(fp_rey1, offset[0], counter_rey1st.data(), n_reyAve, MPI_INT32_T, &status);
    offset[0] += 4 * n_reyAve;
    MPI_File_write_at(fp_rey1, offset[0], counter_rey1stScalar.data(), n_reyAveScalar, MPI_INT32_T, &status);
    offset[0] += 4 * n_reyAveScalar;
    MPI_File_write_at(fp_fav1, offset[1], counter_fav1st.data(), n_favAve, MPI_INT32_T, &status);
    offset[1] += 4 * n_favAve;
    if (if_collect_2nd_moments) {
      MPI_File_write_at(fp_rey2, offset[2], counter_rey2nd.data(), n_rey2nd, MPI_INT32_T, &status);
      offset[2] += 4 * n_rey2nd;
      MPI_File_write_at(fp_fav2, offset[3], counter_fav2nd.data(), n_fav2nd, MPI_INT32_T, &status);
      offset[3] += 4 * n_fav2nd;
    }
  }
  for (int b = 0; b < mesh.n_block; ++b) {
    const auto &zone = field[b].h_ptr;
    int mz = 1;
    if (!perform_spanwise_average) {
      mz = mesh[b].mz;
    }
    const int mx = mesh[b].mx, my = mesh[b].my;
    const auto sz = static_cast<long long>(mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;

    cudaMemcpy(field[b].collect_reynolds_1st.data(), zone->collect_reynolds_1st.data(),
               sz * (n_reyAve + n_reyAveScalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(field[b].collect_favre_1st.data(), zone->collect_favre_1st.data(), sz * n_favAve,
               cudaMemcpyDeviceToHost);
    if (if_collect_2nd_moments) {
      cudaMemcpy(field[b].collect_reynolds_2nd.data(), zone->collect_reynolds_2nd.data(), sz * n_rey2nd,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(field[b].collect_favre_2nd.data(), zone->collect_favre_2nd.data(), sz * n_fav2nd,
                 cudaMemcpyDeviceToHost);
    }

    // We create this datatype because in the original MPI_File_write_at, the number of elements is a 64-bit integer.
    // However, the nx*ny*nz may be larger than 2^31, so we need to use MPI_Type_create_subarray to create a datatype
    const auto ty = ty_1gg[b];

    MPI_File_write_at(fp_rey1, offset[0], &mx, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], &my, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], &mz, 1, MPI_INT32_T, &status);
    offset[0] += 4;
    MPI_File_write_at(fp_rey1, offset[0], field[b].collect_reynolds_1st.data(), n_reyAve + n_reyAveScalar, ty, &status);
    offset[0] += sz * (n_reyAve + n_reyAveScalar);

    MPI_File_write_at(fp_fav1, offset[1], &mx, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], &my, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], &mz, 1, MPI_INT32_T, &status);
    offset[1] += 4;
    MPI_File_write_at(fp_fav1, offset[1], field[b].collect_favre_1st.data(), n_favAve, ty, &status);
    offset[1] += sz * n_favAve;

    if (if_collect_2nd_moments) {
      MPI_File_write_at(fp_rey2, offset[2], &mx, 1, MPI_INT32_T, &status);
      offset[2] += 4;
      MPI_File_write_at(fp_rey2, offset[2], &my, 1, MPI_INT32_T, &status);
      offset[2] += 4;
      MPI_File_write_at(fp_rey2, offset[2], &mz, 1, MPI_INT32_T, &status);
      offset[2] += 4;
      MPI_File_write_at(fp_rey2, offset[2], field[b].collect_reynolds_2nd.data(), n_rey2nd, ty, &status);
      offset[2] += sz * n_rey2nd;

      MPI_File_write_at(fp_fav2, offset[3], &mx, 1, MPI_INT32_T, &status);
      offset[3] += 4;
      MPI_File_write_at(fp_fav2, offset[3], &my, 1, MPI_INT32_T, &status);
      offset[3] += 4;
      MPI_File_write_at(fp_fav2, offset[3], &mz, 1, MPI_INT32_T, &status);
      offset[3] += 4;
      MPI_File_write_at(fp_fav2, offset[3], field[b].collect_favre_2nd.data(), n_fav2nd, ty, &status);
      offset[3] += sz * n_fav2nd;
    }
  }

  MPI_File_close(&fp_rey1);
  MPI_File_close(&fp_fav1);
  if (if_collect_2nd_moments) {
    MPI_File_close(&fp_rey2);
    MPI_File_close(&fp_fav2);
  }

  // Any other stats that need no ghost layer
  // other files
  if (!perform_spanwise_average) {
    if (tke_budget) {
      export_tke_budget_file(parameter, mesh, field, offset_tke_budget, counter_tke_budget, ty_1gg);
    }
    if (scalar_fluc_budget) {
      export_species_collect_file<ScalarFlucBudget>
          (parameter, mesh, field, offset_scalar_fluc_budget, counter_scalar_fluc_budget, ty_1gg);
    }
    if (species_velocity_correlation) {
      export_species_collect_file<ScalarVelocityCorrelation>
      (parameter, mesh, field, offset_species_velocity_correlation, counter_species_velocity_correlation,
       ty_0gg);
    }
    if (species_dissipation_rate) {
      export_species_collect_file<ScalarDissipationRate>(parameter, mesh, field, offset_species_dissipation_rate,
                                                         counter_species_dissipation_rate, ty_0gg);
    }
  }
  if (perform_spanwise_average && parameter.get_bool("output_statistics_plt")) {
    plot_statistical_data(param);
  }
}

__global__ void collect_singlePointStat_1ghostLayer_Step1(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= extent[0] + 1 || j >= extent[1] + 1 || k >= extent[2] + 1) return;

  const auto &bv = zone->bv;
  const auto &sv = zone->sv;

  // The first order statistics of the flow field
  // Reynolds averaged variables
  auto &rey1st = zone->collect_reynolds_1st;
  const auto rho = bv(i, j, k, 0), p = bv(i, j, k, 4);
  rey1st(i, j, k, 0) += rho; // rho
  rey1st(i, j, k, 1) += p;   // p
  for (int l = 2; l < param->n_reyAve; ++l) {
    rey1st(i, j, k, l) += bv(i, j, k, param->reyAveVarIndex[l]);
  }
  for (int l = 0, n_reyAve = param->n_reyAve; l < param->n_reyAveScalar; ++l) {
    rey1st(i, j, k, l + n_reyAve) += sv(i, j, k, param->reyAveScalarIndex[l]);
  }
  // Favre averaged variables
  auto &fav1st = zone->collect_favre_1st;
  const real u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3), T = bv(i, j, k, 5);
  fav1st(i, j, k, 0) += rho * u; // rho*u
  fav1st(i, j, k, 1) += rho * v; // rho*v
  fav1st(i, j, k, 2) += rho * w; // rho*w
  fav1st(i, j, k, 3) += rho * T; // rho*T
  int scalar_offset = 4;
  if (param->if_collect_spec_favreAvg) {
    for (int l = 0; l < param->n_spec; ++l) {
      fav1st(i, j, k, scalar_offset + l) += rho * sv(i, j, k, l);
    }
    scalar_offset += param->n_spec;
  }
  for (int l = 0; l < param->n_ps; ++l) {
    fav1st(i, j, k, scalar_offset + l) += rho * sv(i, j, k, param->i_ps + l);
  }

  // The second order statistics of the flow field
  if (param->if_collect_2nd_moments) {
    // Reynolds averaged variables
    auto &rey2nd = zone->collect_reynolds_2nd;
    rey2nd(i, j, k, 0) += rho * rho; // rho*rho
    rey2nd(i, j, k, 1) += p * p;     // p*p
    if (param->rho_p_correlation)
      rey2nd(i, j, k, 2) += rho * p; // rho*p

    // Favre averaged variables
    auto &fav2nd = zone->collect_favre_2nd;
    fav2nd(i, j, k, 0) += rho * u * u; // rho*u*u
    fav2nd(i, j, k, 1) += rho * v * v; // rho*v*v
    fav2nd(i, j, k, 2) += rho * w * w; // rho*w*w
    fav2nd(i, j, k, 3) += rho * u * v; // rho*u*v
    fav2nd(i, j, k, 4) += rho * u * w; // rho*u*w
    fav2nd(i, j, k, 5) += rho * v * w; // rho*v*w
    fav2nd(i, j, k, 6) += rho * T * T; // rho*T*T
    scalar_offset = 7;
    if (param->if_collect_spec_favreAvg) {
      for (int l = 0; l < param->n_spec; ++l) {
        fav2nd(i, j, k, scalar_offset + l) += rho * sv(i, j, k, l) * sv(i, j, k, l);
      }
      scalar_offset += param->n_spec;
    }
    for (int l = 0; l < param->n_ps; ++l) {
      fav2nd(i, j, k, scalar_offset + l) += rho * sv(i, j, k, param->i_ps + l) * sv(i, j, k, param->i_ps + l);
    }
  }

  // Stats which needs an additional ghost layer
  if (param->stat_tke_budget) {
    collect_tke_budget(zone, param, i, j, k);
  }
}

__global__ void collect_singlePointStat_spanAvg_Step1(DZone *zone, DParameter *param) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  if (i >= zone->mx + 1 || j >= zone->my + 1) return;

  const auto &bv = zone->bv;
  const auto &sv = zone->sv;

  // The first order statistics of the flow field
  real rho_sum = 0.0, p_sum = 0.0, rhoU_sum = 0.0, rhoV_sum = 0.0, rhoW_sum = 0.0, rhoT_sum = 0.0;
  real rhoY_sum[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER]{};
  real rhoYY_sum[MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER]{};
  real rhoRho_sum{0.0}, pP_sum{0.0}, Rij_sum[6]{}, rhoTT_sum{0.0}, rhoP_sum{0.0};
  const int ns_stat = param->if_collect_spec_favreAvg ? param->n_spec : 0;
  for (int k = 0; k < zone->mz; ++k) {
    const auto rho = bv(i, j, k, 0), p = bv(i, j, k, 4);
    const auto u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3), T = bv(i, j, k, 5);
    // Reynolds averaged variables
    rho_sum += rho; // rho
    p_sum += p;     // p

    // Favre averaged variables
    rhoU_sum += rho * u; // rho*u
    rhoV_sum += rho * v; // rho*v
    rhoW_sum += rho * w; // rho*w
    rhoT_sum += rho * T; // rho*T
    if (ns_stat > 0) {
      for (int l = 0; l < ns_stat; ++l) rhoY_sum[l] += rho * sv(i, j, k, l);
    }
    for (int l = 0; l < param->n_ps; ++l) rhoY_sum[ns_stat + l] += rho * sv(i, j, k, param->i_ps + l);

    if (param->if_collect_2nd_moments) {
      rhoRho_sum += rho * rho;                           // rho*rho
      pP_sum += p * p;                                   // p*p
      if (param->rho_p_correlation) rhoP_sum += rho * p; // rho*p

      Rij_sum[0] += rho * u * u; // rho*u*u
      Rij_sum[1] += rho * v * v; // rho*v*v
      Rij_sum[2] += rho * w * w; // rho*w*w
      Rij_sum[3] += rho * u * v; // rho*u*v
      Rij_sum[4] += rho * u * w; // rho*u*w
      Rij_sum[5] += rho * v * w; // rho*v*w
      rhoTT_sum += rho * T * T;  // rho*T*T
      if (ns_stat > 0) { for (int l = 0; l < ns_stat; ++l) rhoYY_sum[l] += rho * sv(i, j, k, l) * sv(i, j, k, l); }
      for (int l = 0; l < param->n_ps; ++l)
        rhoYY_sum[ns_stat + l] +=
            rho * sv(i, j, k, param->i_ps + l) * sv(i, j, k, param->i_ps + l);
    }
  }

  const real iNz = 1.0 / zone->mz;
  auto &rey1st = zone->collect_reynolds_1st;
  rey1st(i, j, 0, 0) += rho_sum * iNz; // rho
  rey1st(i, j, 0, 1) += p_sum * iNz;   // p

  auto &fav1st = zone->collect_favre_1st;
  fav1st(i, j, 0, 0) += rhoU_sum * iNz; // rho*u
  fav1st(i, j, 0, 1) += rhoV_sum * iNz; // rho*v
  fav1st(i, j, 0, 2) += rhoW_sum * iNz; // rho*w
  fav1st(i, j, 0, 3) += rhoT_sum * iNz; // rho*T
  for (int l = 0; l < ns_stat + param->n_ps; ++l) fav1st(i, j, 0, 4 + l) += rhoY_sum[l] * iNz;


  if (param->if_collect_2nd_moments) {
    auto &rey2nd = zone->collect_reynolds_2nd;
    rey2nd(i, j, 0, 0) += rhoRho_sum * iNz;                             // rho*rho
    rey2nd(i, j, 0, 1) += pP_sum * iNz;                                 // p*p
    if (param->rho_p_correlation) rey2nd(i, j, 0, 2) += rhoP_sum * iNz; // rho*p

    auto &fav2nd = zone->collect_favre_2nd;
    fav2nd(i, j, 0, 0) += Rij_sum[0] * iNz; // rho*u*u
    fav2nd(i, j, 0, 1) += Rij_sum[1] * iNz; // rho*v*v
    fav2nd(i, j, 0, 2) += Rij_sum[2] * iNz; // rho*w*w
    fav2nd(i, j, 0, 3) += Rij_sum[3] * iNz; // rho*u*v
    fav2nd(i, j, 0, 4) += Rij_sum[4] * iNz; // rho*u*w
    fav2nd(i, j, 0, 5) += Rij_sum[5] * iNz; // rho*v*w
    fav2nd(i, j, 0, 6) += rhoTT_sum * iNz;  // rho*T*T
    for (int l = 0; l < ns_stat + param->n_ps; ++l)
      fav2nd(i, j, 0, 7 + l) += rhoYY_sum[l] * iNz;
  }
}

__global__ void collect_singlePointStat_1ghostLayer_Step2(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= extent[0] + 1 || j >= extent[1] + 1 || k >= extent[2] + 1) return;

  if (param->stat_scalar_fluc_budget) {
    // Please make sure the tke budget is collected before the scalar fluc budget
    collect_scalar_fluc_budget(zone, param, i, j, k);
  }
}

__global__ void collect_single_point_additional_statistics(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  if (param->stat_species_velocity_correlation) {
    collect_species_velocity_correlation(zone, param, i, j, k);
  }
  if (param->stat_species_dissipation_rate) {
    collect_species_dissipation_rate(zone, param, i, j, k);
  }
}

__global__ void compute_statistical_data_spanwise_average(DZone *zone, const DParameter *param, int N) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  if (i >= zone->mx || j >= zone->my) return;

  const auto &col_rey1st = zone->collect_reynolds_1st;
  const auto &col_fav1st = zone->collect_favre_1st;
  const auto &col_rey2nd = zone->collect_reynolds_2nd;
  const auto &col_fav2nd = zone->collect_favre_2nd;

  // Here, we assume all variables are collected with the same N.
  const real iN = 1.0 / N;
  auto &mean = zone->stat_reynolds_1st;
  mean(i, j, 0, 0) = col_rey1st(i, j, 0, 0) * iN;
  mean(i, j, 0, 1) = col_rey1st(i, j, 0, 1) * iN;

  auto &favre = zone->stat_favre_1st;
  const real iRho = 1.0 / col_rey1st(i, j, 0, 0); // 1/rho
  const real u = col_fav1st(i, j, 0, 0) * iRho;   // u
  const real v = col_fav1st(i, j, 0, 1) * iRho;   // v
  const real w = col_fav1st(i, j, 0, 2) * iRho;   // w
  const real T = col_fav1st(i, j, 0, 3) * iRho;   // T
  const int ns_stat = param->if_collect_spec_favreAvg ? param->n_spec : 0;
  const int n_scalar_stat = ns_stat + param->n_ps;
  favre(i, j, 0, 0) = col_fav1st(i, j, 0, 0) * iRho;                                                 // u
  favre(i, j, 0, 1) = col_fav1st(i, j, 0, 1) * iRho;                                                 // v
  favre(i, j, 0, 2) = col_fav1st(i, j, 0, 2) * iRho;                                                 // w
  favre(i, j, 0, 3) = col_fav1st(i, j, 0, 3) * iRho;                                                 // T
  for (int l = 0; l < n_scalar_stat; ++l) favre(i, j, 0, 4 + l) = col_fav1st(i, j, 0, 4 + l) * iRho; // Y
  if (param->if_collect_2nd_moments) {
    auto &rey2nd = zone->stat_reynolds_2nd;
    rey2nd(i, j, 0, 0) = col_rey2nd(i, j, 0, 0) * iN - mean(i, j, 0, 0) * mean(i, j, 0, 0); // rho*rho - rho^2
    rey2nd(i, j, 0, 1) = col_rey2nd(i, j, 0, 1) * iN - mean(i, j, 0, 1) * mean(i, j, 0, 1); // p*p - p^2
    if (param->rho_p_correlation)
      rey2nd(i, j, 0, 2) = col_rey2nd(i, j, 0, 2) * iN - mean(i, j, 0, 0) * mean(i, j, 0, 1); // rho

    auto &favre2nd = zone->stat_favre_2nd;
    favre2nd(i, j, 0, 0) = col_fav2nd(i, j, 0, 0) * iRho - u * u;
    favre2nd(i, j, 0, 1) = col_fav2nd(i, j, 0, 1) * iRho - v * v;
    favre2nd(i, j, 0, 2) = col_fav2nd(i, j, 0, 2) * iRho - w * w;
    favre2nd(i, j, 0, 3) = col_fav2nd(i, j, 0, 3) * iRho - u * v;
    favre2nd(i, j, 0, 4) = col_fav2nd(i, j, 0, 4) * iRho - u * w;
    favre2nd(i, j, 0, 5) = col_fav2nd(i, j, 0, 5) * iRho - v * w;
    favre2nd(i, j, 0, 6) = col_fav2nd(i, j, 0, 6) * iRho - T * T;
    for (int l = 0; l < n_scalar_stat; ++l)
      favre2nd(i, j, 0, 7 + l) = col_fav2nd(i, j, 0, 7 + l) * iRho - favre(i, j, 0, 4 + l) * favre(i, j, 0, 4 + l);
  }
}
} // cfd

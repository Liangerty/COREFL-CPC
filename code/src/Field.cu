#include "Field.h"
#include "BoundCond.h"
#include "UDIO.h"
#include <fstream>
#include "Parallel.h"
#include "gxl_lib/MyString.h"
#include "Transport.cuh"
#include "DParameter.cuh"
#include "stat_lib/TkeBudget.cuh"
#include "stat_lib/SpeciesStat.cuh"

cfd::Field::Field(Parameter &parameter, const Block &block_in) :
  block(block_in) {
  const int mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  // Let us re-compute the number of variables to be solved here.
  n_var = parameter.get_int("n_var");
  // The variable "n_scalar_transported" is the number of scalar variables to be transported.
  const int n_other_var{parameter.get_int("n_other_var")}; // Default, mach number
  // The variable "n_scalar" is the number of scalar variables in total, including those not transported.
  // This is different from the variable "n_scalar_transported" only when the flamelet model is used.
  const int n_scalar{parameter.get_int("n_scalar")};

  // Acquire memory for variable arrays
  bv.resize(mx, my, mz, 6, ngg);
  sv.resize(mx, my, mz, n_scalar, ngg);
  ov.resize(mx, my, mz, n_other_var, ngg);
  udv.resize(mx, my, mz, UserDefineIO::n_dynamic_auxiliary, ngg);

  if (int n_rand = parameter.get_int("random_number_per_point"); n_rand > 0) {
    rng_state.resize(mx, my, mz, n_rand, ngg);
  }
  fluc_val.resize(mx, my, mz, parameter.get_int("fluctuation_variable_number"), ngg);
  // This is commented. We would allocate the statistical data even when we do not collect the statistics.
  // if (parameter.get_bool("if_collect_statistics")) {
  // If we need to collect the statistics, we need to allocate memory for the data.
  collect_reynolds_1st.resize(
    mx, my, mz, parameter.get_int("n_stat_reynolds_1st") + parameter.get_int("n_stat_reynolds_1st_scalar"), 1);
  collect_favre_1st.resize(mx, my, mz, parameter.get_int("n_stat_favre_1st"), 1);
  collect_reynolds_2nd.resize(mx, my, mz, parameter.get_int("n_stat_reynolds_2nd"), 1);
  collect_favre_2nd.resize(mx, my, mz, parameter.get_int("n_stat_favre_2nd"), 1);
  if (parameter.get_bool("output_statistics_plt")) {
    if (parameter.get_bool("perform_spanwise_average")) {
      stat_reynolds_1st.resize(mx, my, 1, parameter.get_int("n_stat_reynolds_1st"), 0);
      stat_favre_1st.resize(mx, my, 1, parameter.get_int("n_stat_favre_1st"), 0);
      stat_reynolds_2nd.resize(mx, my, 1, parameter.get_int("n_stat_reynolds_2nd"), 0);
      stat_favre_2nd.resize(mx, my, 1, parameter.get_int("n_stat_favre_2nd"), 0);
    } else {
      stat_reynolds_1st.resize(mx, my, mz, parameter.get_int("n_stat_reynolds_1st"), 1);
      stat_favre_1st.resize(mx, my, mz, parameter.get_int("n_stat_favre_1st"), 1);
      stat_reynolds_2nd.resize(mx, my, mz, parameter.get_int("n_stat_reynolds_2nd"), 1);
      stat_favre_2nd.resize(mx, my, mz, parameter.get_int("n_stat_favre_2nd"), 1);
    }
  }
  // other statistics
  if (parameter.get_bool("stat_tke_budget")) {
    collect_tke_budget.resize(mx, my, mz, TkeBudget::n_collect, TkeBudget::ngg);
  }
  const int n_species_stat = parameter.get_int("n_species_stat");
  const int n_ps = parameter.get_int("n_ps");
  const int n_scalar_stat = n_species_stat + n_ps;
  if (n_scalar_stat > 0) {
    if (parameter.get_bool("stat_scalar_fluc_budget")) {
      collect_scalar_budget.resize(mx, my, mz, n_scalar_stat * ScalarFlucBudget::n_collect, ScalarFlucBudget::ngg);
    }
    if (parameter.get_bool("stat_species_velocity_correlation")) {
      collect_scalar_vel_correlation.resize(mx, my, mz, n_scalar_stat * ScalarVelocityCorrelation::n_collect,
                                            ScalarVelocityCorrelation::ngg);
    }
    if (parameter.get_bool("stat_species_dissipation_rate")) {
      collect_scalar_budget.resize(mx, my, mz, n_scalar_stat * ScalarDissipationRate::n_collect,
                                   ScalarDissipationRate::ngg);
    }
  }
  // }

  if (parameter.get_bool("sponge_layer")) {
    sponge_mean_cv.resize(mx, my, mz, n_var, 0);
  }
}

std::vector<int> identify_variable_labels(const cfd::Parameter &parameter, std::vector<std::string> &var_name,
  const cfd::Species &species, bool &has_pressure, bool &has_temperature, bool &has_tke) {
  std::vector<int> labels;
  const int n_spec = species.n_spec;
  const int n_turb = parameter.get_int("n_turb");
  for (auto &name: var_name) {
    int l = 999;
    // The first three names are x, y and z, they are assigned value 0 and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "X") {
      l = 0;
    } else if (n == "Y") {
      l = 1;
    } else if (n == "Z") {
      l = 2;
    } else if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0 + 3;
    } else if (n == "U") {
      l = 1 + 3;
    } else if (n == "V") {
      l = 2 + 3;
    } else if (n == "W") {
      l = 3 + 3;
    } else if (n == "P" || n == "PRESSURE") {
      l = 4 + 3;
      has_pressure = true;
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5 + 3;
      has_temperature = true;
    } else {
      if (n_spec > 0) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (const auto &[spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label + 3;
            break;
          }
        }
        if (n == "MIXTUREFRACTION") {
          // Mixture fraction
          l = 6 + n_spec + n_turb + 3;
        } else if (n == "MIXTUREFRACTIONVARIANCE") {
          // Mixture fraction variance
          l = 6 + n_spec + n_turb + 1 + 3;
        }
      }
      if (n_turb > 0) {
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K" || n == "TKE") { // turbulent kinetic energy
          if (n_turb == 2) {
            l = 6 + n_spec + 3;
            has_tke = true;
          }
        } else if (n == "OMEGA") { // specific dissipation rate
          if (n_turb == 2) {
            l = 6 + n_spec + 1 + 3;
          }
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          if (n_turb == 1) {
            l = 6 + n_spec + 3;
          }
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

std::array<int, 3> read_dat_profile_for_init(gxl::VectorField3D<real> &profile, const std::string &file,
  const cfd::Parameter &parameter, const cfd::Species &species, int profile_idx) {
  std::ifstream file_in(file);
  if (!file_in.is_open()) {
    printf("Cannot open file %s\n", file.c_str());
    cfd::MpiParallel::exit();
  }

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
  bool has_pressure{false}, has_temperature{false}, has_tke{false};
  auto label_order = identify_variable_labels(parameter, var_name, species, has_pressure, has_temperature, has_tke);
  if (!has_temperature && !has_pressure) {
    printf("The temperature or pressure is not given in the profile, please provide at least one of them!\n");
    cfd::MpiParallel::exit();
  }
  real turb_viscosity_ratio{0}, turb_intensity{0};
  if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
    auto &info = parameter.get_struct(parameter.get_string_array("profile_related_bc_names")[profile_idx]);
    if (info.find("turb_viscosity_ratio") == info.end() || info.find("turbulence_intensity") == info.end()) {
      printf(
        "The turbulence intensity or turbulent viscosity ratio is not given in the profile, please provide both of them!\n");
      cfd::MpiParallel::exit();
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

  std::array<int, 3> extent{mx, my, mz};
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
  profile.resize(extent[0], extent[1], extent[2], n_var + 4, 0);
  for (int k = 0; k < extent[2]; ++k) {
    for (int j = 0; j < extent[1]; ++j) {
      for (int i = 0; i < extent[0]; ++i) {
        for (int l = 0; l < nv_read; ++l) {
          if (label_order[l] < n_var + 1 + 3) {
            profile(i, j, k, label_order[l]) = profile_read(i, j, k, l);
          }
        }

        // If T or p is not given, compute it.
        if (!has_temperature) {
          real mw{cfd::mw_air};
          if (species.n_spec > 0) {
            mw = 0;
            for (int l = 0; l < species.n_spec; ++l) mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
            mw = 1 / mw;
          }
          profile(i, j, k, 5 + 3) = profile(i, j, k, 4 + 3) * mw / (cfd::R_u * profile(i, j, k, 0 + 3));
        }
        if (!has_pressure) {
          real mw{cfd::mw_air};
          if (species.n_spec > 0) {
            mw = 0;
            for (int l = 0; l < species.n_spec; ++l) mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
            mw = 1 / mw;
          }
          profile(i, j, k, 4 + 3) = profile(i, j, k, 5 + 3) * cfd::R_u * profile(i, j, k, 0 + 3) / mw;
        }
        if (parameter.get_int("turbulence_method") != 0 && parameter.get_int("RANS_model") == 2 && !has_tke) {
          // If the turbulence intensity is given, we need to compute the turbulent viscosity ratio.
          real mu{};
          if (species.n_spec > 0) {
            real mw = 0;
            std::vector<real> Y;
            for (int l = 0; l < species.n_spec; ++l) {
              mw += profile(i, j, k, 6 + l + 3) / species.mw[l];
              Y.push_back(profile(i, j, k, 6 + l + 3));
            }
            mw = 1 / mw;
            mu = compute_viscosity(profile(i, j, k, 5 + 3), mw, Y.data(), species);
          } else {
            mu = cfd::Sutherland(profile(i, j, k, 5 + 3));
          }
          real mut = mu * turb_viscosity_ratio;
          const real vel2 = profile(i, j, k, 1 + 3) * profile(i, j, k, 1 + 3) +
                            profile(i, j, k, 2 + 3) * profile(i, j, k, 2 + 3) +
                            profile(i, j, k, 3 + 3) * profile(i, j, k, 3 + 3);
          profile(i, j, k, 6 + species.n_spec + 3) = 1.5 * vel2 * turb_intensity * turb_intensity;
          profile(i, j, k, 6 + species.n_spec + 1 + 3) =
              profile(i, j, k, 0 + 3) * profile(i, j, k, 6 + species.n_spec + 3) / mut;
        }
      }
    }
  }
  return extent;
}

std::array<int, 3> read_profile_to_init(gxl::VectorField3D<real> &profile, int profile_idx,
  const cfd::Parameter &parameter, const cfd::Species &species) {
  const std::string &file = parameter.get_string_array("profile_file_names")[profile_idx];
  const auto dot = file.find_last_of('.');
  const auto suffix = file.substr(dot + 1, file.size());
  if (suffix == "dat") {
    const auto extent = read_dat_profile_for_init(profile, file, parameter, species, profile_idx);
    return extent;
  } /*else if (suffix == "plt") {
//    read_plt_profile();
  }*/
  return {0, 0, 0};
}

__global__ void initialize_bv_with_inflow(const real *var_info, int n_inflow, cfd::DZone *zone,
  const real *coordinate_ranges, int n_scalar, const int *if_profile, ggxl::VectorField3D<real> *profiles,
  const int *extents) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  auto &x = zone->x, &y = zone->y, &z = zone->z;
  auto &bv = zone->bv, &sv = zone->sv;
  const real *rho = var_info, *u = rho + n_inflow, *v = u + n_inflow, *w = v + n_inflow, *p = w + n_inflow, *T =
      p + n_inflow;
  const real *scalar_inflow = T + n_inflow;
  int i_init{0};

  if (n_inflow > 1) {
    for (int l = 0; l < n_inflow - 1; ++l) {
      if (x(i, j, k) >= coordinate_ranges[l * 6] && x(i, j, k) <= coordinate_ranges[l * 6 + 1]
          && y(i, j, k) >= coordinate_ranges[l * 6 + 2] && y(i, j, k) <= coordinate_ranges[l * 6 + 3]
          && z(i, j, k) >= coordinate_ranges[l * 6 + 4] && z(i, j, k) <= coordinate_ranges[l * 6 + 5]) {
        i_init = l + 1;
        break;
      }
    }
  }
  if (if_profile[i_init]) {
    // Assign the profile by 0th order interpolation
    auto &profile = profiles[i_init];
    const auto xx = x(i, j, k), yy = y(i, j, k), zz = z(i, j, k);
    const auto extent = &extents[i_init * 3];
    real d_mix = 1e+6;
    int i0{0}, j0{0}, k0{0};
    for (int kk = 0; kk < extent[2]; ++kk) {
      for (int jj = 0; jj < extent[1]; ++jj) {
        for (int ii = 0; ii < extent[0]; ++ii) {
          const real d = sqrt((xx - profile(ii, jj, kk, 0)) * (xx - profile(ii, jj, kk, 0)) +
                              (yy - profile(ii, jj, kk, 1)) * (yy - profile(ii, jj, kk, 1)) +
                              (zz - profile(ii, jj, kk, 2)) * (zz - profile(ii, jj, kk, 2)));
          if (d < d_mix) {
            d_mix = d;
            i0 = ii;
            j0 = jj;
            k0 = kk;
          }
        }
      }
    }

    // 0th order interpolation
    bv(i, j, k, 0) = profile(i0, j0, k0, 3);
    bv(i, j, k, 1) = profile(i0, j0, k0, 4);
    bv(i, j, k, 2) = profile(i0, j0, k0, 5);
    bv(i, j, k, 3) = profile(i0, j0, k0, 6);
    bv(i, j, k, 4) = profile(i0, j0, k0, 7);
    bv(i, j, k, 5) = profile(i0, j0, k0, 8);
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = profile(i0, j0, k0, 9 + l);
    }
  } else {
    bv(i, j, k, 0) = rho[i_init];
    bv(i, j, k, 1) = u[i_init];
    bv(i, j, k, 2) = v[i_init];
    bv(i, j, k, 3) = w[i_init];
    bv(i, j, k, 4) = p[i_init];
    bv(i, j, k, 5) = T[i_init];
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = scalar_inflow[l * n_inflow + i_init];
    }
  }
}

void cfd::Field::initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow> &inflows,
  const std::vector<real> &xs, const std::vector<real> &xe, const std::vector<real> &ys, const std::vector<real> &ye,
  const std::vector<real> &zs, const std::vector<real> &ze, const Species &species) const {
  const int n = static_cast<int>(inflows.size());
  const int n_scalar = parameter.get_int("n_scalar");
  std::vector<real> var_info((6 + n_scalar) * n, 0);
  real *rho = var_info.data(), *u = rho + n, *v = u + n, *w = v + n, *p = w + n, *T = p + n;
  real *scalar_inflow = T + n;

  std::vector<int> if_profile(n, 0);
  std::vector<int> profile_id(n, 0);
  std::vector<gxl::VectorField3D<real>> profiles(n);
  std::vector<int> extent;
  std::vector<std::array<int, 3>> extents(n);

  for (int i = 0; i < static_cast<int>(inflows.size()); ++i) {
    std::tie(rho[i], u[i], v[i], w[i], p[i], T[i]) = inflows[i].var_info();
    if_profile[i] = inflows[i].inflow_type == 1 ? 1 : 0;
    if (if_profile[i]) {
      profile_id[i] = inflows[i].profile_idx;
      extents[i] = read_profile_to_init(profiles[i], i, parameter, species);
      extent.push_back(extents[i][0]);
      extent.push_back(extents[i][1]);
      extent.push_back(extents[i][2]);
    }
  }
  for (size_t i = 0; i < inflows.size(); ++i) {
    const auto sv_this = inflows[i].sv;
    for (int l = 0; l < n_scalar; ++l) {
      scalar_inflow[n * l + i] = sv_this[l];
    }
  }
  real *var_info_device;
  cudaMalloc(&var_info_device, sizeof(real) * var_info.size());
  cudaMemcpy(var_info_device, var_info.data(), sizeof(real) * var_info.size(), cudaMemcpyHostToDevice);
  real *coordinate_ranges_device;
  cudaMalloc(&coordinate_ranges_device, sizeof(real) * 6 * (n - 1));
  std::vector<real> coordinate_ranges;
  for (int l = 0; l < n - 1; ++l) {
    coordinate_ranges.emplace_back(xs[l]);
    coordinate_ranges.emplace_back(xe[l]);
    coordinate_ranges.emplace_back(ys[l]);
    coordinate_ranges.emplace_back(ye[l]);
    coordinate_ranges.emplace_back(zs[l]);
    coordinate_ranges.emplace_back(ze[l]);
  }
  cudaMemcpy(coordinate_ranges_device, coordinate_ranges.data(), sizeof(real) * coordinate_ranges.size(),
             cudaMemcpyHostToDevice);
  std::vector<ggxl::VectorField3D<real>> profile_dev_temp(n);
  for (int l = 0; l < n; ++l) {
    profile_dev_temp[l].allocate_memory(extents[l][0], extents[l][1], extents[l][2], 9 + n_scalar, 0);
    cudaMemcpy(profile_dev_temp[l].data(), profiles[l].data(),
               sizeof(real) * profile_dev_temp[l].size() * (9 + n_scalar), cudaMemcpyHostToDevice);
  }
  ggxl::VectorField3D<real> *profiles_device;
  cudaMalloc(&profiles_device, sizeof(ggxl::VectorField3D<real>) * n);
  cudaMemcpy(profiles_device, profile_dev_temp.data(), sizeof(ggxl::VectorField3D<real>) * n, cudaMemcpyHostToDevice);
  int *extent_device;
  cudaMalloc(&extent_device, sizeof(int) * 3 * n);
  if (!extent.empty())
    cudaMemcpy(extent_device, extent.data(), sizeof(int) * 3 * n, cudaMemcpyHostToDevice);
  int *if_profile_device;
  cudaMalloc(&if_profile_device, sizeof(int) * n);
  cudaMemcpy(if_profile_device, if_profile.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
  dim3 tpb{8, 8, 4};
  if (block.mz == 1) {
    tpb = {16, 16, 1};
  }
  const int ngg{block.ngg};
  dim3 bpg = {
    (block.mx + 2 * ngg - 1) / tpb.x + 1, (block.my + 2 * ngg - 1) / tpb.y + 1, (block.mz + 2 * ngg - 1) / tpb.z + 1
  };
  initialize_bv_with_inflow<<<bpg, tpb>>>(var_info_device, n, d_ptr, coordinate_ranges_device,
                                          n_scalar, if_profile_device, profiles_device, extent_device);
  cudaMemcpy(bv.data(), h_ptr->bv.data(), sizeof(real) * h_ptr->bv.size() * 6, cudaMemcpyDeviceToHost);
  cudaMemcpy(sv.data(), h_ptr->sv.data(), sizeof(real) * h_ptr->sv.size() * n_scalar, cudaMemcpyDeviceToHost);
}

void cfd::Field::setup_device_memory(const Parameter &parameter) {
  h_ptr = new DZone;
  const auto mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  h_ptr->mx = mx, h_ptr->my = my, h_ptr->mz = mz, h_ptr->ngg = ngg;
  h_ptr->iMax = mx - 1, h_ptr->jMax = my - 1, h_ptr->kMax = mz - 1;

  h_ptr->x.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->x.data(), block.x.data(), sizeof(real) * h_ptr->x.size(), cudaMemcpyHostToDevice);
  h_ptr->y.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->y.data(), block.y.data(), sizeof(real) * h_ptr->y.size(), cudaMemcpyHostToDevice);
  h_ptr->z.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->z.data(), block.z.data(), sizeof(real) * h_ptr->z.size(), cudaMemcpyHostToDevice);
  if (parameter.get_bool("turbulence") && parameter.get_int("turbulence_method") == 2) {
    h_ptr->des_delta.allocate_memory(mx, my, mz, ngg);
    cudaMemcpy(h_ptr->des_delta.data(), block.des_scale.data(), sizeof(real) * h_ptr->des_delta.size(),
               cudaMemcpyHostToDevice);
  }
  h_ptr->bType_il.allocate_memory(my, mz, 0);
  cudaMemcpy(h_ptr->bType_il.data(), block.bType_il.data(), sizeof(int) * h_ptr->bType_il.size(),
             cudaMemcpyHostToDevice);
  h_ptr->bType_ir.allocate_memory(my, mz, 0);
  cudaMemcpy(h_ptr->bType_ir.data(), block.bType_ir.data(), sizeof(int) * h_ptr->bType_ir.size(),
             cudaMemcpyHostToDevice);
  h_ptr->bType_jl.allocate_memory(mx, mz, 0);
  cudaMemcpy(h_ptr->bType_jl.data(), block.bType_jl.data(), sizeof(int) * h_ptr->bType_jl.size(),
             cudaMemcpyHostToDevice);
  h_ptr->bType_jr.allocate_memory(mx, mz, 0);
  cudaMemcpy(h_ptr->bType_jr.data(), block.bType_jr.data(), sizeof(int) * h_ptr->bType_jr.size(),
             cudaMemcpyHostToDevice);
  h_ptr->bType_kl.allocate_memory(mx, my, 0);
  cudaMemcpy(h_ptr->bType_kl.data(), block.bType_kl.data(), sizeof(int) * h_ptr->bType_kl.size(),
             cudaMemcpyHostToDevice);
  h_ptr->bType_kr.allocate_memory(mx, my, 0);
  cudaMemcpy(h_ptr->bType_kr.data(), block.bType_kr.data(), sizeof(int) * h_ptr->bType_kr.size(),
             cudaMemcpyHostToDevice);


  const auto n_bound{block.boundary.size()};
  h_ptr->n_boundary = static_cast<int>(n_bound);
  const auto n_inner{block.inner_face.size()};
  const auto n_par{block.parallel_face.size()};
  auto mem_sz = sizeof(Boundary) * n_bound;
  cudaMalloc(&h_ptr->boundary, mem_sz);
  cudaMemcpy(h_ptr->boundary, block.boundary.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(InnerFace) * n_inner;
  cudaMalloc(&h_ptr->innerFace, mem_sz);
  cudaMemcpy(h_ptr->innerFace, block.inner_face.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(ParallelFace) * n_par;
  cudaMalloc(&h_ptr->parFace, mem_sz);
  cudaMemcpy(h_ptr->parFace, block.parallel_face.data(), mem_sz, cudaMemcpyHostToDevice);

  h_ptr->jac.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->jac.data(), block.jacobian.data(), sizeof(real) * h_ptr->jac.size(), cudaMemcpyHostToDevice);
  h_ptr->metric.allocate_memory(mx, my, mz, 9, ngg);
  cudaMemcpy(h_ptr->metric.data(), block.metric.data(), sizeof(real) * h_ptr->metric.size() * 9,
             cudaMemcpyHostToDevice);

  h_ptr->cv.allocate_memory(mx, my, mz, n_var, ngg);
  h_ptr->bv.allocate_memory(mx, my, mz, 6, ngg);
  h_ptr->bv_last.allocate_memory(mx, my, mz, 4, 0);
  h_ptr->mul.allocate_memory(mx, my, mz, ngg);

  const auto n_spec{parameter.get_int("n_spec")};
  const auto n_scalar = parameter.get_int("n_scalar");
  h_ptr->sv.allocate_memory(mx, my, mz, n_scalar, ngg);
  h_ptr->rho_D.allocate_memory(mx, my, mz, n_spec, ngg);
  if (n_spec > 0) {
    h_ptr->acoustic_speed.allocate_memory(mx, my, mz, ngg);
    h_ptr->thermal_conductivity.allocate_memory(mx, my, mz, ngg);
    h_ptr->gamma.allocate_memory(mx, my, mz, ngg);
    h_ptr->cp.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("reaction") == 1) {
      // Finite rate chemistry
      h_ptr->reaction_timeScale.allocate_memory(mx, my, mz, ngg);
      if (const int chemSrcMethod = parameter.get_int("chemSrcMethod"); chemSrcMethod == 1) {
        // EPI
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec * n_spec, 0);
      } else if (chemSrcMethod == 2) {
        // DA
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec, 0);
      }
    } else if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
      // Flamelet model
      // h_ptr->scalar_diss_rate.allocate_memory(mx, my, mz, ngg);
      // Maybe we can also implicitly treat the source term here.
    }
  }
  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS method or DES method
    // h_ptr->mut.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      h_ptr->wall_distance.allocate_memory(mx, my, mz, ngg);
      if (parameter.get_int("turb_implicit") == 1) {
        h_ptr->turb_src_jac.allocate_memory(mx, my, mz, 2, 0);
      }
    }
  }
  if (parameter.get_int("if_compute_wall_distance") == 1) {
    h_ptr->wall_distance.allocate_memory(mx, my, mz, ngg);
  }

  h_ptr->dq.allocate_memory(mx, my, mz, n_var, 0);
  if (parameter.get_string("hybrid_inviscid_scheme") != "NO") {
    h_ptr->shock_sensor.allocate_memory(mx, my, mz, ngg);
  }
  h_ptr->fFlux.allocate_memory(mx, my, mz, n_var, ngg);
  h_ptr->gFlux.allocate_memory(mx, my, mz, n_var, ngg);
  h_ptr->hFlux.allocate_memory(mx, my, mz, n_var, ngg);
  if (parameter.get_int("viscous_order") == 2) {
    h_ptr->vis_flux.allocate_memory(mx, my, mz, n_var - 1, 1);
  }
  // else {
  //   h_ptr->fv.allocate_memory(mx, my, mz, n_var - 1, ngg);
  //   h_ptr->gv.allocate_memory(mx, my, mz, n_var - 1, ngg);
  //   h_ptr->hv.allocate_memory(mx, my, mz, n_var - 1, ngg);
  // }
  if (!(!parameter.get_bool("steady") && parameter.get_int("temporal_scheme") == 3 &&
        parameter.get_bool("fixed_time_step"))) {
    h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 0);
    h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 0);
    h_ptr->dt_local.allocate_memory(mx, my, mz, 0);
  }
  if (parameter.get_int("implicit_method") == 1) { // DPLUR
    if (!(!parameter.get_bool("steady") && parameter.get_int("temporal_scheme") == 3)) {
      // If DPLUR type, when computing the products of convective jacobian and dq, we need 1 layer of ghost grids whose dq=0.
      // Except those inner or parallel communication faces, they need to get the dq from neighbor blocks.
      h_ptr->dq.allocate_memory(mx, my, mz, n_var, 1);
      h_ptr->dq0.allocate_memory(mx, my, mz, n_var, 1);
      h_ptr->dqk.allocate_memory(mx, my, mz, n_var, 1);
      h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 1);
      h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 1);
    }
  }
  if (parameter.get_int("inviscid_scheme") == 2) {
    // Roe scheme
    h_ptr->entropy_fix_delta.allocate_memory(mx, my, mz, 1);
  }
  if (parameter.get_int("inviscid_scheme") == 72) {
    //    // WENO scheme
    //    h_ptr->Fp.allocate_memory(mx, my, mz, n_var, ngg);
    //    h_ptr->Fm.allocate_memory(mx, my, mz, n_var, ngg);
    //    h_ptr->flux.allocate_memory(mx, my, mz, n_var, ngg);
    //    h_ptr->f_1st.allocate_memory(mx, my, mz, n_var - 5, ngg);
  }

  if (!parameter.get_bool("steady")) {
    // unsteady simulation
    if (parameter.get_int("temporal_scheme") == 2) {
      // dual time stepping
      h_ptr->qn1.allocate_memory(mx, my, mz, n_var, ngg);
      h_ptr->qn_star.allocate_memory(mx, my, mz, n_var, ngg);
      h_ptr->in_last_step.allocate_memory(mx, my, mz, 4, 0);
    }
    if (parameter.get_int("temporal_scheme") == 3) {
      // rk scheme
      h_ptr->qn.allocate_memory(mx, my, mz, n_var, ngg);
    }
    if (parameter.get_int("reaction") > 0) {
      h_ptr->Cn.allocate_memory(mx, my, mz, n_var, 0);
    }
  }

  if (int n_rand = parameter.get_int("random_number_per_point"); n_rand > 0) {
    h_ptr->rng_state.allocate_memory(mx, my, mz, n_rand, ngg);
  }
  h_ptr->fluc_val.allocate_memory(mx, my, mz, parameter.get_int("fluctuation_variable_number"), ngg);

  // This is commented. We would allocate the statistical data even when we do not collect the statistics.
  // if (parameter.get_bool("if_collect_statistics")) {
  // if (parameter.get_bool("output_statistics_plt")) {
  //   if (parameter.get_bool("perform_spanwise_average")) {
  //     h_ptr->stat_reynolds_1st.allocate_memory(
  //       mx, my, 1, parameter.get_int("n_stat_reynolds_1st") + parameter.get_int("n_stat_reynolds_1st_scalar"), 0);
  //     h_ptr->stat_reynolds_2nd.allocate_memory(mx, my, 1, parameter.get_int("n_stat_reynolds_2nd"), 0);
  //     h_ptr->stat_favre_1st.allocate_memory(mx, my, 1, parameter.get_int("n_stat_favre_1st"), 0);
  //     h_ptr->stat_favre_2nd.allocate_memory(mx, my, 1, parameter.get_int("n_stat_favre_2nd"), 0);
  //   } else {
  //     h_ptr->stat_reynolds_1st.allocate_memory(
  //       mx, my, mz, parameter.get_int("n_stat_reynolds_1st") + parameter.get_int("n_stat_reynolds_1st_scalar"), 1);
  //     h_ptr->stat_reynolds_2nd.allocate_memory(mx, my, mz, parameter.get_int("n_stat_reynolds_2nd"), 1);
  //     h_ptr->stat_favre_1st.allocate_memory(mx, my, mz, parameter.get_int("n_stat_favre_1st"), 1);
  //     h_ptr->stat_favre_2nd.allocate_memory(mx, my, mz, parameter.get_int("n_stat_favre_2nd"), 1);
  //   }
  // }
  // // The collected data includes one layer of ghost mesh, which may be used to compute the gradients.
  // h_ptr->collect_reynolds_1st.allocate_memory(
  //   mx, my, mz, parameter.get_int("n_stat_reynolds_1st") + parameter.get_int("n_stat_reynolds_1st_scalar"), 1);
  // h_ptr->collect_reynolds_2nd.allocate_memory(mx, my, mz, parameter.get_int("n_stat_reynolds_2nd"), 1);
  // h_ptr->collect_favre_1st.allocate_memory(mx, my, mz, parameter.get_int("n_stat_favre_1st"), 1);
  // h_ptr->collect_favre_2nd.allocate_memory(mx, my, mz, parameter.get_int("n_stat_favre_2nd"), 1);

  // other statistics
  if (parameter.get_bool("stat_tke_budget")) {
    h_ptr->collect_tke_budget.allocate_memory(mx, my, mz, TkeBudget::n_collect, TkeBudget::ngg);
  }
  const int n_species_stat = parameter.get_int("n_species_stat");
  const int n_ps = parameter.get_int("n_ps");
  const int n_scalar_stat = n_species_stat + n_ps;
  if (n_scalar_stat > 0) {
    if (parameter.get_bool("stat_scalar_fluc_budget")) {
      h_ptr->collect_scalar_budget.allocate_memory(mx, my, mz, ScalarFlucBudget::n_collect * n_scalar_stat,
                                                   ScalarFlucBudget::ngg);
    }
    if (parameter.get_bool("stat_species_velocity_correlation")) {
      h_ptr->collect_scalar_vel_correlation.allocate_memory(
        mx, my, mz, ScalarVelocityCorrelation::n_collect * n_scalar_stat, ScalarVelocityCorrelation::ngg);
    }
    if (parameter.get_bool("stat_species_dissipation_rate")) {
      h_ptr->collect_scalar_budget.allocate_memory(
        mx, my, mz, ScalarDissipationRate::n_collect * n_scalar_stat, ScalarDissipationRate::ngg);
    }
  }
  // }

  if (parameter.get_bool("sponge_layer")) {
    h_ptr->sponge_mean_cv.allocate_memory(mx, my, mz, n_var, 0);
  }

  for (const auto ll: parameter.get_int_array("ov_labels")) {
    if (ll == 51) {
      h_ptr->mach.allocate_memory(mx, my, mz, ngg);
    } else if (ll == 52) {
      h_ptr->mut.allocate_memory(mx, my, mz, ngg);
    } else if (ll == 53) {
      // h_ptr->fd.allocate_memory(mx, my, mz, ngg);
    } else if (ll == 54) {
      h_ptr->scalar_diss_rate.allocate_memory(mx, my, mz, ngg);
    }
  }

  // Assign memory for auxiliary variables
  h_ptr->udv.allocate_memory(mx, my, mz, UserDefineIO::n_dynamic_auxiliary, ngg);

  cudaMalloc(&d_ptr, sizeof(DZone));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DZone), cudaMemcpyHostToDevice);

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("\tProcess [[%d]], the free memory is %7.3fGB, the used memory is %7.3fGB\n", parameter.get_int("myid"),
         static_cast<real>(free) / 1024 / 1024 / 1024, static_cast<real>(total - free) / 1024 / 1024 / 1024);
}

void cfd::Field::copy_data_from_device(const Parameter &parameter) {
  const auto size = (block.mx + 2 * block.ngg) * (block.my + 2 * block.ngg) * (block.mz + 2 * block.ngg);

  cudaMemcpy(bv.data(), h_ptr->bv.data(), 6 * size * sizeof(real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(ov.data(), h_ptr->mach.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  // if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
  //   cudaMemcpy(ov[1], h_ptr->mut.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  // }
  cudaMemcpy(sv.data(), h_ptr->sv.data(), parameter.get_int("n_scalar") * size * sizeof(real), cudaMemcpyDeviceToHost);
  // if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
  //   cudaMemcpy(ov[2], h_ptr->scalar_diss_rate.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  // }
  const int n_other_var = parameter.get_int("n_other_var");
  const auto &ov_labels = parameter.get_int_array("ov_labels");
  for (int l = 0; l < n_other_var; ++l) {
    const real *hPtr = nullptr;
    switch (ov_labels[l]) {
      case 51:
        hPtr = h_ptr->mach.data();
        break;
      case 52:
        hPtr = h_ptr->mut.data();
        break;
      case 53:
        // hPtr = h_ptr->fd.data();
        break;
      case 54:
        hPtr = h_ptr->scalar_diss_rate.data();
        break;
      case 56:
        hPtr = h_ptr->shock_sensor.data();
        break;
      default:
        hPtr = nullptr;
    }
    if (hPtr != nullptr) {
      cudaMemcpy(ov[l], hPtr, size * sizeof(real), cudaMemcpyDeviceToHost);
    }
  }

  if (const int n_rand = parameter.get_int("random_number_per_point"); n_rand > 0) {
    cudaMemcpy(rng_state.data(), h_ptr->rng_state.data(), n_rand * size * sizeof(real),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fluc_val.data(), h_ptr->fluc_val.data(),
               parameter.get_int("fluctuation_variable_number") * size * sizeof(real), cudaMemcpyDeviceToHost);
  }

  copy_auxiliary_data_from_device(*this, size);
}

void cfd::Field::deallocate_memory(const Parameter &parameter) {
  h_ptr->x.deallocate_memory();
  h_ptr->y.deallocate_memory();
  h_ptr->z.deallocate_memory();
  if (parameter.get_bool("turbulence") && parameter.get_int("turbulence_method") == 2) {
    h_ptr->des_delta.deallocate_memory();
  }
  h_ptr->jac.deallocate_memory();
  h_ptr->metric.deallocate_memory();

  h_ptr->cv.deallocate_memory();
  h_ptr->bv.deallocate_memory();
  h_ptr->bv_last.deallocate_memory();
  h_ptr->mach.deallocate_memory();
  h_ptr->mul.deallocate_memory();

  const auto n_scalar = parameter.get_int("n_scalar");
  if (n_scalar > 0)
    h_ptr->sv.deallocate_memory();
  const auto n_spec{parameter.get_int("n_spec")};
  if (n_spec > 0) {
    h_ptr->rho_D.deallocate_memory();
    h_ptr->acoustic_speed.deallocate_memory();
    h_ptr->thermal_conductivity.deallocate_memory();
    h_ptr->gamma.deallocate_memory();
    h_ptr->cp.deallocate_memory();
    if (parameter.get_int("reaction") == 1) {
      // Finite rate chemistry
      if (const int chemSrcMethod = parameter.get_int("chemSrcMethod"); chemSrcMethod == 1 || chemSrcMethod == 2) {
        // EPI
        h_ptr->chem_src_jac.deallocate_memory();
      }
    } else if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
      // Flamelet model
      h_ptr->scalar_diss_rate.deallocate_memory();
    }
  }
  if (parameter.get_int("turbulence_method") == 1 || parameter.get_int("turbulence_method") == 2) {
    // RANS method or DES method
    h_ptr->mut.deallocate_memory();
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      h_ptr->wall_distance.deallocate_memory();
      if (parameter.get_int("turb_implicit") == 1) {
        h_ptr->turb_src_jac.deallocate_memory();
      }
    }
  }
  if (parameter.get_int("if_compute_wall_distance") == 1) {
    h_ptr->wall_distance.deallocate_memory();
  }
  h_ptr->dq.deallocate_memory();
  if (!(!parameter.get_bool("steady") && parameter.get_int("temporal_scheme") == 3 &&
        parameter.get_bool("fixed_time_step"))) {
    h_ptr->inv_spectr_rad.deallocate_memory();
    h_ptr->visc_spectr_rad.deallocate_memory();
    h_ptr->dt_local.deallocate_memory();
  }
  if (parameter.get_int("implicit_method") == 1) { // DPLUR
    if (!(!parameter.get_bool("steady") && parameter.get_int("temporal_scheme") == 3)) {
      h_ptr->dq.deallocate_memory();
      h_ptr->dq0.deallocate_memory();
      h_ptr->dqk.deallocate_memory();
      h_ptr->inv_spectr_rad.deallocate_memory();
      h_ptr->visc_spectr_rad.deallocate_memory();
    }
  }
  if (parameter.get_int("inviscid_scheme") == 2) {
    // Roe scheme
    h_ptr->entropy_fix_delta.deallocate_memory();
  }

  if (!parameter.get_bool("steady")) {
    // unsteady simulation
    if (parameter.get_int("temporal_scheme") == 2) {
      // dual time stepping
      h_ptr->qn1.deallocate_memory();
      h_ptr->qn_star.deallocate_memory();
      h_ptr->in_last_step.deallocate_memory();
    }
    if (parameter.get_int("temporal_scheme") == 3) {
      // rk scheme
      h_ptr->qn.deallocate_memory();
    }
  }

  if (parameter.get_bool("if_collect_statistics")) {
    // If we need to collect the statistics, we need to allocate memory for the data.
    //    h_ptr->mean_value.deallocate_memory();
    //    h_ptr->reynolds_stress_tensor.deallocate_memory();
    //    h_ptr->user_defined_statistical_data.deallocate_memory();
    //    if (parameter.get_bool("perform_spanwise_average")) {
    //      h_ptr->mean_value_span_ave.deallocate_memory();
    //      h_ptr->reynolds_stress_tensor_span_ave.deallocate_memory();
    //      h_ptr->user_defined_statistical_data_span_ave.deallocate_memory();
    //    }
    //
    //    // The collected data includes one layer of ghost mesh, which may be used to compute the gradients.
    //    h_ptr->firstOrderMoment.deallocate_memory();
    //    h_ptr->velocity2ndMoment.deallocate_memory();
    //    h_ptr->userCollectForStat.deallocate_memory();
  }

  if (UserDefineIO::n_dynamic_auxiliary > 0) {
    h_ptr->udv.deallocate_memory();
  }

  cudaFree(&d_ptr);
  delete h_ptr;
}

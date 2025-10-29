#include "Parameter.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "gxl_lib/MyString.h"
#include <fmt/format.h>
#include "kernels.h"
#include "ChemData.h"

cfd::Parameter::Parameter(int *argc, char ***argv) {
  int myid, n_proc;
  MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  int_parameters["myid"] = myid;
  int_parameters["n_proc"] = n_proc;

  setup_default_settings();
  read_param_from_file();
  diagnose_parallel_info();
  setup_gpu_device(n_proc, myid);
  deduce_known_info();
}

cfd::Parameter::Parameter(const std::string &filename) {
  std::ifstream file(filename);
  read_one_file(file);
  file.close();
}

void cfd::Parameter::read_param_from_file() {
  for (auto &name: file_names) {
    std::ifstream file(name);
    read_one_file(file);
    file.close();
  }
}

void cfd::Parameter::read_one_file(std::ifstream &file) {
  std::string input{}, type{}, key{}, temp{};
  std::istringstream line(input);
  while (std::getline(file, input)) {
    if (input.starts_with("//") || input.starts_with("!") || input.empty()) {
      continue;
    }
    line.clear();
    line.str(input);
    line >> type;
    line >> key >> temp;
    if (type == "int") {
      int val{};
      line >> val;
      int_parameters[key] = val;
    } else if (type == "real") {
      real val{};
      line >> val;
      real_parameters[key] = val;
    } else if (type == "bool") {
      bool val{};
      line >> val;
      bool_parameters[key] = val;
    } else if (type == "string") {
      std::string val{};
      line >> val;
      string_parameters[key] = val;
    } else if (type == "array") {
      if (key == "int") {
        std::vector<int> arr;
        std::string name{temp};
        line >> temp >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        int_array[name] = arr;
      } else if (key == "real") {
        std::vector<real> arr;
        std::string name{temp};
        line >> temp >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        real_array[name] = arr;
      } else if (key == "string") {
        std::vector<std::string> arr;
        std::string name{temp};
        line >> temp >> temp;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        string_array[name] = arr;
      }
    } else if (type == "struct") {
      auto the_struct = read_struct(file);
      struct_array[key] = the_struct;
    } else if (type == "range") {
      std::string name{temp};
      line >> temp >> temp; // {
      if (key == "real") {
        std::vector<real> arr;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        auto num = arr.size();
        if (num == 2) {
          real_range[name] = Range<real>(arr[0], arr[1]);
        } else if (num == 4) {
          real_range[name] = Range<real>(arr[0], arr[1], arr[2], arr[3]);
        } else if (num == 6) {
          real_range[name] = Range<real>(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
        } else {
          fmt::print("The number of values in the range {} is not correct.\n", name);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      } else if (key == "int") {
        std::vector<int> arr;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        auto num = arr.size();
        if (num == 2) {
          int_range[name] = Range<int>(arr[0], arr[1]);
        } else if (num == 4) {
          int_range[name] = Range<int>(arr[0], arr[1], arr[2], arr[3]);
        } else if (num == 6) {
          int_range[name] = Range<int>(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
        } else {
          fmt::print("The number of values in the range {} is not correct.\n", name);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }
  }
  file.close();
}

template<typename T>
int cfd::Parameter::read_line_to_array(std::istringstream &line, std::vector<T> &arr) {
  std::string temp{};
  while (line >> temp) {
    if (temp == "}") {
      // Which means the array has been read
      return 0;
    }
    if (temp == "//") {
      // which means the array is not over, but values are on the next line
      break;
    }
    T val{};
    if constexpr (std::is_same_v<T, real>) {
      val = std::stod(temp);
    } else if constexpr (std::is_same_v<T, int>) {
      val = std::stoi(temp);
    } else if constexpr (std::is_same_v<T, std::string>) {
      val = temp;
    }
    arr.push_back(val);
  }
  return 1; // Which means we need to read the next line
}

std::map<std::string, std::variant<std::string, int, real>> cfd::Parameter::read_struct(std::ifstream &file) {
  std::map<std::string, std::variant<std::string, int, real>> struct_to_read;
  std::string input{}, key{}, temp{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(file, input, line)) {
    line >> key;
    if (key == "}") { // the "}" must be placed on a separate line.
      break;
    }
    if (key == "string") {
      std::string val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "int") {
      int val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "real") {
      real val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    }
  }
  return struct_to_read;
}

void cfd::Parameter::deduce_known_info() {
  int_parameters.emplace("step", 0);

  // Ghost grid number is decided from the inviscid scheme and viscous scheme.
  // For common 2nd order schemes, we need 2 ghost grids.
  // Here, because we now only implement 2nd order central difference for viscous flux, we need 2 ghost grids.
  // The main concern for the number of ghost grids is the inviscid flux, which in turn is decided from the reconstruction method.
  int ngg{2};
  int reconstruction_scheme = get_int("reconstruction");
  std::string reconstruction_name{};
  if (reconstruction_scheme == 1) {
    reconstruction_name = "1st-order upwind";
  } else if (reconstruction_scheme == 2) {
    reconstruction_name = "MUSCL";
  } else if (reconstruction_scheme == 3) {
    reconstruction_name = "NND2";
  }

  // Next, based on the reconstruction scheme and inviscid flux method, we re-assign a new field called "inviscid_tag"
  // to identify the method to be used.
  // 2-Roe, 3-AUSM+, 4-HLLC. These are used by default if the corresponding reconstruction methods are 1stOrder, MUSCL, or NND2.
  // Once the reconstruction method is changed to high-order ones, such as WENO5, WENO7, etc., we would use a new tag to identify
  // them. Because the reconstructed variables would be conservative variables, and there may be additional operations,
  // such as characteristic reconstruction, high-order central difference, etc.
  // Summary: 2-Roe, 3-AUSM+, 4-HLLC, 14-HLLC+WENO
  int inviscid_type{0};
  auto hybrid_inviscid_scheme{get_string("hybrid_inviscid_scheme")};
  if (hybrid_inviscid_scheme == "UD7+WENO7") {
    int inviscid_scheme{get_int("inviscid_scheme")};
    if (inviscid_scheme != 71 && inviscid_scheme != 72)
      update_parameter("inviscid_scheme", 72);
    inviscid_type = 6;
    ngg = 4;
  } else {
    // Not hybrid scheme
    int inviscid_scheme{get_int("inviscid_scheme")};
    if (inviscid_scheme == 2) {
      inviscid_type = 2;
    }
    if (reconstruction_scheme == 4 || reconstruction_scheme == 5) {
      inviscid_type = 1;
      // WENO reconstructions
    }
    if (inviscid_scheme == 51 || inviscid_scheme == 52) {
      if (get_real("shockSensor_threshold") > 1e-10) {
        // If the shock sensor is set to 0 or negative value, the WENO7 is used throughout
        // If not, the hybrid scheme is used
        hybrid_inviscid_scheme = "UD5+WENO5";
        update_parameter("hybrid_inviscid_scheme", hybrid_inviscid_scheme);
      }
      inviscid_type = 3;
      ngg = 3;
    }
    if (inviscid_scheme == 71 || inviscid_scheme == 72) {
      if (get_real("shockSensor_threshold") > 1e-10) {
        // If the shock sensor is set to 0 or negative value, the WENO7 is used throughout
        // If not, the hybrid scheme is used
        hybrid_inviscid_scheme = "UD7+WENO7";
        update_parameter("hybrid_inviscid_scheme", hybrid_inviscid_scheme);
      }
      inviscid_type = 3;
      ngg = 4;
    }
    if (inviscid_scheme == 6) {
      inviscid_type = 4;
    }
  }
  update_parameter("inviscid_type", inviscid_type);

  int inviscid_scheme{get_int("inviscid_scheme")};
  if (int_parameters["myid"] == 0)
    fmt::print("\n{:*^80}\n", "Inviscid scheme Information");
  std::string inviscid_scheme_name{};
  if (hybrid_inviscid_scheme != "NO") {
    inviscid_scheme_name = hybrid_inviscid_scheme;
  } else {
    if (inviscid_scheme == 2) {
      inviscid_scheme_name = "Roe";
      if (int_parameters["myid"] == 0)
        fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
    } else if (inviscid_scheme == 3) {
      inviscid_scheme_name = "AUSM+";
      if (int_parameters["myid"] == 0)
        fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
    } else if (inviscid_scheme == 4) {
      inviscid_scheme_name = "HLLC";
      if (int_parameters["myid"] == 0)
        fmt::print("\t->-> {:<20} : reconstruction method.\n", reconstruction_name);
    } else if (inviscid_scheme == 51) {
      inviscid_scheme_name = "WENO5-cp";
      ngg = 3;
    } else if (inviscid_scheme == 52) {
      inviscid_scheme_name = "WENO5-ch";
      ngg = 3;
    } else if (inviscid_scheme == 71) {
      inviscid_scheme_name = "WENO7-cp";
      ngg = 4;
    } else if (inviscid_scheme == 72) {
      inviscid_scheme_name = "WENO7-ch";
      ngg = 4;
    } else if (inviscid_scheme == 6) {
      inviscid_scheme_name = "energy-preserving-6";
      ngg = 3;
    }
  }

  if (int viscous_order = get_int("viscous_order"); viscous_order == 6) {
    if (ngg < 3) ngg = 3;
  } else if (viscous_order == 8) {
    if (ngg < 4) ngg = 4;
  }

  update_parameter("ngg", ngg);
  if (int_parameters["myid"] == 0) {
    fmt::print("\t->-> {:<20} : inviscid scheme.\n", inviscid_scheme_name);
    fmt::print("\t->-> {:<20} : number of ghost layers\n", ngg);
    if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 71 || inviscid_scheme == 72) {
      if (bool_parameters["positive_preserving"] == 1) {
        fmt::print("\t->-> {:<20} : positive preserving.\n", "Yes");
      }
    }
  }

  if (bool_parameters["steady"] == 0 && int_parameters["temporal_scheme"] == 3) {
    // RK-3, the chemical source should be treated explicitly.
    update_parameter("chemSrcMethod", 0);
  }

  update_parameter("n_var", 5);
  update_parameter("n_turb", 0);
  int n_scalar{0};
  if (bool_parameters["turbulence"] == 1) {
    if (int_parameters["turbulence_method"] == 1 || int_parameters["turbulence_method"] == 2) { // RANS or DES
      if (int_parameters["RANS_model"] == 1) {                                                  // SA
        update_parameter("n_turb", 1);
        update_parameter("n_var", 5 + 1);
        n_scalar += 1;
      } else { // SST
        update_parameter("n_turb", 2);
        update_parameter("n_var", 5 + 2);
        n_scalar += 2;
      }
    }
  } else {
    update_parameter("RANS_model", 0);
  }
  update_parameter("n_scalar", n_scalar);
}

void cfd::Parameter::setup_default_settings() {
  int_parameters["problem_type"] = 0;
  string_parameters["canonical_problem"] = "default";

  int_parameters["mesh_file_type"] = 0;
  int_parameters["gridIsBinary"] = 0;
  real_parameters["gridScale"] = 1.0;
  int_parameters["if_compute_wall_distance"] = 0;
  int_parameters["wall_distance"] = 0;
  int_parameters["symmetric_conservative_metric"] = 0;

  bool_parameters["steady"] = true;               // Steady simulation is the default choice
  int_parameters["implicit_method"] = 0;          // Explicit is used by default
  int_parameters["DPLUR_inner_step"] = 3;         // The default DPLUR inner iterations.
  real_parameters["convergence_criteria"] = 1e-8; // The criteria of convergence

  // If we conduct transient simulations, the following parameters are used by default.
  int_parameters["temporal_scheme"] = 3;      // RK3 is used by default
  bool_parameters["fixed_time_step"] = false; // The time step is computed with CFL condition.
  real_parameters["dt"] = 1e-8;
  real_parameters["n_flowThroughTime"] = -1;
  real_parameters["domain_length"] = 1.0;
  real_parameters["characteristic_velocity"] = -1;
  real_parameters["set_current_physical_time"] = -1;

  bool_parameters["steady_before_transient"] = false;
  // Whether to conduct a steady simulation before transient simulation.

  // If dual-time stepping is used, the following parameters are needed.
  int_parameters["inner_iteration"] = 20;
  int_parameters["max_inner_iteration"] = 30;
  int_parameters["iteration_adjust_step"] = 20;

  // For spatial discretization
  int_parameters["inviscid_scheme"] = 3;         // AUSM+
  int_parameters["reconstruction"] = 2;          // MUSCL
  int_parameters["limiter"] = 0;                 // minmod
  real_parameters["entropy_fix_factor"] = 0.125; // For Roe scheme, we need an entropy fix factor.
  int_parameters["viscous_order"] = 2;           // The default viscous order is 2.
  bool_parameters["gradPInDiffusionFlux"] = false;
  string_parameters["hybrid_inviscid_scheme"] = "NO";
  int_parameters["shock_sensor"] = 0;
  real_parameters["shockSensor_threshold"] = 0;
  real_parameters["shockSensor_epsilon"] = -1;

  // When WENO is used, pp limiter is a choice
  bool_parameters["positive_preserving"] = false;

  int_parameters["species"] = 0;
  int_parameters["reaction"] = 0;
  int_parameters["hardCodedMech"] = 0;
  string_parameters["therm_file"] = "chemistry/therm.dat";
  string_parameters["transport_file"] = "chemistry/tran.dat";
  int_parameters["chemSrcMethod"] = 0; // explicit treatment of the chemical source term is used by default.
  real_parameters["c_chi"] = 1;

  bool_parameters["turbulence"] = false;            // laminar
  int_parameters["turbulence_method"] = 1;          // 1-RAS, 2-DES
  int_parameters["RANS_model"] = 2;                 // 1-SA, 2-SST
  int_parameters["turb_implicit"] = 1;              // turbulent source term is treated implicitly by default
  int_parameters["compressibility_correction"] = 0; // No compressibility correction is added by default
  int_parameters["des_scale_method"] = 0;
  // How to compute the grid scale in DES simulations. 0 - cubic root of cell volume, 1 - max of cell dimension

  int_parameters["n_ps"] = 0;

  int_parameters["n_profile"] = 0;
  string_array["profile_related_bc_names"] = {};
  string_array["profile_file_names"] = {};

  int_parameters["groups_init"] = 1;
  string_parameters["default_init"] = "freestream";

  int_parameters["diffusivity_method"] = 1;
  real_parameters["schmidt_number"] = 0.5;
  real_parameters["prandtl_number"] = 0.72;
  real_parameters["turbulent_prandtl_number"] = 0.9;
  real_parameters["turbulent_schmidt_number"] = 0.9;

  bool_parameters["if_collect_statistics"] = false;
  bool_parameters["if_continue_collect_statistics"] = false;
  int_parameters["start_collect_statistics_iter"] = 0;
  bool_parameters["perform_spanwise_average"] = false;
  bool_parameters["output_statistics_plt"] = true;

  int_array["post_process"] = {};
  int_array["output_bc"] = {};

  int_parameters["if_monitor_points"] = 0; // 0 - no monitor, 1 - monitor
  string_parameters["monitor_file"] = "input/monitor_points.txt";
  string_array["monitor_var"] = {"density", "u", "v", "w", "pressure", "temperature"};
  int_parameters["if_monitor_blocks"] = 0; // 0 - no monitor, 1 - monitor
  string_parameters["monitor_block_file"] = "input/monitor_blocks.txt";
  int_parameters["monitor_block_frequency"] = 100;
  string_array["monitor_block_var"] = {"u", "v", "w", "pressure"};

  int_parameters["n_inflow_fluctuation"] = 0;
  int_array["need_rng"] = {};
  int_array["need_fluctuation_profile"] = {};
  bool_parameters["wall_white_noise"] = false;
  string_array["fluctuation_profile_file"] = {};
  string_array["fluctuation_profile_related_bc_name"] = {};
  bool_parameters["perform_spanwise_average"] = false;
  bool_parameters["positive_preserving"] = false;
  bool_parameters["need_physical_time"] = false;

  bool_parameters["use_df"] = false;
  int_array["df_label"] = {};

  bool_parameters["sponge_layer"] = false;
  int_parameters["sponge_function"] = 0; // 0 - (Nektar++, CPC, 2024)
  int_array["sponge_layer_direction"] = {};
  int_parameters["sponge_iter"] = 0;
  int_array["sponge_scalar_iter"] = {};
  real_parameters["spongeXMinusStart"] = 0;
  real_parameters["spongeXMinusEnd"] = 0;
  real_parameters["spongeXPlusStart"] = 0;
  real_parameters["spongeXPlusEnd"] = 0;
  real_parameters["spongeYMinusStart"] = 0;
  real_parameters["spongeYMinusEnd"] = 0;
  real_parameters["spongeYPlusStart"] = 0;
  real_parameters["spongeYPlusEnd"] = 0;
  real_parameters["spongeZMinusStart"] = 0;
  real_parameters["spongeZMinusEnd"] = 0;
  real_parameters["spongeZPlusStart"] = 0;
  real_parameters["spongeZPlusEnd"] = 0;

  int_parameters["characteristic_velocity_ml"] = 1;
  bool_parameters["compatible_mixing_layer"] = false;
  real_parameters["velocity_ratio"] = -1;

  int_parameters["output_time_series"] = 0;
  bool_parameters["limit_flow"] = false;

  // flamelet model
  int_parameters["flamelet_format"] = 0;
  string_parameters["flamelet_file_name"] = "flamelet-lib-zzprimx.txt";

  // output control
  struct_array["other_output_variables"] = {};

  // stat control
  string_array["stat_rey_1st"] = {};
  string_array["stat_species"] = {};
  bool_parameters["rho_p_correlation"] = false;
  bool_parameters["stat_tke_budget"] = false;
  bool_parameters["stat_scalar_fluc_budget"] = false;
  bool_parameters["stat_species_velocity_correlation"] = false;
  bool_parameters["stat_species_dissipation_rate"] = false;

  // info about additional variables
  string_array["available_field_var"] = {};

  int_parameters["fluctuation_form"] = 0;
  int_parameters["random_number_per_point"] = 0;
  int_parameters["fluctuation_variable_number"] = 0;
  real_parameters["fluctuation_intensity"] = 0;
  int_parameters["N_spanwise_waves"] = 1;
  real_parameters["x0_fluc"] = 0;
  real_parameters["y0_fluc"] = 0;
  real_parameters["z0_fluc"] = 0;
}

void cfd::Parameter::diagnose_parallel_info() {
  if (int_parameters["myid"] == 0) {
    fmt::print("{:*^80}\n", "Parallel Information");
    const bool parallel = bool_parameters["parallel"];
    const int n_proc = int_parameters["n_proc"];
    if (parallel) {
      if (n_proc == 1) {
        fmt::print("You chose parallel computation, but the number of processes is equal to 1.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fmt::print("\t->-> Parallel computation chosen! Number of processes: ->-> {}. \n", n_proc);
    } else {
      if (n_proc > 1) {
        fmt::print("You chose serial computation, but the number of processes is not equal to 1, n_proc={}.\n",
                   n_proc);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fmt::print("\t->-> Serial computation chosen!\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void cfd::Parameter::deduce_sim_info(const Species &spec) {
  int n_var = 5, n_scalar = 0, n_scalar_transported = 0, n_other_var = 1;
  int i_turb_cv = 5, i_fl_cv = 0;
  int i_ps = 0, i_ps_cv = 5; // ps - passive scalar

  if (get_int("species") == 1) {
    n_scalar += get_int("n_spec");
    i_ps += get_int("n_spec");
    if (get_int("reaction") != 2) {
      // Mixture / Finite rate chemistry
      n_scalar_transported += get_int("n_spec");
      i_turb_cv += get_int("n_spec");
      i_ps_cv += get_int("n_spec");
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
      i_fl_cv = 5 + get_int("n_turb");
      i_ps_cv = i_fl_cv + 2;
      ++n_other_var; // scalar dissipation rate
    }
  } else if (get_int("species") == 2) {
    n_scalar += get_int("n_spec") + 2;
    i_ps += get_int("n_spec");
    if (get_int("reaction") != 2) {
      // Mixture with mixture fraction and variance solved.
      n_scalar_transported += get_int("n_spec") + 2;
      i_turb_cv += get_int("n_spec");
      i_fl_cv = i_turb_cv + get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      i_fl_cv = 5 + get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
    i_ps_cv = i_fl_cv + 2;
  }
  if (get_bool("turbulence")) {
    // turbulence simulation
    if (const auto turb_method = get_int("turbulence_method"); turb_method == 1 || turb_method == 2) {
      // RANS or DES type
      n_scalar_transported += get_int("n_turb");
      n_scalar += get_int("n_turb");
      i_ps += get_int("n_turb");
      i_ps_cv += get_int("n_turb");
      ++n_other_var; // mut
    }
  }
  if (get_int("n_ps")) {
    // Some passive scalar is transported
    n_scalar_transported += get_int("n_ps");
    n_scalar += get_int("n_ps");
  }
  n_var += n_scalar_transported;
  update_parameter("n_var", n_var);
  update_parameter("n_scalar", n_scalar);
  update_parameter("n_scalar_transported", n_scalar_transported);
  update_parameter("i_turb_cv", i_turb_cv);
  update_parameter("i_fl", get_int("n_turb") + get_int("n_spec"));
  update_parameter("i_fl_cv", i_fl_cv);
  update_parameter("i_ps", i_ps);
  update_parameter("i_ps_cv", i_ps_cv);
  n_other_var += static_cast<int>(get_string_array("available_field_var").size());
  update_parameter("n_other_var", n_other_var);

  get_variable_names(spec);

  // Convert canonical problem names to internal labels
  auto canonical_problem = get_string("canonical_problem");
  canonical_problem = gxl::to_upper(canonical_problem);
  int problem_type = 0;
  if (canonical_problem == "MIXINGLAYER") {
    problem_type = 1;
  } else if (canonical_problem == "JICF") {
    problem_type = 2;
  }
  update_parameter("problem_type", problem_type);
  // printf("canonical problem = %s, problem type = %d\n", canonical_problem.c_str(), problem_type);

  const int myid = get_int("myid");
  if (myid == 0) {
    fmt::print("\n{:*^80}\n", "Simulation Details");
    printf("\t->-> %-20s : canonical problem\n", canonical_problem.c_str());
    printf("\t->-> %-20d : number of equations to solve\n", n_var);
    printf("\t->-> %-20d : number of scalar variables\n", n_scalar);
    if (const auto n_ps = get_int("n_ps"); n_ps > 0) {
      printf("\t->-> %-20d : number of passive scalar variables\n", n_ps);
    }

    if (get_bool("steady")) {
      // steady
      fmt::print("\n\t->-> {:<20} : flow simulation will be conducted.\n", "Steady");
      fmt::print("\t\t->-> {:<20} : CFL number\n", get_real("cfl"));
      if (int_parameters["implicit_method"] == 0) {
        // explicit
        fmt::print("\t\t->-> {:<20} : time integration\n", "explicit");
      } else {
        // implicit
        fmt::print("\t\t->-> {:<20} : time integration\n", "implicit DPLUR");
        fmt::print("\t\t->-> {:<20} : inner iterations for DPLUR\n", get_int("DPLUR_inner_step"));
      }
      fmt::print("\t\t->-> {:<20} : convergence criteria\n", get_real("convergence_criteria"));
    } else {
      // transient simulation
      fmt::print("\t->-> {:<20} : flow simulation will be conducted.\n", "Transient");
      if (int_parameters["temporal_scheme"] == 2) {
        // dual-time stepping
        fmt::print("\t\t->-> {:<20} : temporal scheme\n", "Dual-time stepping");
        fmt::print("\t\t->-> {:<20} : inner iterations for dual-time stepping\n", get_int("inner_iteration"));
        fmt::print("\t\t->-> {:<20} : max inner iterations for dual-time stepping\n", get_int("max_inner_iteration"));
        fmt::print("\t\t->-> {:<20} : iteration adjust step\n", get_int("iteration_adjust_step"));
      } else if (int_parameters["temporal_scheme"] == 3) {
        // RK-3
        fmt::print("\t\t->-> {:<25} : temporal scheme\n", "3rd-order Runge-Kutta");
        if (bool_parameters["fixed_time_step"]) {
          fmt::print("\t\t->-> {:<25} : physical time step(s)\n", get_real("dt"));
        } else {
          fmt::print("\t\t->-> {:<25} : CFL number\n", get_real("cfl"));
        }
      }
    }

    fmt::print("\n\t->-> {:<20} : order CDS for viscous flux\n", get_int("viscous_order"));

    if (get_bool("turbulence")) {
      bool need_ras_model{false};
      if (const auto method = get_int("turbulence_method"); method == 1) {
        fmt::print("\n\t->-> {:<20} : simulation\n", "RAS");
        need_ras_model = true;
      } else if (method == 2) {
        fmt::print("\n\t->-> {:<20} : simulation\n", "DDES");
        need_ras_model = true;
      }
      if (need_ras_model) {
        if (get_int("RANS_model") == 1) {
          fmt::print("\t\t->-> {:<20} : model\n", "SA");
        } else if (get_int("RANS_model") == 2) {
          fmt::print("\t\t->-> {:<20} : model\n", "k-omega SST");
        }
        if (int_parameters["turb_implicit"]) {
          fmt::print("\t\t->-> {:<20} : turbulence source term treatment\n", "implicit");
        }
        std::string cc_method{};
        bool cc_flag = false;
        if (const auto cc = get_int("compressibility_correction"); cc == 1) {
          cc_flag = true;
          cc_method = "Wilcox";
        } else if (cc == 2) {
          cc_flag = true;
          cc_method = "Sarkar";
        } else if (cc == 3) {
          cc_flag = true;
          cc_method = "Zeman";
        }
        if (cc_flag)
          fmt::print("\t\t->-> {:<20} : compressibility correction\n", cc_method);
      }
    }

    if (get_bool("sponge_layer")) {
      fmt::print("\n\t->-> {:<20} : sponge layer\n", "With");
      const auto dirs = int_array["sponge_layer_direction"];
      for (const auto dir: dirs) {
        if (dir == 0) {
          fmt::print("\t\t->-> {:<20} : direction\n", "x-");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeXMinusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeXMinusEnd"));
        } else if (dir == 1) {
          fmt::print("\t\t->-> {:<20} : direction\n", "x+");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeXPlusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeXPlusEnd"));
        } else if (dir == 2) {
          fmt::print("\t\t->-> {:<20} : direction\n", "y-");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeYMinusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeYMinusEnd"));
        } else if (dir == 3) {
          fmt::print("\t\t->-> {:<20} : direction\n", "y+");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeYPlusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeYPlusEnd"));
        } else if (dir == 4) {
          fmt::print("\t\t->-> {:<20} : direction\n", "z-");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeZMinusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeZMinusEnd"));
        } else if (dir == 5) {
          fmt::print("\t\t->-> {:<20} : direction\n", "z+");
          fmt::print("\t\t\t->-> {:<20} : start\n", get_real("spongeZPlusStart"));
          fmt::print("\t\t\t->-> {:<20} : end\n", get_real("spongeZPlusEnd"));
        }
      }
    }
  }
}

void cfd::Parameter::get_variable_names(const Species &spec) {
  // First, basic variables.
  std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature"};
  VNs = {
    {"X", -3},
    {"Y", -2},
    {"Z", -1},
    // Basic variables
    {"RHO", 0},
    {"DENSITY", 0},
    {"U", 1},
    {"V", 2},
    {"W", 3},
    {"P", 4},
    {"PRESSURE", 4},
    {"T", 5},
    {"TEMPERATURE", 5},
  };
  int nv = static_cast<int>(var_name.size());
  if (get_int("species") != 0) {
    const int ns = spec.n_spec;
    var_name.resize(nv + ns);
    for (auto &names = spec.spec_list; auto &[name, ind]: names) {
      var_name[ind + nv] = name;
      auto nn = name;
      nn = gxl::to_upper(nn);
      VNs[nn] = 1000 + ind;
    }
    nv += ns;
  }
  if (get_bool("turbulence")) {
    if (const auto turb_method = get_int("turbulence_method"); turb_method == 1 || turb_method == 2) {
      // RAS/DDES
      if (const int ras_model = get_int("RANS_model"); ras_model == 1) {
        nv += 1;
        var_name.emplace_back("nuTilde");

        VNs["NUTILDE"] = VNsDefault.at("NUTILDE");
      } else {
        nv += 2;
        var_name.emplace_back("tke");
        var_name.emplace_back("omega");

        VNs["K"] = VNsDefault.at("K");
        VNs["TKE"] = VNsDefault.at("TKE");
        VNs["OMEGA"] = VNsDefault.at("OMEGA");
      }

      // The flamelet model must be used with RANS or DES, thus the if is contained in this branch.
      if (get_int("species") != 0 && get_int("reaction") == 2) {
        // Flamelet model
        nv += 2;
        var_name.emplace_back("MixtureFraction");
        var_name.emplace_back("MixtureFractionVariance");

        VNs["MIXTUREFRACTION"] = VNsDefault.at("MIXTUREFRACTION");
        VNs["MIXTUREFRACTIONVARIANCE"] = VNsDefault.at("MIXTUREFRACTIONVARIANCE");
      }
    }
  }
  if (const int n_ps = get_int("n_ps"); n_ps > 0) {
    nv += n_ps;
    for (int i = 0; i < n_ps; ++i) {
      var_name.emplace_back("PS" + std::to_string(i + 1));

      VNs["PS" + std::to_string(i + 1)] = 100 + i;
    }
  }

  // Next, additional variables with assigned memory.
  std::vector<int> ov_labels;
  var_name.emplace_back("mach");
  ov_labels.emplace_back(VNsDefault.at("MACH"));
  ++nv;
  if (const auto turb_method = get_int("turbulence_method"); turb_method == 1 || turb_method == 2) {
    // RAS/DDES
    var_name.emplace_back("mut");
    ov_labels.emplace_back(VNsDefault.at("MUT"));
    ++nv;
    if (turb_method == 2) {
      var_name.emplace_back("fd");
      ov_labels.emplace_back(VNsDefault.at("FD"));
      ++nv;
    }
    if (get_int("species") != 0 && get_int("reaction") == 2) {
      // Flamelet model
      var_name.emplace_back("ScalarDissipationRate");
      ov_labels.emplace_back(VNsDefault.at("SCALARDISSIPATIONRATE"));
      ++nv;
    }
  }

  // Last, variables to be computed by choice.
  if (auto ovs = struct_array["other_output_variables"]; ovs.find("laminar_viscosity") != ovs.end()) {
    var_name.emplace_back("laminar_viscosity");
    update_parameter("output_mul", true);
    ++nv;
  }
  if (const auto ovs = string_array["available_field_var"]; !ovs.empty()) {
    for (auto OV: ovs) {
      auto ov = OV;           // Original variable name
      OV = gxl::to_upper(OV); // Upper case
      var_name.emplace_back(ov);
      ov_labels.emplace_back(VNsDefault.at(OV));
      ++nv;
    }
  }

  update_parameter("var_name", var_name);
  update_parameter("ov_labels", ov_labels);
}

std::vector<int>
cfd::Parameter::identify_variable_labels(std::vector<std::string> &var_name, const Species &species) const {
  std::vector<int> labels;

  for (auto &name: var_name) {
    int l = 999;
    auto NAME = gxl::to_upper(name);

    bool found = false;
    for (const auto &[nm, lb]: VNs) {
      if (NAME == nm) {
        l = lb;
        found = true;
        break;
      }
    }
    if (!found && species.n_spec > 0) {
      for (const auto &spec_name = species.spec_list; const auto &[spec, sp_label]: spec_name) {
        if (NAME == gxl::to_upper(spec)) {
          l = 1000 + sp_label;
          found = true;
          break;
        }
      }
    }
    if (!found && get_int("n_ps") > 0) {
      for (int i = 0; i < get_int("n_ps"); ++i) {
        if (NAME == "PS" + std::to_string(i + 1)) {
          l = 100 + i;
          // found = true;
          break;
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

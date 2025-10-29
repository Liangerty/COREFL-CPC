#include "Parameter.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "gxl_lib/MyString.h"

cfd::Parameter::Parameter(int *argc, char ***argv) {
}

cfd::Parameter::Parameter(const std::string &filename) {
  int_parameters["GridgenOrPointwise"] = 1;
  int_parameters["writeBinary"] = 1;
  bool_parameters["setZ"] = false;
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

  int_parameters.emplace("step", 0);

  deduce_known_info();

  update_parameter("n_var", 5);
  update_parameter("n_turb", 0);
  int n_scalar{0};
  if (bool_parameters["turbulence"] == 1) {
    if (int_parameters["turbulence_method"] == 1 || int_parameters["turbulence_method"] == 2) { // RANS or DES
      if (int_parameters["RANS_model"] == 1) {// SA
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
//          fmt::print("The number of values in the range {} is not correct.\n", name);
//          MPI_Abort(MPI_COMM_WORLD, 1);
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
//          fmt::print("The number of values in the range {} is not correct.\n", name);
//          MPI_Abort(MPI_COMM_WORLD, 1);
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
  // Ghost grid number is decided from the inviscid scheme and viscous scheme.
  // For common 2nd order schemes, we need 2 ghost grids.
  // Here, because we now only implement 2nd order central difference for viscous flux, we need 2 ghost grids.
  // The main concern for the number of ghost grids is the inviscid flux, which in turn is decided from the reconstruction method.
  int ngg{2};
  int reconstruction_scheme = get_int("reconstruction");
  switch (reconstruction_scheme) {
    case 4: // WENO5
      ngg = 3;
      break;
    case 5: // WENO7
      ngg = 4;
      break;
    default: // 1st, MUSCL, NND2, etc.
      break;
  }

  int inviscid_scheme{get_int("inviscid_scheme")};
  if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 6) {
    ngg = 3;
  } else if (inviscid_scheme == 71 || inviscid_scheme == 72) {
    ngg = 4;
  }

  update_parameter("ngg", ngg);

  // Next, based on the reconstruction scheme and inviscid flux method, we re-assign a new field called "inviscid_tag"
  // to identify the method to be used.
  // 2-Roe, 3-AUSM+, 4-HLLC. These are used by default if the corresponding reconstruction methods are 1stOrder, MUSCL, or NND2.
  // Once the reconstruction method is changed to high-order ones, such as WENO5, WENO7, etc., we would use a new tag to identify
  // them. Because, the reconstructed variables would be conservative variables, and there may be additional operations,
  // such as characteristic reconstruction, high-order central difference, etc.
  // Summary: 2-Roe, 3-AUSM+, 4-HLLC, 14-HLLC+WENO
  int inviscid_tag{get_int("inviscid_scheme")};
  int inviscid_type{0};
  if (inviscid_tag == 2) {
    inviscid_type = 2;
  }
  if (reconstruction_scheme == 4 || reconstruction_scheme == 5) {
    inviscid_type = 1;
    // WENO reconstructions
  }
  if (inviscid_scheme == 51 || inviscid_scheme == 52 || inviscid_scheme == 71 || inviscid_scheme == 72) {
    inviscid_type = 3;
  }
  if (inviscid_scheme == 6) {
    inviscid_type = 4;
  }
//  update_parameter("inviscid_tag", inviscid_tag);
  update_parameter("inviscid_type", inviscid_type);

  if (bool_parameters["steady"] == 0 && int_parameters["temporal_scheme"] == 3) {
    // RK-3, the chemical source should be treated explicitly.
    update_parameter("chemSrcMethod", 0);
  }
}

void cfd::Parameter::setup_default_settings() {
  int_parameters["problem_type"] = 0;

  int_parameters["mesh_file_type"] = 0;
  real_parameters["gridScale"] = 1.0;
  int_parameters["if_compute_wall_distance"] = 0;
  int_parameters["wall_distance"] = 0;

  bool_parameters["steady"] = true; // Steady simulation is the default choice
  int_parameters["implicit_method"] = 0; // Explicit is used by default
  int_parameters["DPLUR_inner_step"] = 3; // The default DPLUR inner iterations.
  real_parameters["convergence_criteria"] = 1e-8; // The criteria of convergence

  // If we conduct transient simulations, the following parameters are used by default.
  int_parameters["temporal_scheme"] = 3; // RK3 is used by default
  bool_parameters["fixed_time_step"] = false; // The time step is computed with CFL condition.

  // If dual-time stepping is used, the following parameters are needed.
  int_parameters["inner_iteration"] = 20;
  int_parameters["max_inner_iteration"] = 30;
  int_parameters["iteration_adjust_step"] = 20;

  // For spatial discretization
  int_parameters["inviscid_scheme"] = 3; // AUSM+
  int_parameters["reconstruction"] = 2; // MUSCL
  int_parameters["limiter"] = 0; //minmod
  real_parameters["entropy_fix_factor"] = 0.125; // For Roe scheme, we need an entropy fix factor.
  int_parameters["viscous_order"] = 2; // Default viscous order is 2.
  bool_parameters["gradPInDiffusionFlux"] = false;

  // When WENO is used, pp limiter is a choice
  bool_parameters["positive_preserving"] = false;

  int_parameters["species"] = 0;
  int_parameters["reaction"] = 0;
  string_parameters["therm_file"] = "chemistry/therm.dat";
  string_parameters["transport_file"] = "chemistry/tran.dat";
  int_parameters["chemSrcMethod"] = 0; // explicit treatment of the chemical source term is used by default.
  real_parameters["c_chi"] = 1;

  bool_parameters["turbulence"] = false;// laminar
  int_parameters["turbulence_method"] = 1;// 1-RAS, 2-DES
  int_parameters["RANS_model"] = 2; // 1-SA, 2-SST
  int_parameters["turb_implicit"] = 1; // turbulent source term is treated implicitly by default
  int_parameters["compressibility_correction"] = 0; // No compressibility correction is added by default
  int_parameters["des_scale_method"] = 0; // How to compute the grid scale in DES simulations. 0 - cubic root of cell volume, 1 - max of cell dimension

  int_parameters["n_profile"] = 0;

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

  int_array["post_process"] = {};
  int_array["output_bc"] = {};

  int_parameters["if_monitor"] = 0; // 0 - no monitor, 1 - monitor
  string_parameters["monitor_file"] = "input/monitor_points.txt";
  string_array["monitor_var"] = {"density", "u", "v", "w", "pressure", "temperature"};

  int_parameters["n_inflow_fluctuation"] = 0;
  int_array["need_rng"] = {};
  int_array["need_fluctuation_profile"] = {};
  string_array["fluctuation_profile_file"] = {};
  string_array["fluctuation_profile_related_bc_name"] = {};
  bool_parameters["perform_spanwise_average"] = false;
  bool_parameters["positive_preserving"] = false;
}

//void cfd::Parameter::diagnose_parallel_info() {
//  if (int_parameters["myid"] == 0) {
//    const bool parallel = bool_parameters["parallel"];
//    const int n_proc = int_parameters["n_proc"];
//    if (parallel) {
//      if (n_proc == 1) {
//        fmt::print("You chose parallel computation, but the number of processes is equal to 1.\n");
//        MPI_Abort(MPI_COMM_WORLD, 1);
//      }
//      fmt::print("Parallel computation chosen! Number of processes: {}. \n", n_proc);
//    } else {
//      if (n_proc > 1) {
//        fmt::print("You chose serial computation, but the number of processes is not equal to 1, n_proc={}.\n",
//                   n_proc);
//        MPI_Abort(MPI_COMM_WORLD, 1);
//      }
//      fmt::print("Serial computation chosen!\n");
//    }
//  }
//  MPI_Barrier(MPI_COMM_WORLD);
//}

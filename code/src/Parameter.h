#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <array>
#include "Define.h"
#include <map>
#include <variant>

namespace cfd {
template<typename T>
struct Range {
  T xs, xe, ys, ye, zs, ze;
};

struct Species;

class Parameter {
  std::unordered_map<std::string, int> int_parameters{};
  std::unordered_map<std::string, real> real_parameters{};
  std::unordered_map<std::string, bool> bool_parameters{};
  std::unordered_map<std::string, std::string> string_parameters{};
  std::unordered_map<std::string, std::vector<int>> int_array{};
  std::unordered_map<std::string, std::vector<real>> real_array{};
  std::unordered_map<std::string, std::vector<std::string>> string_array{};
  std::unordered_map<std::string, std::map<std::string, std::variant<std::string, int, real>>> struct_array;
  std::unordered_map<std::string, Range<int>> int_range{};
  std::unordered_map<std::string, Range<real>> real_range{};

public:
  std::map<std::string, int> VNs;

  explicit Parameter(int *argc, char ***argv);

  explicit Parameter(const std::string &filename);

  Parameter(const Parameter &) = delete;

  Parameter(Parameter &&) = delete;

  Parameter operator=(Parameter &&) = delete;

  Parameter &operator=(const Parameter &) = delete;

  int &get_int(const std::string &name) { return int_parameters.at(name); }

  [[nodiscard]] const int &get_int(const std::string &name) const { return int_parameters.at(name); }

  [[nodiscard]] bool has_int(const std::string &name) const {
    return int_parameters.find(name) != int_parameters.end();
  }

  real &get_real(const std::string &name) { return real_parameters.at(name); }

  [[nodiscard]] const real &get_real(const std::string &name) const { return real_parameters.at(name); }

  bool &get_bool(const std::string &name) { return bool_parameters.at(name); }

  [[nodiscard]] const bool &get_bool(const std::string &name) const { return bool_parameters.at(name); }

  std::string &get_string(const std::string &name) { return string_parameters.at(name); }

  [[nodiscard]] const std::string &get_string(const std::string &name) const { return string_parameters.at(name); }

  [[nodiscard]] const auto &get_struct(const std::string &name) const { return struct_array.at(name); }

  [[nodiscard]] const auto &get_string_array(const std::string &name) const { return string_array.at(name); }

  [[nodiscard]] const auto &get_real_array(const std::string &name) const { return real_array.at(name); }

  [[nodiscard]] const auto &get_int_array(const std::string &name) const { return int_array.at(name); }

  [[nodiscard]] const auto &get_real_range(const std::string &name) const { return real_range.at(name); }

  [[nodiscard]] bool has_int_array(const std::string &name) const { return int_array.find(name) != int_array.end(); }

  void update_parameter(const std::string &name, const bool new_value) { bool_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const int new_value) { int_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const real new_value) { real_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const std::string new_value) { string_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const std::vector<int> &new_value) { int_array[name] = new_value; }

  void update_parameter(const std::string &name, const std::vector<real> &new_value) { real_array[name] = new_value; }

  void update_parameter(const std::string &name, const std::vector<std::string> &new_value) {
    string_array[name] = new_value;
  }

  /**
 * \brief Deduces simulation information based on the current parameters.
   * This function is called on driver initialization, when the species and reactions info has been read.
 */
  void deduce_sim_info(const Species &spec);

  std::vector<int> identify_variable_labels(std::vector<std::string> &var_name, const Species &species) const;

  ~Parameter() = default;

private:
  const std::array<std::string, 1> file_names{
    "./input/setup.txt"
  };

  void read_param_from_file();

  void deduce_known_info();

  void read_one_file(std::ifstream &file);

  template<typename T>
  int read_line_to_array(std::istringstream &line, std::vector<T> &arr);

  static std::map<std::string, std::variant<std::string, int, real>> read_struct(std::ifstream &file);

  // template<typename T>
  // static std::unordered_map<std::string, Range<T>> read_range(std::ifstream &file);

  void get_variable_names(const Species &spec);

  void setup_default_settings();

  void diagnose_parallel_info();
};

// name labels for my program
// PASSIVE SCALARS are labeled from 100
// Species names are not included, and they will be labeled from 1000
static const std::map<std::string, int> VNsDefault /*short for Variable Names*/{
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
  // TURBULENCE MODEL RELATED
  {"K", 6},
  {"TKE", 6},
  {"OMEGA", 7},
  // FLAMELET RELATED
  {"MIXTUREFRACTION", 8},
  {"MIXTUREFRACTIONVARIANCE", 9},
  // In the future, SA may be implemented, let us temporarily name the variable to be "nuTilde"
  {"NUTILDE", 10},

  // Some available variables
  {"MACH", 51},
  {"MUT", 52},
  {"FD", 53},
  {"SCALARDISSIPATIONRATE", 54},
  {"QCRITERIA", 55},
  {"SHOCKSENSOR", 56},
};
}

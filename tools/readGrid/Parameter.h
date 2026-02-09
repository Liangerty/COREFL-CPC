#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <array>
#include "Define.h"
#include "gxl_lib/Array.hpp"
#include <map>
#include <variant>

namespace cfd {

template<typename T>
struct Range {
  T xs, xe, ys, ye, zs, ze;
};

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

  void update_parameter(const std::string &name, const int new_value) { int_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const real new_value) { real_parameters[name] = new_value; }

  void update_parameter(const std::string &name, const std::vector<int> &new_value) { int_array[name] = new_value; }

  void update_parameter(const std::string &name, const std::vector<real> &new_value) { real_array[name] = new_value; }

  void update_parameter(const std::string &name,
                        const std::vector<std::string> &new_value) { string_array[name] = new_value; }

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

  template<typename T>
  static std::unordered_map<std::string, Range<T>> read_range(std::ifstream &file);

  void setup_default_settings();

  void diagnose_parallel_info();
};

}

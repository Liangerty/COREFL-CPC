#include "ChemData.h"
#include <fstream>
#include "gxl_lib/MyString.h"
#include "fmt/core.h"
#include "Parallel.h"
#include "Constants.h"
#include "Element.h"
#include <algorithm>
#include <cmath>

cfd::Species::Species(Parameter &parameter) {
  parameter.update_parameter("n_spec", 0);
  if (parameter.get_int("species")) {
    std::ifstream file("./input/" + parameter.get_string("mechanism_file"));
    std::string input{};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      if (input == "ELEMENTS" || input == "ELEM") {
        // As the ANSYS_Chemkin-Pro_Input_Manual told, the element part must start with "ELEMENTS" or "ELEM",
        // which are all capitalized.
        break;
      }
    }
    // Read elements
    int n_elem{0};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      gxl::to_upper(input);
      if (input == "END") continue; // If this line is "END", there must be a "SPECIES" or "SPEC" followed.
      if (input == "SPECIES" || input == "SPEC") break;
      elem_list.emplace(input, n_elem++);
    }

    // Species
    int num_spec{0};
    bool has_therm{false};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      gxl::to_upper(input);
      if (input == "END") continue; // If this line is "END", there must be a "REACTIONS" or "THERMO" followed.
      if (input == "REACTIONS" || input == "REAC") break;
      if (input == "THERMO") {
        // The thermodynamic info is in this mechanism file
        has_therm = true;
        break;
      }
      spec_list.emplace(input, num_spec);
      ++num_spec;
    }
    set_nspec(num_spec, n_elem);
    if (num_spec > MAX_SPEC_NUMBER) {
      fmt::print(
        "The number of species in this simulation is {}, larger than the allowed species number {}. Please modify the CMakeLists.txt to increase the MAX_SPEC_NUMBER.\n",
        num_spec, MAX_SPEC_NUMBER);
      MpiParallel::exit();
    }
    parameter.update_parameter("n_spec", num_spec);
    spec_name.resize(num_spec);
    for (auto &[name, label]: spec_list) {
      spec_name[label] = name;
    }
    if (parameter.get_int("reaction") != 2) {
      // Not flamelet model, update these variables. If flamelet, these variables will be updated later.
      parameter.update_parameter("n_var", parameter.get_int("n_var") + num_spec);
      parameter.update_parameter("n_scalar", parameter.get_int("n_scalar") + num_spec);
    }

    if (!has_therm) {
      file.close();
      file.open("./input/" + parameter.get_string("therm_file"));
    }
    const bool has_trans = read_therm(file, has_therm);

    if (!has_trans) {
      file.close();
      file.open("input/" + parameter.get_string("transport_file"));
    }
    read_tran(file);

    if (parameter.get_int("myid") == 0) {
      fmt::print("\n{:*^80}\n", "Species Information");
      fmt::print("\t->-> {:<20} : number of species\n\t", n_spec);
      int counter_spec{0};
      for (auto &[name, label]: spec_list) {
        fmt::print("{}\t", name);
        ++counter_spec;
        if (counter_spec % 10 == 0) {
          fmt::print("\n\t");
        }
      }
      fmt::print("\n");
    }
  }
}

void cfd::Species::compute_cp(real temp, real *cp) const & {
  const real t2{temp * temp}, t3{t2 * temp}, t4{t3 * temp};
  #ifdef HighTempMultiPart
  const auto &c = therm_poly_coeff;
  for (int i = 0; i < n_spec; ++i) {
    real tt{temp};
    if (tt < temperature_range(i, 0)) {
      tt = temperature_range(i, 0);
      const real tt2{tt * tt}, tt3{tt2 * tt}, tt4{tt3 * tt};
      cp[i] = c(0, 0, i) + c(1, 0, i) * tt + c(2, 0, i) * tt2 + c(3, 0, i) * tt3 + c(4, 0, i) * tt4;
    } else if (tt > temperature_range(i, n_temperature_range[i])) {
      tt = temperature_range(i, n_temperature_range[i]);
      auto j = n_temperature_range[i] - 1;
      const real tt2{tt * tt}, tt3{tt2 * tt}, tt4{tt3 * tt};
      cp[i] = c(0, j, i) + c(1, j, i) * tt + c(2, j, i) * tt2 + c(3, j, i) * tt3 + c(4, j, i) * tt4;
    } else {
      for (int j = 0; j < n_temperature_range[i]; ++j) {
        if (temperature_range(i, j) <= tt && tt <= temperature_range(i, j + 1)) {
          cp[i] = c(0, j, i) + c(1, j, i) * tt + c(2, j, i) * t2 + c(3, j, i) * t3 + c(4, j, i) * t4;
          break;
        }
      }
    }
    cp[i] *= R_u / mw[i];
  }
  #else // Combustion2Part
  for (int i = 0; i < n_spec; ++i) {
    real tt{temp};
    if (temp < t_low[i]) {
      tt = t_low[i];
      const real tt2{tt * tt}, tt3{tt2 * tt}, tt4{tt3 * tt};
      auto &coeff = low_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 +
              coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
    } else {
      auto &coeff = tt < t_mid[i] ? low_temp_coeff : high_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * t2 +
              coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    cp[i] *= R_u / mw[i];
  }
  #endif
}

void cfd::Species::compute_enthalpy(real t, real *h) const & {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  #ifdef HighTempMultiPart
  // undefined
  #else
  for (int i = 0; i < n_spec; ++i) {
    if (t < t_low[i]) {
      const real tt = t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = low_temp_coeff;
      h[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
             0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
      const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      h[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
    } else {
      auto &coeff = t < t_mid[i] ? low_temp_coeff : high_temp_coeff;
      h[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
             0.2 * coeff(i, 4) * t5 + coeff(i, 5);
    }
    h[i] *= R_u / mw[i];
  }
  #endif
}

void cfd::Species::compute_enthalpy_and_cp(real t, real *h, real *cp) const & {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  #ifdef HighTempMultiPart
  #else
  for (int i = 0; i < n_spec; ++i) {
    if (t < t_low[i]) {
      const double tt = t_low[i];
      const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = low_temp_coeff;
      h[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
             0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      h[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
    } else {
      auto &coeff = t < t_mid[i] ? low_temp_coeff : high_temp_coeff;
      h[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
             0.2 * coeff(i, 4) * t5 + coeff(i, 5);
      cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    h[i] *= R_u / mw[i];
    cp[i] *= R_u / mw[i];
  }
  #endif
}

void cfd::Species::set_nspec(int n_sp, int n_elem) {
  n_spec = n_sp;
  elem_comp.resize(n_sp, n_elem);
  mw.resize(n_sp, 0);
  #ifdef HighTempMultiPart
  n_temperature_range.resize(n_sp, 2);
  #else // Combustion2Part
  t_low.resize(n_sp, 300);
  t_mid.resize(n_sp, 1000);
  t_high.resize(n_sp, 5000);
  high_temp_coeff.resize(n_sp, 7);
  low_temp_coeff.resize(n_sp, 7);
  #endif
  geometry.resize(n_sp, 0);
  LJ_potent_inv.resize(n_sp, 0);
  vis_coeff.resize(n_sp, 0);
  WjDivWi_to_One4th.resize(n_sp, n_sp);
  sqrt_WiDivWjPl1Mul8.resize(n_sp, n_sp);
  binary_diffusivity_coeff.resize(n_sp, n_sp);
  kb_over_eps_jk.resize(n_sp, n_sp);
  ZRotF298.resize(n_sp, 0);
}

bool cfd::Species::read_therm(std::ifstream &therm_dat, bool read_from_comb_mech) {
  std::string input{};
  if (!read_from_comb_mech) {
    read_until(therm_dat, input, "THERMO", gxl::Case::upper); // "THERMO"
  }
  #ifdef HighTempMultiPart
  real T_low{300}, T_mid{1000}, T_high{5000};
  #endif // Combustion2Part
  while (std::getline(therm_dat, input)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    std::istringstream line(input);
    #ifdef Combustion2Part
    real T_low{300}, T_mid{1000}, T_high{5000};
    #endif
    line >> T_low >> T_mid >> T_high;
    #ifdef Combustion2Part
    t_low.resize(n_spec, T_low);
    t_mid.resize(n_spec, T_mid);
    t_high.resize(n_spec, T_high);
    #endif
    break;
  }

  std::string key{};
  int n_read{0};
  std::vector<int> have_read;
  std::istringstream line(input);
  bool has_trans{false};
  #ifdef HighTempMultiPart
  std::vector<std::vector<real>> temporary_range(n_spec, {T_low, T_mid, T_high});
  std::vector<int> n_temperature_spec(n_spec, 3);
  std::vector<gxl::MatrixDyn<real>> therm_poly_tempo(n_spec);
  int range_max = 2;
  #endif
  while (getline_to_stream(therm_dat, input, line, gxl::Case::upper)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    line >> key;
    // If the keyword is "END", just read the next line, if it's eof, then we won't come into this loop.
    // Else, a keyword "REACTIONS" or "TRANSPORT" may be encountered.
    if (key == "END") {
      if (n_read < n_spec) {
        fmt::print("The thermodynamic data aren't enough. We need {} species info but only {} are supplied.\n", n_spec,
                   n_read);
      }
      continue;
    }
    if (key == "REACTIONS" || input == "REAC") break;
    if (key == "TRANSPORT" || key == "TRAN") {
      has_trans = true;
      break;
    }
    if (n_read >= n_spec) break;

    // Let us read the species.
    key.assign(input, 0, 18);
    gxl::to_stringstream(key, line);
    line >> key;
    if (!spec_list.contains(key)) {
      #ifdef HighTempMultiPart
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      if (key != "TEMP") {
        gxl::getline(therm_dat, input);
        gxl::getline(therm_dat, input);
      } else {
        // The keyword is "TEMP", there will be more temperature ranges
        std::vector<real> temp_range;
        while (line >> key) {
          temp_range.push_back(std::stod(key));
        }
        auto range_number = temp_range.size() - 1;
        for (int i = 0; i < range_number; ++i) {
          gxl::getline(therm_dat, input);
          gxl::getline(therm_dat, input);
        }
      }
      #else // Combustion2Part
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      #endif
      continue;
    }
    const int curr_sp = spec_list.at(key);
    // If the species info has been read, then the second set of parameters are ignored.
    bool read{false};
    for (auto ss: have_read) {
      if (ss == curr_sp) {
        read = true;
        break;
      }
    }
    if (read) {
      #ifdef HighTempMultiPart
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      if (key != "TEMP") {
        gxl::getline(therm_dat, input);
        gxl::getline(therm_dat, input);
        gxl::getline(therm_dat, input);
      } else {
        // The keyword is "TEMP", there will be more temperature ranges
        std::vector<real> temp_range;
        while (line >> key) {
          temp_range.push_back(std::stod(key));
        }
        auto range_number = temp_range.size() - 1;
        for (int i = 0; i < range_number; ++i) {
          gxl::getline(therm_dat, input);
          gxl::getline(therm_dat, input);
        }
      }
      #else // Combustion2Part
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      #endif
      continue;
    }

    #ifdef HighTempMultiPart
    key.assign(input, 45, 10);  // T_low
    temporary_range[curr_sp][0] = std::stod(key);
    key.assign(input, 55, 10);  // T_high
    temporary_range[curr_sp][2] = std::stod(key);
    key.assign(input, 65, 10);  // Probably specify a different T_mid
    gxl::to_stringstream(key, line);
    line >> key;
    if (!key.empty()) {
      temporary_range[curr_sp][1] = std::stod(key);
    }
    #else // Combustion2Part
    key.assign(input, 45, 10); // T_low
    t_low[curr_sp] = std::stod(key);
    key.assign(input, 55, 10); // T_high
    t_high[curr_sp] = std::stod(key);
    key.assign(input, 65, 10); // Probably specify a different T_mid
    gxl::to_stringstream(key, line);
    line >> key;
    if (!key.empty()) t_mid[curr_sp] = std::stod(key);
    #endif

    // Read element composition
    std::string comp_str{};
    for (int i = 0; i < 4; ++i) {
      comp_str.assign(input, 24 + i * 5, 5);
      gxl::trim_left(comp_str);
      if (comp_str.empty() || comp_str.starts_with('0')) break;
      gxl::to_stringstream(comp_str, line);
      line >> key;
      if (!elem_list.contains(key)) continue;
      int stoi{0};
      line >> stoi;
      elem_comp(curr_sp, elem_list[key]) = stoi;
    }
    // Compute the relative molecular weight
    double mole_weight{0};
    for (const auto &[element, label]: elem_list) {
      mole_weight += Element{element}.get_atom_weight() * elem_comp(curr_sp, label);
    }
    mw[curr_sp] = mole_weight;

    // Read the thermodynamic fitting coefficients
    std::getline(therm_dat, input);
    #ifdef HighTempMultiPart
    if (input[0] == 'T') {
      // The keyword is "TEMP", there will be more temperature ranges
      gxl::to_stringstream(input, line);
      line >> key; // TEMP
      std::vector<real> temp_range;
      while (line >> key) {
        temp_range.push_back(std::stod(key));
      }
      n_temperature_spec[curr_sp] = (int) temp_range.size();
      n_temperature_range[curr_sp] = (int) temp_range.size() - 1;
      range_max = std::max(range_max, n_temperature_range[curr_sp]);
      temporary_range[curr_sp].resize(n_temperature_range[curr_sp] + 1);
      for (int i = 0; i < n_temperature_range[curr_sp] + 1; ++i) {
        temporary_range[curr_sp][i] = temp_range[i];
      }
      therm_poly_tempo[curr_sp].resize(n_temperature_range[curr_sp], 7);
      // Read polynomial coefficients for the n_range ranges
      std::string cs1{}, cs2{}, cs3{}, cs4{}, cs5{};
      for (int i = n_temperature_range[curr_sp] - 1; i >= 0; --i) {
        std::getline(therm_dat, input);
        cs1.assign(input, 0, 15);
        cs2.assign(input, 15, 15);
        cs3.assign(input, 30, 15);
        cs4.assign(input, 45, 15);
        cs5.assign(input, 60, 15);
        therm_poly_tempo[curr_sp](i, 0) = std::stod(cs1);
        therm_poly_tempo[curr_sp](i, 1) = std::stod(cs2);
        therm_poly_tempo[curr_sp](i, 2) = std::stod(cs3);
        therm_poly_tempo[curr_sp](i, 3) = std::stod(cs4);
        therm_poly_tempo[curr_sp](i, 4) = std::stod(cs5);
        // second line
        std::getline(therm_dat, input);
        cs1.assign(input, 0, 15);
        cs2.assign(input, 15, 15);
        therm_poly_tempo[curr_sp](i, 5) = std::stod(cs1);
        therm_poly_tempo[curr_sp](i, 6) = std::stod(cs2);
      }
    } else {
      // This is the usual case, when only 2 temperature ranges exist.
      therm_poly_tempo[curr_sp].resize(2, 7);
      std::string cs1{}, cs2{}, cs3{}, cs4{}, cs5{};
      double c1, c2, c3, c4, c5;
      cs1.assign(input, 0, 15);
      cs2.assign(input, 15, 15);
      cs3.assign(input, 30, 15);
      cs4.assign(input, 45, 15);
      cs5.assign(input, 60, 15);
      therm_poly_tempo[curr_sp](1, 0) = std::stod(cs1);
      therm_poly_tempo[curr_sp](1, 1) = std::stod(cs2);
      therm_poly_tempo[curr_sp](1, 2) = std::stod(cs3);
      therm_poly_tempo[curr_sp](1, 3) = std::stod(cs4);
      therm_poly_tempo[curr_sp](1, 4) = std::stod(cs5);
      // second line
      std::getline(therm_dat, input);
      cs1.assign(input, 0, 15);
      cs2.assign(input, 15, 15);
      cs3.assign(input, 30, 15);
      cs4.assign(input, 45, 15);
      cs5.assign(input, 60, 15);
      therm_poly_tempo[curr_sp](1, 5) = std::stod(cs1);
      therm_poly_tempo[curr_sp](1, 6) = std::stod(cs2);
      therm_poly_tempo[curr_sp](0, 0) = std::stod(cs3);
      therm_poly_tempo[curr_sp](0, 1) = std::stod(cs4);
      therm_poly_tempo[curr_sp](0, 2) = std::stod(cs5);
      // third line
      std::getline(therm_dat, input);
      cs1.assign(input, 0, 15);
      cs2.assign(input, 15, 15);
      cs3.assign(input, 30, 15);
      cs4.assign(input, 45, 15);
      therm_poly_tempo[curr_sp](0, 3) = std::stod(cs1);
      therm_poly_tempo[curr_sp](0, 4) = std::stod(cs2);
      therm_poly_tempo[curr_sp](0, 5) = std::stod(cs3);
      therm_poly_tempo[curr_sp](0, 6) = std::stod(cs4);
    }
    #else // Combustion2Part
    std::string cs1{}, cs2{}, cs3{}, cs4{}, cs5{};
    double c1, c2, c3, c4, c5;
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 0) = c1;
    high_temp_coeff(curr_sp, 1) = c2;
    high_temp_coeff(curr_sp, 2) = c3;
    high_temp_coeff(curr_sp, 3) = c4;
    high_temp_coeff(curr_sp, 4) = c5;
    // second line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 5) = c1;
    high_temp_coeff(curr_sp, 6) = c2;
    low_temp_coeff(curr_sp, 0) = c3;
    low_temp_coeff(curr_sp, 1) = c4;
    low_temp_coeff(curr_sp, 2) = c5;
    // third line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    low_temp_coeff(curr_sp, 3) = c1;
    low_temp_coeff(curr_sp, 4) = c2;
    low_temp_coeff(curr_sp, 5) = c3;
    low_temp_coeff(curr_sp, 6) = c4;
    #endif

    have_read.push_back(curr_sp);
    ++n_read;
  }

  #ifdef HighTempMultiPart
  // After reading all info, we need to resize the arrays of thermodynamic info
  temperature_range.resize(n_spec, range_max + 1);
  therm_poly_coeff.resize(7, range_max, n_spec, 0); // This is column major
  for (int l = 0; l < n_spec; ++l) {
    for (int i = 0; i < n_temperature_spec[l]; ++i) {
      temperature_range(l, i) = temporary_range[l][i];
    }
    for (int i = 0; i < n_temperature_spec[l] - 1; ++i) {
      for (int j = 0; j < 7; ++j) {
        therm_poly_coeff(j, i, l) = therm_poly_tempo[l](i, j);
      }
    }
  }
  #endif

  return has_trans;
}

void cfd::Species::read_tran(std::ifstream &tran_dat) {
  std::string input{}, key{};
  std::istringstream line(input);
  int n_read{0};
  std::vector<int> have_read;

  std::vector<double> eps_kb(n_spec, 0), sigma(n_spec, 0), mu(n_spec, 0), alpha(n_spec, 0);
  while (getline_to_stream(tran_dat, input, line, gxl::Case::upper)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    line >> key;
    if (key.starts_with("END") || key.starts_with("REACTIONS") || key.starts_with("REAC")) {
      if (n_read < n_spec) {
        fmt::print("The transport data aren't enough. We need {} species info but only {} are supplied.\n", n_spec,
                   n_read);
      }
      break;
    }
    if (!spec_list.contains(key)) {
      continue;
    }
    if (n_read >= n_spec) break;
    const int curr_sp = spec_list.at(key);
    // If the species info has been read, then the second set of parameters are ignored.
    bool read{false};
    for (auto ss: have_read) {
      if (ss == curr_sp) {
        read = true;
        break;
      }
    }
    if (read) {
      continue;
    }

    gxl::to_stringstream(input, line);
    line >> key >> geometry[curr_sp] >> eps_kb[curr_sp] >> sigma[curr_sp] >> mu[curr_sp] >> alpha[curr_sp]
        >> ZRotF298[curr_sp];
    LJ_potent_inv[curr_sp] = 1.0 / eps_kb[curr_sp];
    vis_coeff[curr_sp] =
        2.6693e-6 * sqrt(mw[curr_sp]) / (sigma[curr_sp] * sigma[curr_sp]);
    // compute ZRotF298*F(298)
    real TRedInv = eps_kb[curr_sp] / 298;
    real F298 = 1 + 0.5 * pi * pi * pi * sqrt(TRedInv) + (2 + 0.25 * pi * pi) * TRedInv +
                sqrt(pi * pi * pi) * sqrt(TRedInv * TRedInv * TRedInv);
    ZRotF298[curr_sp] *= F298;

    have_read.push_back(curr_sp);
    ++n_read;
  }

  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_spec; ++j) {
      WjDivWi_to_One4th(i, j) = std::pow(mw[j] / mw[i], 0.25);
      sqrt_WiDivWjPl1Mul8(i, j) = 1.0 / std::sqrt(8 * (1 + mw[i] / mw[j]));
    }
  }

  gxl::MatrixDyn<real> W_jk(n_spec, n_spec, 0), sigma_jk(n_spec, n_spec, 0), xi_jk(n_spec, n_spec, 1);
  gxl::MatrixDyn<real> eps_jk_kb(n_spec, n_spec, 0), mu_jk2(n_spec, n_spec, 0), delta_ij_star(n_spec, n_spec, 0);
  for (int j = 0; j < n_spec; ++j) {
    W_jk(j, j) = mw[j] * 0.5;
    mu_jk2(j, j) = mu[j] * mu[j];
    sigma_jk(j, j) = sigma[j];
    binary_diffusivity_coeff(j, j) = 0.0188324 / (std::sqrt(W_jk(j, j)) * sigma_jk(j, j) * sigma_jk(j, j));
    for (int k = j + 1; k < n_spec; ++k) {
      W_jk(j, k) = mw[j] * mw[k] / (mw[j] + mw[k]);
      W_jk(k, j) = W_jk(j, k);

      mu_jk2(j, k) = mu[j] * mu[k];
      mu_jk2(k, j) = mu_jk2(j, k);
      if (is_polar(mu[j]) != is_polar(mu[k])) {
        xi_jk(j, k) = compute_xi(j, k, mu.data(), sigma.data(), eps_kb.data(), alpha.data());
        xi_jk(k, j) = xi_jk(j, k);
        mu_jk2(j, k) = 0;
        mu_jk2(k, j) = 0;
      }
      sigma_jk(j, k) = 0.5 * (sigma[j] + sigma[k]) * std::pow(xi_jk(j, k), -1.0 / 6.0);

      binary_diffusivity_coeff(j, k) = 0.0188324 / (std::sqrt(W_jk(j, k)) * sigma_jk(j, k) * sigma_jk(j, k));
      binary_diffusivity_coeff(k, j) = binary_diffusivity_coeff(j, k);

      eps_jk_kb(j, k) = xi_jk(j, k) * xi_jk(j, k) * std::sqrt(eps_kb[j] * eps_kb[k]);
      kb_over_eps_jk(j, k) = 1.0 / eps_jk_kb(j, k);
      kb_over_eps_jk(k, j) = kb_over_eps_jk(j, k);

      delta_ij_star(j, k) = 0.5 * compute_reduced_dipole_moment(j, mu.data(), eps_kb.data(), sigma.data()) *
                            compute_reduced_dipole_moment(k, mu.data(), eps_kb.data(), sigma.data());
    }
  }
}

int cfd::Species::is_polar(real dipole_moment) {
  return dipole_moment == 0 ? 0 : 1;
}

real cfd::Species::compute_xi(int j, int k, real *dipole_moment, const real *sigma, const real *eps_kb,
  const real *alpha) {
  // labels for n(non-polar), p(polar)
  int n{0}, p{0};
  if (is_polar(dipole_moment[j])) {
    p = j;
    n = k;
  } else {
    p = k;
    n = j;
  }

  const real alpha_n_star{alpha[n] / std::pow(sigma[n], 3)};
  const real mu_p_star{compute_reduced_dipole_moment(p, dipole_moment, eps_kb, sigma)};
  const real xi = 1.0 + 0.25 * alpha_n_star * mu_p_star * mu_p_star * std::sqrt(eps_kb[p] / eps_kb[n]);
  return xi;
}

real cfd::Species::compute_reduced_dipole_moment(int i, real *dipole_moment, const real *eps_kb, const real *sigma) {
  if (!is_polar(dipole_moment[i])) {
    return 0;
  }
  // default dipole moment (mu) in Debye = 1e-18 cm^{3/2}ergs^{1/2} = 1e-18 cm^{3/2}(1e-7 J)^{1/2}
  // default collision diameter (sigma) in Angstrom = 1e-8 cm
  const real Debye_to_cm_Joule = 1e-18 * std::sqrt(1e-7);
  const real cm_to_Angstrom = std::pow(1e+8, 1.5);
  const real convert_units = Debye_to_cm_Joule * cm_to_Angstrom / std::sqrt(boltzmann_constants);
  return convert_units * dipole_moment[i] / std::sqrt(eps_kb[i] * sigma[i] * sigma[i] * sigma[i]);
}

cfd::Reaction::Reaction(Parameter &parameter, const Species &species) {
  parameter.update_parameter("n_reac", 0);
  if (!parameter.get_int("species")) {
    parameter.update_parameter("reaction", 0);
    return;
  }
  if (parameter.get_int("reaction") != 1) {
    return;
  }
  std::ifstream file("./input/" + parameter.get_string("mechanism_file"));
  std::string input{};
  const std::vector<std::string> reac_candidate{"REACTIONS", "REAC"};
  read_until(file, input, reac_candidate, gxl::Case::keep);
  std::istringstream line(input);
  std::string key{};
  line >> key >> key;
  if (!key.empty()) {
    // The units are not default units, we need some conversion here.
    // Currently, we just use the default ones, here is blanked.
  }

  int has_read{0};
  const int ns{species.n_spec};
  set_nreac(100, ns);
  std::vector<int> duplicate_reactions{};
  getline_to_stream(file, input, line, gxl::Case::upper);
  while (input != "END") {
    if (input[0] == '!' || input.empty()) {
      getline_to_stream(file, input, line, gxl::Case::upper);
      continue;
    }

    if (has_read >= n_reac) {
      set_nreac(n_reac + 100, ns);
    }

    // If not a comment, then a reaction is specified.
    read_reaction_line(input, has_read, species);
    //No matter if this reaction has some auxilliary information, just read and if not,
    //it will return as nothing happens.
    bool is_dup{false};
    input = get_auxi_info(file, has_read, species, is_dup);

    if (is_dup) {
      bool found = false;
      int dup{}, pos{};
      for (auto r = 0; r < duplicate_reactions.size(); ++r) {
        bool all_same{true};
        const int idx_dup{duplicate_reactions[r]};
        for (int i = 0; i < ns; ++i) {
          if (stoi_f(idx_dup, i) != stoi_f(has_read, i) || stoi_b(idx_dup, i) != stoi_b(has_read, i)) {
            all_same = false;
            break;
          }
        }
        if (all_same) {
          found = true;
          dup = idx_dup;
          pos = r;
          break;
        }
      }
      if (found) {
        A2[dup] = A[has_read];
        b2[dup] = b[has_read];
        Ea2[dup] = Ea[has_read];
        duplicate_reactions.erase(duplicate_reactions.cbegin() + pos);
        for (int i = 0; i < ns; ++i) {
          stoi_f(has_read, i) = 0;
          stoi_b(has_read, i) = 0;
        }
        label[has_read] = 1;
        order[has_read] = 0;
      } else {
        duplicate_reactions.push_back(has_read);
        ++has_read;
      }
      continue;
    }

    ++has_read;
  }
  set_nreac(has_read, ns);
  if (parameter.get_int("myid") == 0) {
    fmt::print("\t->-> {:<20} : number of reactions\n", n_reac);
    std::string method{"explicit"};
    if (auto mm = parameter.get_int("chemSrcMethod"); mm == 1) {
      method = "Exact Point Implicit";
    } else if (mm == 2) {
      method = "Diagonal Approximation";
    }
    fmt::print("\t\t->-> {:<25} : chemical source treatment\n", method);
  }
  parameter.update_parameter("n_reac", n_reac);
}

void cfd::Reaction::set_nreac(int nr, int ns) {
  n_reac = nr;
  label.resize(nr, 1); // By default, reactions are reversible ones.
  rev_type.resize(nr); // By default, reactions are elementary ones; 1 is used for REV type.
  stoi_f.resize(nr, ns, 0);
  stoi_b.resize(nr, ns, 0);
  order.resize(nr, 0);
  A.resize(nr, 0);
  b.resize(nr, 0);
  Ea.resize(nr, 0);
  A2.resize(nr, 0);
  b2.resize(nr, 0);
  Ea2.resize(nr, 0);
  third_body_coeff.resize(nr, ns, 1);
  troe_alpha.resize(nr, 0);
  troe_t3.resize(nr, 0);
  troe_t1.resize(nr, 0);
  troe_t2.resize(nr, 0);
}

void cfd::Reaction::read_reaction_line(std::string input, int idx, const Species &species) {
  // First determine if the reaction is pressure-dependent or needs catalyst
  // clean the equation for further use.
  // std::erase(input, ' ');
  const std::string leftparen = "(+", catalyst = "+M";
  auto iter_leftparen = input.find(leftparen);
  auto iter_catalyst = input.find(catalyst);
  auto iter_rightparen = input.find(')', iter_leftparen);
  if (iter_catalyst != std::string::npos) { //If the reaction requires catalyst
    label[idx] = 4;
    if (iter_leftparen != std::string::npos) { //It is also a pressure-dependent reaction
      label[idx] = 5;
      input.erase(input.cbegin() + iter_leftparen,
                  input.cbegin() + iter_rightparen + 1); //Erase the catalyst because we don't need it.
      iter_leftparen = input.find(leftparen);
      iter_rightparen = input.find_last_of(')');
      input.erase(input.cbegin() + iter_leftparen, input.cbegin() + iter_rightparen + 1);
    } else {
      input.erase(iter_catalyst, 2);
      iter_catalyst = input.find(catalyst);
      input.erase(iter_catalyst, 2);
    }
  }

  auto it_small = input.find('<'), it_eq = input.find('='); // , it_big = input.find(">");
  if (it_small == std::string::npos) {
    it_small = it_eq;
  }

  std::replace(input.begin(), input.begin() + it_small + 1, ' ', '<');
  std::string reacEq; //Reaction equation
  std::istringstream line(input);
  line >> reacEq >> A[idx] >> b[idx] >> Ea[idx];
  Ea[idx] /= R_c;

  //Split reaction equation into reactant string and product string.
  it_small = reacEq.find('<');
  it_eq = reacEq.find('=');
  auto it_big = reacEq.find('>');
  if (it_small == std::string::npos) {
    if (it_big != std::string::npos) {
      // => is here, the reaction is irreversible
      label[idx] = 0;
    } else {
      it_big = it_eq;
    }
    it_small = it_eq;
  } else if (it_big == std::string::npos) {
    // There exists < but no >, the < may be added by the replace operation above.
    // The original is just a single =, which means the reaction is reversible.
    it_big = it_eq;
  }
  std::string reactantString, productString;
  reactantString.assign(reacEq, 0, it_small);
  productString.assign(reacEq, it_big + 1, reacEq.size());

  //Find the reactants
  const auto &list = species.spec_list;
  for (auto it_plus = reactantString.find('+'); it_plus != std::string::npos; it_plus = reactantString.find(
                                                                                '+')) {
    //First get reactants except the last one into the map.
    if (reactantString[it_plus + 1] == '+') {
      // The first '+' is in the reactant name
      ++it_plus;
    }

    int stoi = 1;
    std::string reactant1;
    if (isdigit(reactantString[0])) {
      stoi = std::stoi(reactantString.substr(0, 1));
      reactantString.erase(0, 1);
      reactant1.assign(reactantString, 0, it_plus - 1);
    } else {
      reactant1.assign(reactantString, 0, it_plus);
    }
    stoi_f(idx, list.at(reactant1)) += stoi;
    order[idx] -= stoi;
    reactantString.erase(0, reactant1.size() + 1);
  }
  //Next, put the last reactant into the map.
  int stoi = 1;
  if (isdigit(reactantString[0])) {
    stoi = std::stoi(reactantString.substr(0, 1));
    reactantString.erase(0, 1);
  }
  stoi_f(idx, list.at(reactantString)) += stoi;
  order[idx] -= stoi;

  //Find the products
  for (auto it_plus = productString.find('+'); it_plus != std::string::npos; it_plus = productString.find(
                                                                               '+')) {
    //First get products except the last one into the map.
    stoi = 1;
    std::string product1;
    if (isdigit(productString[0])) {
      stoi = std::stoi(productString.substr(0, 1));
      productString.erase(0, 1);
      product1.assign(productString, 0, it_plus - 1);
    } else {
      product1.assign(productString, 0, it_plus);
    }
    stoi_b(idx, list.at(product1)) += stoi;
    order[idx] += stoi;
    productString.erase(0, product1.size() + 1);
  }
  //Next, put the last product into the map.
  stoi = 1;
  if (isdigit(productString[0])) {
    stoi = std::stoi(productString.substr(0, 1));
    productString.erase(0, 1);
  }
  stoi_b(idx, list.at(productString)) += stoi;
  order[idx] += stoi;
}

std::string cfd::Reaction::get_auxi_info(std::ifstream &file, int idx, const Species &species, bool &is_dup) {
  std::string input, key;
  while (gxl::getline(file, input, gxl::Case::upper)) { //&&input.find("=")==std::string::npos
    std::replace(input.begin(), input.end(), '/', ' ');
    std::istringstream line(input);
    line >> key;
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    if (key == "END") {
      break;
    }
    if (input.find('=') != std::string::npos) {
      break;
    }
    if (key == "PLOG") { //Must be modified later to support this kind of reaction
      continue;
    }
    if (key == "DUPLICATE" || key == "DUP") {
      //If the keyword is duplicate, return and turn to duplicate reaction reader.
      is_dup = true;
      label[idx] = 3;
      continue;
    }
    if (key == "REV") {
      line >> A2[idx] >> b2[idx] >> Ea2[idx];
      Ea2[idx] /= R_c;
      rev_type[idx] = 1;
    } else if (key == "LOW") { //Keyword LOW requires 3 new Arrhenius coefficients
      line >> A2[idx] >> b2[idx] >> Ea2[idx];
      Ea2[idx] /= R_c;
      continue;
    } else if (key == "TROE") { //Troe form needs the following coefficients
      label[idx] = 6;
      line >> troe_alpha[idx] >> troe_t3[idx] >> troe_t1[idx];
      std::string what{};
      line >> what;
      if (!what.empty()) {
        troe_t2[idx] = std::stod(what);
        label[idx] = 7;
      }
    } else { //None of the keywords matched, this line should be third body coefficients.
      line.clear();
      line.str(input);

      while (line >> key) {
        real coe{};
        line >> coe;
        third_body_coeff(idx, species.spec_list.at(key)) = coe;
      }
    }
  }
  return input;
}

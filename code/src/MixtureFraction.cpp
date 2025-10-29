#include "MixtureFraction.h"
#include "fmt/format.h"
#include "Element.h"
#include "gxl_lib/Math.hpp"
#include <fstream>

namespace cfd {
MixtureFraction::MixtureFraction(Inflow &fuel, Inflow &oxidizer, const Species &chem_data) :
    n_spec{chem_data.n_spec}, elem_comp{chem_data.elem_comp}, mw{chem_data.mw} {}

BilgerH::BilgerH(Inflow &fuel, Inflow &oxidizer, const Species &spec, int myid) :
    MixtureFraction(fuel, oxidizer, spec) {
  auto &elem_list = spec.elem_list;
  const auto n_elem = static_cast<int>(elem_list.size());
  std::vector<real> nu_fuel(n_elem, 0), xk_fuel(n_spec, 0), xk_oxidizer(n_spec, 0);
  // First, compute the fuel side H-O ratio
  for (int i = 0; i < n_spec; ++i) {
    xk_fuel[i] = fuel.mw * fuel.sv[i] / spec.mw[i];
    xk_oxidizer[i] = oxidizer.mw * oxidizer.sv[i] / spec.mw[i];
    for (int j = 0; j < n_elem; ++j) {
      nu_fuel[j] += elem_comp(i, j) * xk_fuel[i];
    }
  }
  std::vector<real> xk_o2_unity_oxi = xk_oxidizer;
  const real x_o2_in_oxi = xk_oxidizer[spec.spec_list.at("O2")];
  for (int i = 0; i < n_spec; ++i) {
    xk_o2_unity_oxi[i] /= x_o2_in_oxi;
  }
  std::vector<real> nu_oxid(n_elem, 0);
  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_elem; ++j) {
      nu_oxid[j] += elem_comp(i, j) * xk_o2_unity_oxi[i];
    }
  }

  real o2_needed = -0.5 * nu_fuel[elem_list.at("O")];
  o2_needed += 0.25 * nu_fuel[elem_list.at("H")];
  real nu_h = nu_fuel[elem_list.at("H")] + o2_needed * nu_oxid[elem_list.at("H")];
  real nu_o = nu_fuel[elem_list.at("O")] + o2_needed * nu_oxid[elem_list.at("O")];

  // Output the global reaction string to screen
  std::string spec_eqn;
  for (auto &[name, i]: spec.spec_list) {
    if (xk_fuel[i] > 1e-10) {
      spec_eqn += fmt::format("{:.4f} {} + ", xk_fuel[i], name);
    }
  }
  spec_eqn += fmt::format("{:.4f} ( O2 ", o2_needed);
  for (auto &[name, i]: spec.spec_list) {
    if (name == "O2") {
      continue;
    }
    if (xk_oxidizer[i] > 1e-10) {
      spec_eqn += fmt::format("+ {:.4f} {} ", xk_o2_unity_oxi[i], name);
    }
  }
  spec_eqn += ") = Products\n";
  std::string elem_eqn;
  elem_eqn += fmt::format(" {:.4f} H + {:.4f} O -> P\n", nu_h, nu_o);

  nuh_mwh = nu_h * 1.008;
  half_nuo_mwo = nu_o * 15.999 * 0.5;
  std::string coupling_func{"\\beta = "};
  coupling_func += fmt::format("ZH/{:.4f} - ZO/{:.4f}\n", nuh_mwh, half_nuo_mwo);
  if (myid == 0) {
    fmt::print("{}{}{}", spec_eqn, elem_eqn, coupling_func);
    std::ofstream out("output/message/mixture_fraction_info.txt", std::ios::trunc);
    out << fmt::format("{}{}{}", spec_eqn, elem_eqn, coupling_func);
    out.close();
  }

  elem_label[0] = elem_list.at("H");
  elem_label[1] = elem_list.at("O");
  // compute beta_f and beta_o
  // fuel
  real z_h{0}, z_o{0};
  for (int i = 0; i < n_spec; ++i) {
    if (xk_fuel[i] > 1e-10) {
      const real x_w = xk_fuel[i] / fuel.mw;
      z_h += elem_comp(i, elem_label[0]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[1]) * 15.999 * x_w;
    }
  }
  beta_f = compute_coupling_function(z_h, z_o);

  // oxidizer
  z_h = 0;
  z_o = 0;
  for (int i = 0; i < n_spec; ++i) {
    if (xk_oxidizer[i] > 1e-10) {
      const real x_w = xk_oxidizer[i] / oxidizer.mw;
      z_h += elem_comp(i, elem_label[0]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[1]) * 15.999 * x_w;
    }
  }
  beta_o = compute_coupling_function(z_h, z_o);
  beta_diff = beta_f - beta_o;

  // compute the stoichiometric mixture fraction
  real tot_m{0};
  std::vector<real> z_elem(n_elem, 0);
  for (auto &[name, i]: spec.elem_list) {
    z_elem[i] = (nu_fuel[i] + o2_needed * nu_oxid[i]) * Element(name).get_atom_weight();
    tot_m += z_elem[i];
  }
  z_h = z_elem[elem_label[0]] / tot_m;
  z_o = z_elem[elem_label[1]] / tot_m;
  real beta_st = compute_coupling_function(z_h, z_o);
  z_st = (beta_st - beta_o) / beta_diff;

  const real erfc_two_zst = gxl::erfcInv(2 * z_st);
  f_zst = std::exp(-2 * erfc_two_zst * erfc_two_zst);
}

real BilgerH::compute_mixture_fraction(std::vector<real> &yk) {
  real z_h{0}, z_o{0};
  for (int i = 0; i < n_spec; ++i) {
    if (yk[i] > 1e-20) {
      const real y_w = yk[i] / mw[i];
      z_h += elem_comp(i, elem_label[0]) * 1.008 * y_w;
      z_o += elem_comp(i, elem_label[1]) * 15.999 * y_w;
    }
  }
  const real beta = compute_coupling_function(z_h, z_o);
  return (beta - beta_o) / beta_diff;
}

real BilgerH::compute_coupling_function(real z_h, real z_o) const {
  return z_h / nuh_mwh - z_o / half_nuo_mwo;
}

real BilgerCH::compute_coupling_function(real z_c, real z_h, real z_o) const {
  return z_c / nuc_mwc + z_h / nuh_mwh - z_o / half_nuo_mwo;
}

real BilgerCH::compute_mixture_fraction(std::vector<real> &yk) {
  real z_c{0}, z_h{0}, z_o{0};
  for (int i = 0; i < n_spec; ++i) {
    if (yk[i] > 1e-20) {
      const real y_w = yk[i] / mw[i];
      z_c += elem_comp(i, elem_label[0]) * 12.011 * y_w;
      z_h += elem_comp(i, elem_label[1]) * 1.008 * y_w;
      z_o += elem_comp(i, elem_label[2]) * 15.999 * y_w;
    }
  }
  const real beta = compute_coupling_function(z_c, z_h, z_o);
  return (beta - beta_o) / beta_diff;
}

BilgerCH::BilgerCH(Inflow &fuel, Inflow &oxidizer, const Species &spec, int myid) :
    MixtureFraction(fuel, oxidizer, spec) {
  auto &elem_list = spec.elem_list;
  const auto n_elem = static_cast<int>(elem_list.size());
  std::vector<real> nu_fuel(n_elem, 0), xk_fuel(n_spec, 0), xk_oxidizer(n_spec, 0);
  // First, compute the fuel side C-H-O ratio
  for (int i = 0; i < n_spec; ++i) {
    xk_fuel[i] = fuel.mw * fuel.sv[i] / spec.mw[i];
    xk_oxidizer[i] = oxidizer.mw * oxidizer.sv[i] / spec.mw[i];
    for (int j = 0; j < n_elem; ++j) {
      nu_fuel[j] += elem_comp(i, j) * xk_fuel[i];
    }
  }
  std::vector<real> xk_o2_unity_oxi = xk_oxidizer;
  const real x_o2_in_oxi = xk_oxidizer[spec.spec_list.at("O2")];
  for (int i = 0; i < n_spec; ++i) {
    xk_o2_unity_oxi[i] /= x_o2_in_oxi;
  }
  std::vector<real> nu_oxid(n_elem, 0);
  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_elem; ++j) {
      nu_oxid[j] += elem_comp(i, j) * xk_o2_unity_oxi[i];
    }
  }

  real o2_needed = -0.5 * nu_fuel[elem_list.at("O")];
  o2_needed += nu_fuel[elem_list.at("C")];
  o2_needed += 0.25 * nu_fuel[elem_list.at("H")];
  real nu_c{0}, nu_h{0};
  nu_c = nu_fuel[elem_list.at("C")] + o2_needed * nu_oxid[elem_list.at("C")];
  nu_h = nu_fuel[elem_list.at("H")] + o2_needed * nu_oxid[elem_list.at("H")];
  real nu_o = nu_fuel[elem_list.at("O")] + o2_needed * nu_oxid[elem_list.at("O")];

  // Output the global reaction string to screen
  std::string spec_eqn;
  for (auto &[name, i]: spec.spec_list) {
    if (xk_fuel[i] > 1e-10) {
      spec_eqn += fmt::format("{:.4f} {} + ", xk_fuel[i], name);
    }
  }
  spec_eqn += fmt::format("{:.4f} ( O2 ", o2_needed);
  for (auto &[name, i]: spec.spec_list) {
    if (name == "O2") {
      continue;
    }
    if (xk_oxidizer[i] > 1e-10) {
      spec_eqn += fmt::format("+ {:.4f} {} ", xk_o2_unity_oxi[i], name);
    }
  }
  spec_eqn += ") = Products\n";
  std::string elem_eqn;
  elem_eqn += fmt::format("{:.4f} C + ", nu_c);
  elem_eqn += fmt::format(" {:.4f} H + {:.4f} O -> P\n", nu_h, nu_o);

  nuc_mwc = nu_c * 12.011;
  nuh_mwh = nu_h * 1.008;
  half_nuo_mwo = nu_o * 15.999 * 0.5;
  std::string coupling_func{"\\beta = "};
  coupling_func += fmt::format("ZC/{:.4f} + ", nuc_mwc);
  coupling_func += fmt::format("ZH/{:.4f} - ZO/{:.4f}\n", nuh_mwh, half_nuo_mwo);
  if (myid == 0) {
    fmt::print("{}{}{}", spec_eqn, elem_eqn, coupling_func);
    std::ofstream out("output/message/mixture_fraction_info.txt", std::ios::trunc);
    out << fmt::format("{}{}{}", spec_eqn, elem_eqn, coupling_func);
    out.close();
  }

  elem_label[0] = elem_list.at("C");
  elem_label[1] = elem_list.at("H");
  elem_label[2] = elem_list.at("O");
  // compute beta_f and beta_o
  // fuel
  real z_c{0}, z_h{0}, z_o{0};
  for (int i = 0; i < n_spec; ++i) {
    if (xk_fuel[i] > 1e-10) {
      const real x_w = xk_fuel[i] / fuel.mw;
      z_c += elem_comp(i, elem_label[0]) * 12.011 * x_w;
      z_h += elem_comp(i, elem_label[1]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[2]) * 15.999 * x_w;
    }
  }
  beta_f = compute_coupling_function(z_c, z_h, z_o);

  // oxidizer
  z_c = 0;
  z_h = 0;
  z_o = 0;
  for (int i = 0; i < n_spec; ++i) {
    if (xk_oxidizer[i] > 1e-10) {
      const real x_w = xk_oxidizer[i] / oxidizer.mw;
      z_c += elem_comp(i, elem_label[0]) * 12.011 * x_w;
      z_h += elem_comp(i, elem_label[1]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[2]) * 15.999 * x_w;
    }
  }
  beta_o = compute_coupling_function(z_c, z_h, z_o);
  beta_diff = beta_f - beta_o;

  // compute the stoichiometric mixture fraction
  real tot_m{0};
  std::vector<real> z_elem(n_elem, 0);
  for (auto &[name, i]: spec.elem_list) {
    z_elem[i] = (nu_fuel[i] + o2_needed * nu_oxid[i]) * Element(name).get_atom_weight();
    tot_m += z_elem[i];
  }
  z_c = z_elem[elem_label[0]] / tot_m;
  z_h = z_elem[elem_label[1]] / tot_m;
  z_o = z_elem[elem_label[2]] / tot_m;
  real beta_st = compute_coupling_function(z_c, z_h, z_o);
  z_st = (beta_st - beta_o) / beta_diff;

  const real erfc_two_zst = gxl::erfcInv(2 * z_st);
  f_zst = std::exp(-2 * erfc_two_zst * erfc_two_zst);
}

void
acquire_mixture_fraction_expression(const Species &spec, const real *fuel, const real *oxidizer, Parameter &parameter) {
  const int n_spec = spec.n_spec;

  // compute the mw of fuel and oxidizer
  real mw_fuel{0}, mw_oxidizer{0};
  for (int i = 0; i < n_spec; ++i) {
    mw_fuel += fuel[i] / spec.mw[i];
    mw_oxidizer += oxidizer[i] / spec.mw[i];
  }
  mw_fuel = 1 / mw_fuel;
  mw_oxidizer = 1 / mw_oxidizer;

  auto &elem_list = spec.elem_list;
  const auto n_elem = static_cast<int>(elem_list.size());
  std::vector<real> nu_fuel(n_elem, 0), xk_fuel(n_spec, 0), xk_oxidizer(n_spec, 0);
  // First, compute the fuel side C-H-O ratio
  const auto &elem_comp = spec.elem_comp;
  for (int i = 0; i < n_spec; ++i) {
    xk_fuel[i] = mw_fuel * fuel[i] / spec.mw[i];
    xk_oxidizer[i] = mw_oxidizer * oxidizer[i] / spec.mw[i];
    for (int j = 0; j < n_elem; ++j) {
      nu_fuel[j] += elem_comp(i, j) * xk_fuel[i];
    }
  }
  std::vector<real> xk_o2_unity_oxi = xk_oxidizer;
  const real x_o2_in_oxi = xk_oxidizer[spec.spec_list.at("O2")];
  for (int i = 0; i < n_spec; ++i) {
    xk_o2_unity_oxi[i] /= x_o2_in_oxi;
  }
  std::vector<real> nu_oxid(n_elem, 0);
  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_elem; ++j) {
      nu_oxid[j] += elem_comp(i, j) * xk_o2_unity_oxi[i];
    }
  }

  bool has_c{false};
  if (elem_list.find("C") != elem_list.end()) {
    has_c = true;
  }

  real o2_needed = -0.5 * nu_fuel[elem_list.at("O")];
  if (has_c)
    o2_needed += nu_fuel[elem_list.at("C")];
  o2_needed += 0.25 * nu_fuel[elem_list.at("H")];
  real nu_c{0}, nu_h{0};
  if (has_c)
    nu_c = nu_fuel[elem_list.at("C")] + o2_needed * nu_oxid[elem_list.at("C")];
  nu_h = nu_fuel[elem_list.at("H")] + o2_needed * nu_oxid[elem_list.at("H")];
  real nu_o = nu_fuel[elem_list.at("O")] + o2_needed * nu_oxid[elem_list.at("O")];

  // Output the global reaction string to screen
  std::string spec_eqn;
  for (auto &[name, i]: spec.spec_list) {
    if (xk_fuel[i] > 1e-10) {
      spec_eqn += fmt::format("{:.4f} {} + ", xk_fuel[i], name);
    }
  }
  spec_eqn += fmt::format("{:.4f} ( O2 ", o2_needed);
  for (auto &[name, i]: spec.spec_list) {
    if (name == "O2") {
      continue;
    }
    if (xk_oxidizer[i] > 1e-10) {
      spec_eqn += fmt::format("+ {:.4f} {} ", xk_o2_unity_oxi[i], name);
    }
  }
  spec_eqn += ") = Products\n";
  std::string elem_eqn;
  if (has_c)
    elem_eqn += fmt::format("{:.4f} C + ", nu_c);
  elem_eqn += fmt::format("{:.4f} H + {:.4f} O -> P\n", nu_h, nu_o);

  real nuc_mwc_inv = nu_c * 12.011;
  real nuh_mwh_inv = nu_h * 1.008;
  real half_nuo_mwo_inv = nu_o * 15.999 * 0.5;
  std::string coupling_func{"\\beta = "};
  coupling_func += fmt::format("ZC/{:.4f} + ", nuc_mwc_inv);
  coupling_func += fmt::format("ZH/{:.4f} - ZO/{:.4f}\n", nuh_mwh_inv, half_nuo_mwo_inv);
  if (!has_c)
    nuc_mwc_inv = 0;
  else
    nuc_mwc_inv = 1 / nuc_mwc_inv;
  nuh_mwh_inv = 1 / nuh_mwh_inv;
  half_nuo_mwo_inv = 1 / half_nuo_mwo_inv;

  const int myid = parameter.get_int("myid");
  bool output_screen{false};
  if (myid == 0) {
    if (!parameter.has_int("mixture_fraction_expression_outputted")) {
      fmt::print("\t{}\t{}\t{}", spec_eqn, elem_eqn, coupling_func);
      std::ofstream out("output/message/mixture_fraction_info.txt", std::ios::trunc);
      out << fmt::format("{}{}{}", spec_eqn, elem_eqn, coupling_func);
      out.close();
      parameter.update_parameter("mixture_fraction_expression_outputted", 1);
      output_screen = true;
    }
  }

  int elem_label[3]{0, 1, 2};
  if (has_c)
    elem_label[0] = elem_list.at("C");
  elem_label[1] = elem_list.at("H");
  elem_label[2] = elem_list.at("O");

  // compute beta_f and beta_o
  // fuel
  real z_c{0}, z_h{0}, z_o{0};
  for (int i = 0; i < n_spec; ++i) {
    if (xk_fuel[i] > 1e-10) {
      const real x_w = xk_fuel[i] / mw_fuel;
      if (has_c)
        z_c += elem_comp(i, elem_label[0]) * 12.011 * x_w;
      z_h += elem_comp(i, elem_label[1]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[2]) * 15.999 * x_w;
    }
  }
  real beta_f = z_c * nuc_mwc_inv + z_h * nuh_mwh_inv - z_o * half_nuo_mwo_inv;

  // oxidizer
  z_c = 0;
  z_h = 0;
  z_o = 0;
  for (int i = 0; i < n_spec; ++i) {
    if (xk_oxidizer[i] > 1e-10) {
      const real x_w = xk_oxidizer[i] / mw_oxidizer;
      if (has_c)
        z_c += elem_comp(i, elem_label[0]) * 12.011 * x_w;
      z_h += elem_comp(i, elem_label[1]) * 1.008 * x_w;
      z_o += elem_comp(i, elem_label[2]) * 15.999 * x_w;
    }
  }
  real beta_o = z_c * nuc_mwc_inv + z_h * nuh_mwh_inv - z_o * half_nuo_mwo_inv;
  real beta_diff = beta_f - beta_o;

  // compute the stoichiometric mixture fraction
  real tot_m{0};
  std::vector<real> z_elem(n_elem, 0);
  for (auto &[name, i]: spec.elem_list) {
    z_elem[i] = (nu_fuel[i] + o2_needed * nu_oxid[i]) * Element(name).get_atom_weight();
    tot_m += z_elem[i];
  }
  if (has_c)
    z_c = z_elem[elem_label[0]] / tot_m;
  z_h = z_elem[elem_label[1]] / tot_m;
  z_o = z_elem[elem_label[2]] / tot_m;
  real beta_st = z_c * nuc_mwc_inv + z_h * nuh_mwh_inv - z_o * half_nuo_mwo_inv;
  real z_st = (beta_st - beta_o) / beta_diff;
  if (myid == 0 && output_screen) {
    fmt::print("\tz_st={}\n", z_st);
    std::ofstream out("output/message/mixture_fraction_info.txt", std::ios::app);
    out << fmt::format("z_st={}\n", z_st);
    out.close();
  }

  parameter.update_parameter("beta_diff_inv", 1.0 / beta_diff);
  parameter.update_parameter("beta_o", beta_o);
  parameter.update_parameter("nuc_mwc_inv", nuc_mwc_inv);
  parameter.update_parameter("nuh_mwh_inv", nuh_mwh_inv);
  parameter.update_parameter("half_nuo_mwo_inv", half_nuo_mwo_inv);
}
} // cfd
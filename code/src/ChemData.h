#pragma once

#include "Define.h"
#include "Parameter.h"
#include "gxl_lib/Matrix.hpp"

namespace cfd {
struct Species {
  explicit Species(Parameter &parameter);

  int n_spec{0};                        // number of species
  std::map<std::string, int> spec_list; // species list
  std::vector<std::string> spec_name;   // species name

  void compute_cp(real temp, real *cp) const &;

  void compute_enthalpy(real t, real *h) const &;

  void compute_enthalpy_and_cp(real t, real *h, real *cp) const &;

  // The properties of the species. Some previously private derived variables will appear in the corresponding function classes.
  std::map<std::string, int> elem_list; // element list
  gxl::MatrixDyn<int> elem_comp;        // the element composition of the species
  std::vector<real> mw;                 // the array of molecular weights
  // Thermodynamic properties
  #ifdef HighTempMultiPart
  std::vector<int> n_temperature_range; // the number of temperature ranges
  gxl::MatrixDyn<real> temperature_range; // the temperature range of the thermodynamic sections
  gxl::Array3D<real> therm_poly_coeff; // the polynomial coefficients of the thermodynamic sections
  #else // Combustion2Part
  std::vector<real> t_low, t_mid, t_high;               // the array of thermodynamic sections
  gxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff; // the cp/h/s polynomial coefficients
  #endif
  // Transport properties
  std::vector<real> geometry; // if the species is monatomic(0), linear(1), or nonlinear(2)
  std::vector<real> LJ_potent_inv; // the inverse of the Lennard-Jones potential
  std::vector<real> vis_coeff; // the coefficient to compute viscosity
  gxl::MatrixDyn<real> WjDivWi_to_One4th, sqrt_WiDivWjPl1Mul8; // Some constant value to compute partition functions
  gxl::MatrixDyn<real> binary_diffusivity_coeff;
  gxl::MatrixDyn<real> kb_over_eps_jk; // Used to compute reduced temperature for diffusion coefficients
  std::vector<real> ZRotF298;          // the rotational relaxation collision number at 298 K.

private:
  void set_nspec(int n_sp, int n_elem);

  bool read_therm(std::ifstream &therm_dat, bool read_from_comb_mech);

  void read_tran(std::ifstream &tran_dat);

  static int is_polar(real dipole_moment);

  real compute_xi(int j, int k, real *dipole_moment, const real *sigma, const real *eps_kb, const real *alpha);

  real compute_reduced_dipole_moment(int i, real *dipole_moment, const real *eps_kb, const real *sigma);
};

struct Reaction {
  explicit Reaction(Parameter &parameter, const Species &species);

private:
  void set_nreac(int nr, int ns);

  void read_reaction_line(std::string input, int idx, const Species &species);

  std::string get_auxi_info(std::ifstream &file, int idx, const cfd::Species &species, bool &is_dup);

public:
  int n_reac{0};
  // The label represents which method to compute kf and kb.
  // 0 - Irreversible, 1 - Reversible
  // 2 - REV (reversible with both kf and kb Arrhenius coefficients given)
  // 3 - DUP (Multiple sets of kf Arrhenius coefficients given)
  // 4 - Third body reactions ( +M is added on both sides, indicating the reaction needs catylists)
  // 5 - Lindemann Type (Pressure dependent reactions computed with Lindemann type method)
  // 6 - Troe-3 (Pressure dependent reactions computed with Troe type method, 3 parameters)
  // 7 - Troe-4 (Pressure dependent reactions computed with Troe type method, 4 parameters)
  std::vector<int> label;
  std::vector<int> rev_type;
  gxl::MatrixDyn<int> stoi_f, stoi_b;
  std::vector<int> order;
  std::vector<real> A, b, Ea;
  std::vector<real> A2, b2, Ea2;
  gxl::MatrixDyn<real> third_body_coeff;
  std::vector<real> troe_alpha, troe_t3, troe_t1, troe_t2;
};
}

#pragma once

#include "Parameter.h"
#include "Define.h"
#include "gxl_lib/Matrix.hpp"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct Species;
struct Reaction;
struct FlameletLib;

struct DParameter {
  DParameter() = default;

  explicit DParameter(Parameter &parameter, const Species &species, Reaction *reaction);

  int myid = 0;   // The process id of this process
  int dim = 3;    // The dimension of the problem
  int problem_type = 0; // The type of the problem. 0 - General problem, 1 - Mixing layer.

  // Number of equations and variables
  int n_var = 0;                    // The number of variables in the conservative variable
  int n_scalar = 0;               // The number of scalar variables
  int n_scalar_transported = 0;   // The number of scalar variables in the conservative equation, this is only different from n_scalar when we use flamelet model
  int n_ps = 0;                   // The number of passive scalars
  int n_spec = 0;                 // The number of species
  int i_fl = 0;                   // The index of flamelet variable in the scalar variable
  int i_fl_cv = 0;                // The index of flamelet variable in the conservative variable
  int i_turb_cv = 0;              // The index of turbulent variable in the conservative variable
  int i_ps = 0;                   // The index of passive scalar in the scalar variable
  int i_ps_cv = 0;                // The index of passive scalar in the conservative variable
  real *sc_ps = nullptr;          // The Schmidt number of passive scalars
  real *sct_ps = nullptr;         // The turbulent Schmidt number of passive scalars

  int inviscid_scheme = 0;  // The tag for inviscid scheme. 3 - AUSM+
  int reconstruction = 2; // The reconstruction method for inviscid flux computation
  int limiter = 0;  // The tag for limiter method
  int viscous_scheme = 0; // The tag for viscous scheme. 0 - Inviscid, 2 - 2nd order central discretization
  bool positive_preserving = false; // If we want to use positive preserving limiter
  bool gradPInDiffusionFlux = false;
  real entropy_fix_factor = 0; // The factor for entropy fix
  int shock_sensor = 0; // The tag for shock sensor.0 - Modified Ducros sensor; 1 - Modified Jameson sensor; 2 - Sensor based on density and pressure jump
  real sensor_eps = -1;     // The eps value for shock sensor
  real sensor_threshold = 0.1; // The criteria for hybrid scheme.

  real dt = 0; // The global time step. If 0, we use the local time step.
  real physical_time = 0; // The physical time of the simulation
  bool fixed_dt = false; // If we want to use a fixed time step in RK time integration

  int rans_model = 0;  // The tag for RANS model. 0 - Laminar, 1 - SA, 2 - SST
  int turb_implicit = 1;    // If we implicitly treat the turbulent source term. By default, implicitly treat(1), else, 0(explicit)
  int compressibility_correction = 0; // Which compressibility correction to be used. 0 - No compressibility correction, 1 - Wilcox's correction, 2 - Sarkar's correction, 3 - Zeman's correction

  int chemSrcMethod = 0;  // For finite rate chemistry, we need to know how to implicitly treat the chemical source

  int n_reac = 0;
  real Pr = 0.72;
  real cfl = 1;

  // real *mw = nullptr;
  real *imw = nullptr; // Inverse molecular weight
  real *gas_const = nullptr; // Gas constant for each species
#ifdef HighTempMultiPart
  int *n_temperature_range = nullptr;
  ggxl::MatrixDyn<real> temperature_cuts;
  ggxl::Array3D<real> therm_poly_coeff;
#else
  ggxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff;
  real *t_low = nullptr, *t_mid = nullptr, *t_high = nullptr;
#endif

  // Transport properties
  real *geometry = nullptr;
  real *LJ_potent_inv = nullptr;
  real *vis_coeff = nullptr;
  ggxl::MatrixDyn<real> WjDivWi_to_One4th;
  ggxl::MatrixDyn<real> sqrt_WiDivWjPl1Mul8;
  ggxl::MatrixDyn<real> binary_diffusivity_coeff;
  ggxl::MatrixDyn<real> kb_over_eps_jk; // Used to compute reduced temperature for diffusion coefficients
  real *ZRotF298 = nullptr;

  real Sc = 0.5;
  real Prt = 0.9;
  real Sct = 0.9;
  int hardCodedMech = 0; // If some mechanism is hardcoded, asign a value. 0 - No hardcoding, 1 - 9-species 19 reactions H2 mechanism (li, 2004)
  int *reac_type = nullptr;
  int *rev_type = nullptr;
  ggxl::MatrixDyn<int> stoi_f, stoi_b;
  int *reac_order = nullptr;
  real *A = nullptr, *b = nullptr, *Ea = nullptr;
  real *A2 = nullptr, *b2 = nullptr, *Ea2 = nullptr;
  ggxl::MatrixDyn<real> third_body_coeff;
  real *troe_alpha = nullptr, *troe_t3 = nullptr, *troe_t1 = nullptr, *troe_t2 = nullptr;

  // Reference value info, currently used by weno, only rho_ref and a_ref2 are used.
  real rho_ref = 1.0;
  real a_ref2 = 1.0;
  real v_ref = 1.0;
  real T_ref = 1.0;
  real mach_ref = 0;
  real p_ref = 1.0;
  real weno_eps_scale = 1.0;
  real v_char = 1.0; // characteristic velocity
  real delta_u = 0; // The difference between the velocity of the two streams in the mixing layer

  // stat data info
  int n_reyAve = 0;
  int n_reyAveScalar = 0;
  int n_species_stat = 0;
  bool perform_spanwise_average = false; // If we want to perform spanwise average
  bool rho_p_correlation = false;
  bool stat_tke_budget = false;
  bool stat_scalar_fluc_budget = false;
  bool stat_species_dissipation_rate = false;
  bool stat_species_velocity_correlation = false;
  int *reyAveVarIndex = nullptr;
  int *reyAveScalarIndex = nullptr;
  int *specStatIndex = nullptr;
  int statReyAveCount = 0;
  int statFavreAveCount = 0;
  int statViscousStressAddCount = 0;

  // For jicf computations
  int n_jet = 0; // The number of jets
  real jet_radius = 0; // The radius of the jet in meters
  real *xc_jet = nullptr; // The x coordinate of the jet center
  real *zc_jet = nullptr; // The z coordinate of the jet center
  real *jet_u = nullptr; // The u velocity of the jet
  real *jet_v = nullptr; // The v velocity of the jet
  real *jet_w = nullptr; // The w velocity of the jet
  real *jet_T = nullptr; // The temperature of the jet
  real *jet_p = nullptr; // The pressure of the jet
  real *jet_rho = nullptr; // The density of the jet
  ggxl::MatrixDyn<real> jet_sv; // The species of the jet

  int fluctuation_form = 0;
  real fluctuation_intensity = 0;
  real convective_velocity = 0; // The convective velocity of the mixing layer
  real delta_omega0 = 0; // The initial value of the vorticity thickness of the mixing layer
  int N_spanwise_wave = 1; // The number of spanwise waves in the mixing layer
  real x0_fluc, y0_fluc, z0_fluc; // The position of the fluctuation in the mixing layer

  // Sponge layer info
  // bool sponge_layer = false;
  // int sponge_function = 0; // 0 - (Nektar++, CPC, 2024)
  // int sponge_iter = 0;
  // int *sponge_scalar_iter = nullptr;
  // real sponge_sigma0 = 0;
  // real sponge_sigma1 = 0;
  // real sponge_sigma2 = 0;
  // real sponge_sigma3 = 0;
  // real sponge_sigma4 = 0;
  // real sponge_sigma5 = 0;
  // int spongeX = 0; // 0 - no sponge; 1 - sponge layer at x-; 2 - sponge layer at x+; 3 - sponge layer at both x- and x+.
  // int spongeY = 0; // 0 - no sponge; 1 - sponge layer at y-; 2 - sponge layer at y+; 3 - sponge layer at both y- and y+.
  // int spongeZ = 0; // 0 - no sponge; 1 - sponge layer at z-; 2 - sponge layer at z+; 3 - sponge layer at both z- and z+.
  // real spongeXMinusStart = 0;
  // real spongeXMinusEnd = 0;
  // real spongeXPlusStart = 0;
  // real spongeXPlusEnd = 0;
  // real spongeYMinusStart = 0;
  // real spongeYMinusEnd = 0;
  // real spongeYPlusStart = 0;
  // real spongeYPlusEnd = 0;
  // real spongeZMinusStart = 0;
  // real spongeZMinusEnd = 0;
  // real spongeZPlusStart = 0;
  // real spongeZPlusEnd = 0;

  // For mixing layer computation, we need to collect statistical data of the mixture fraction.
  real beta_diff_inv = 0, beta_o = 0;
  real nuc_mwc_inv = 0, nuh_mwh_inv = 0, half_nuo_mwo_inv = 0;

private:
  struct LimitFlow {
    // ll for lower limit, ul for upper limit.
    static constexpr int max_n_var = 5 + 2;
    real ll[max_n_var];
    real ul[max_n_var];
    real sv_inf[MAX_SPEC_NUMBER + 4];
  };

public:
  LimitFlow limit_flow{};

//  ~DParameter();
};

__global__ void update_dt_global(DParameter *param, real dt);
}

#pragma once

#include "Define.h"
#include "gxl_lib/Array.cuh"
#include "Parameter.h"
#include "Mesh.h"
#include "ChemData.h"
#include <curand_kernel.h>

namespace cfd {
struct Inflow;

struct DZone {
  DZone() = default;

  int mx = 0, my = 0, mz = 0, ngg = 0;
  int n_boundary = 0;
  int iMin = 0, iMax = 0, jMin = 0, jMax = 0, kMin = 0, kMax = 0;
  ggxl::Array3D<real> x, y, z;
  Boundary *boundary = nullptr;
  InnerFace *innerFace = nullptr;
  ParallelFace *parFace = nullptr;
  ggxl::Array3D<real> jac;
//  ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric;
  ggxl::VectorField3D<real> metric;
  ggxl::Array3D<real> wall_distance;
  // DES related grid scale
  ggxl::Array3D<real> des_delta;
  ggxl::Array2D<int> bType_il, bType_ir, bType_jl, bType_jr, bType_kl, bType_kr;
  // The value is 0 if it is an inner boundary point or periodic point, larger than 0 means it is a physical boundary

  // Conservative variable: 0-:rho, 1-:rho*u, 2-:rho*v, 3-:rho*w, 4-:rho*(E+V*V/2), 5->(4+Ns)-:rho*Y
  ggxl::VectorField3D<real> cv;
  ggxl::VectorField3D<real> bv; // Basic variable: 0-density, 1-u, 2-v, 3-w, 4-pressure, 5-temperature
  ggxl::VectorField3D<real> sv;
  // Scalar variables: [0,n_spec) - mass fractions; [n_spec,n_spec+n_turb) - turbulent variables
  ggxl::VectorField3D<real> udv;
  // Other variables that are used for specific purpose, which need to be specified by the user.
  ggxl::VectorField3D<real> bv_last; // Basic variable of last step
  ggxl::Array3D<real> mach;          // Mach number
  ggxl::Array3D<real> mul;           // Dynamic viscosity

  ggxl::VectorField3D<curandState> rng_state; // Random number generators for fluctuations
  ggxl::VectorField3D<real> fluc_val;         // The fluctuation values

  // Mixture variables
  ggxl::VectorField3D<real> rho_D; // the mass diffusivity of species
  ggxl::Array3D<real> gamma;       // specific heat ratio
  ggxl::Array3D<real> cp;          // specific heat for constant pressure
  ggxl::Array3D<real> acoustic_speed;
  ggxl::Array3D<real> thermal_conductivity; // Thermal conductivity

  // chemical jacobian matrix or diagonal
  ggxl::VectorField3D<real> chem_src_jac;
  ggxl::Array3D<real> reaction_timeScale;

  // Turbulent variables
  ggxl::Array3D<real> mut;                // turbulent viscosity
  ggxl::VectorField3D<real> turb_src_jac; // turbulent source jacobian, for implicit treatment
  // Flamelet variables
  ggxl::Array3D<real> scalar_diss_rate; // scalar dissipation rate

  // Variables used in computation
  ggxl::VectorField3D<real> dq;       // The residual for flux computing
  ggxl::VectorField3D<real> fFlux; // The fluxes at the cell faces.
  ggxl::VectorField3D<real> gFlux; // The fluxes at the cell faces.
  ggxl::VectorField3D<real> hFlux; // The fluxes at the cell faces.
  ggxl::VectorField3D<real> vis_flux; // The fluxes at the cell faces.
  // ggxl::VectorField3D<real> grad_bv;  // The gradient of basic variables

  ggxl::VectorField3D<real> dq0;          // Used when DPLUR is enabled
  ggxl::VectorField3D<real> dqk;          // Used when DPLUR is enabled
  ggxl::Array3D<real[3]> inv_spectr_rad;  // inviscid spectral radius. Used when DPLUR type temporal scheme is used.
  ggxl::Array3D<real[3]> visc_spectr_rad; // viscous spectral radius.
  ggxl::Array3D<real> dt_local;           //local time step. Used for steady flow simulation
  ggxl::Array3D<real> entropy_fix_delta;  // The coefficient for entropy fix, which is used in Roe scheme.
  ggxl::Array3D<real> shock_sensor;       //The sensor which tells us the shock position.

  // RK-3 related variables
  ggxl::VectorField3D<real> qn; // The conservative variables from the last step.

  // WENO-related variables
  //  ggxl::VectorField3D<real> Fp, Fm, flux, f_1st; // The fluxes at the cell faces.

  // Dual-time stepping related variables
  ggxl::VectorField3D<real> qn1;          // The conservative variables from the last iteration.
  ggxl::VectorField3D<real> qn_star;      // qn_star=(4*qn-qn1)/(2J*dt)
  ggxl::VectorField3D<real> in_last_step; // The basic variables of the last inner iteration.

  // wu splitting related variables
  ggxl::VectorField3D<real> Cn;

  // single point statistics
  // collect arrays
  ggxl::VectorField3D<real> collect_reynolds_1st;
  ggxl::VectorField3D<real> collect_reynolds_2nd;
  ggxl::VectorField3D<real> collect_favre_1st;
  ggxl::VectorField3D<real> collect_favre_2nd;
  // mean arrays
  ggxl::VectorField3D<real> stat_reynolds_1st;
  ggxl::VectorField3D<real> stat_reynolds_2nd;
  ggxl::VectorField3D<real> stat_favre_1st;
  ggxl::VectorField3D<real> stat_favre_2nd;

  // other statistical variables
  ggxl::VectorField3D<real> collect_tke_budget;
  ggxl::VectorField3D<real> collect_scalar_budget;
  ggxl::VectorField3D<real> collect_scalar_vel_correlation;

  // sponge layer related mean conservative variables
  ggxl::VectorField3D<real> sponge_mean_cv;
};

struct DParameter;

struct Field {
  Field(Parameter &parameter, const Block &block_in);

  void
  initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow> &inflows,
    const std::vector<real> &xs, const std::vector<real> &xe,
    const std::vector<real> &ys, const std::vector<real> &ye,
    const std::vector<real> &zs, const std::vector<real> &ze,
    const cfd::Species &species) const;

  void setup_device_memory(const Parameter &parameter);

  void copy_data_from_device(const Parameter &parameter);

  void deallocate_memory(const Parameter &parameter);

  int n_var = 5;
  const Block &block;
  ggxl::VectorField3DHost<real> bv; // basic variables, including density, u, v, w, p, temperature
  ggxl::VectorField3DHost<real> sv;
  // passive scalar variables, including species mass fractions, turbulent variables, mixture fractions, etc.
  ggxl::VectorField3DHost<real> ov;
  // other variables used in the computation, e.g., the Mach number, the mut in turbulent computation, scalar dissipation rate in flamelet, etc.
  ggxl::VectorField3DHost<real> udv;              // User defined variables.
  ggxl::VectorField3DHost<curandState> rng_state; // Random number generators for fluctuations
  ggxl::VectorField3DHost<real> fluc_val;         // The fluctuation values

  // The following data is used for collecting statistics, whose memory is only allocated when we activate the statistics.
  //  ggxl::VectorField3DHost<real> firstOrderMoment, secondOrderMoment, userDefinedStatistics;
  //  ggxl::VectorField3DHost<real> mean_value, reynolds_stress_tensor_and_rms, user_defined_statistical_data;
  // single point statistics
  // collect arrays
  ggxl::VectorField3DHost<real> collect_reynolds_1st;
  ggxl::VectorField3DHost<real> collect_reynolds_2nd;
  ggxl::VectorField3DHost<real> collect_favre_1st;
  ggxl::VectorField3DHost<real> collect_favre_2nd;
  // mean arrays
  ggxl::VectorField3DHost<real> stat_reynolds_1st;
  ggxl::VectorField3DHost<real> stat_reynolds_2nd;
  ggxl::VectorField3DHost<real> stat_favre_1st;
  ggxl::VectorField3DHost<real> stat_favre_2nd;

  // other statistical variables
  ggxl::VectorField3DHost<real> collect_tke_budget;
  ggxl::VectorField3DHost<real> collect_scalar_budget;
  ggxl::VectorField3DHost<real> collect_scalar_vel_correlation;

  // sponge layer related mean conservative variables
  ggxl::VectorField3DHost<real> sponge_mean_cv;

  DZone *d_ptr = nullptr;
  DZone *h_ptr = nullptr;
};
}

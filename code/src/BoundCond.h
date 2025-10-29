/**
 * @file BoundCond.h
 * @brief Boundary conditions
 * @details This file contains the definition of boundary conditions. We currently support the following boundary conditions:
 * - 5 Inflow (Inflow) This is the supersonic inflow boundary condition.
 * - 2 Wall (Wall) This is the wall boundary condition.
 * - 6 Outflow (Outflow) This is the supersonic outflow boundary condition.
 * - 3 Symmetry (Symmetry) This is the symmetry boundary condition.
 * - 4 Farfield (FarField) This is the farfield boundary condition with Riemann invariant.
 * - 7 Subsonic inflow (SubsonicInflow) This is the subsonic inflow boundary condition.
 * @version 0.1
 * @date 2023-09-08
 * @note Initial version of the file with basic boundary conditions.
 *
 */
#pragma once

#include "ChemData.h"
#include "Constants.h"

namespace cfd {
struct Inflow {
  explicit Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter);

  explicit Inflow(const std::string &inflow_name, const Species &spec, const Parameter &parameter);

  [[nodiscard]] std::tuple<real, real, real, real, real, real> var_info() const;

  void copy_to_gpu(Inflow *d_inflow, const Parameter &parameter);

  int label = 5;
  // The type of the inflow
  // 0 - constant inflow
  // 1 - profile inflow
  // 2 - mixing layer inflow
  int inflow_type = 0;

  // This set is used for inflow_type = 0 (constant inflow).
  // When inflow_type = 2, this set is used as the upper part.
  real mach = -1;
  real pressure = -1;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real aoa = -1000;
  real u = 1, v = 0, w = 0;
  real *sv = nullptr;
  real mw = mw_air;
  real viscosity = 0;
  real mut = 0;
  real reynolds_number = -1;
  real mixture_fraction = -1;

  // When inflow_type = 2, this set is used as the lower part.
  real p_lower = 101325;
  real t_lower = -1;
  real density_lower = -1;
  real u_lower = 1, v_lower = 0, w_lower = 0;
  real *sv_lower = nullptr;
  real mw_lower = mw_air;
  real mu_lower = 0;
  real mut_lower = 0;
  real mixture_fraction_lower = -1;
  // When inflow_type = 2, we need the initial vorticity thickness
  // When white noise, we use y<4*delta_omega to add the moise, use a big value when not mixing layers to make the whole face available.
  real delta_omega = 1000;

  int profile_idx = 0;

  // The method of fluctuation.
  // 0 - no fluctuation
  // 1 - random fluctuation
  // 2 - profile with real and imaginary part acquired by stability analysis.
  int fluctuation_type = 0;
  // If the fluctuation type is 1, this is the intensity of the fluctuation.
  real fluctuation_intensity = 0;
  real streamwise_wavelength = 0;
  real spanwise_wavelength = 0;
  // If the fluctuation type is 2, the following parameters should be given.
  int fluc_prof_idx = 0;
  real fluctuation_frequency = 0;

  // For random values
  real *random_phase = nullptr;
};

struct Wall {
  //  explicit Wall(int type_label, std::ifstream &bc_file);

  explicit Wall(const std::map<std::string, std::variant<std::string, int, real>> &info, Parameter &parameter);

  enum class ThermalType {
    isothermal, adiabatic, equilibrium_radiation
  };

  int label = 2;
  ThermalType thermal_type = ThermalType::isothermal;

  // If the wall thermal type is radiation equilibrium, this is set as the inflow temperature.
  real temperature{300};
  real emissivity{0.8};
  // 0 - no fluctuation
  // 1 - Pirozzoli & Li fluctuations, suppose the spanwise z direction is periodic, and not partitioned by MPI processes.
  int fluctuation_type = 0;
  real fluctuation_intensity = 0;
  real streamwise_wavelength = 0;
  real spanwise_wavelength = 0;
  real fluctuation_frequency = 0;
  real fluctuation_x0 = 0;
  real fluctuation_x1 = 0;
  real Zl[10] = {};
  real Tm[5] = {};
  // For circular cylinder, the shock wave may not develop to the stagnation point. Blow it to develop!
  int if_blow_shock_wave = 0;
};

struct Outflow {
  explicit Outflow(const std::string &inflow_name, Parameter &parameter);

  int label = 6;

  int nonReflecting = 0;
};

struct Symmetry {
  explicit Symmetry(const std::string &inflow_name, const Parameter &parameter);

  int label = 3;
};

struct FarField {
  explicit FarField(const std::string &inflow_name, Species &spec, Parameter &parameter);

  void copy_to_gpu(FarField *d_farfield, const Parameter &parameter);

  int label = 4;
  real mach = -1;
  real pressure = -1;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
  real acoustic_speed = 0;
  real entropy = 0;
  real *sv = nullptr;
  real mw = mw_air;
  real viscosity = 0;
  real mut = 0;
  real specific_heat_ratio = gamma_air;
  real reynolds_number = -1;
};

/**
 * @brief Subsonic inflow boundary condition, default label is 7
 * @details This is the subsonic inflow boundary condition.
 * @note We need to specify the ratio of total pressure and reference pressure, the ratio of total temperature and reference temperature, and the velocity direction in u, v, and w.
 * The implementation references the Fun3D manual and the CFL3D code.
 * @refitem Carlson, J.-R. Inflow/Outflow Boundary Conditions with Application to FUN3D. (2011).
 * @warning This implementation can only be used in air simulation. The multi-species simulation is not supported.
 */
struct SubsonicInflow {
  explicit SubsonicInflow(const std::string &inflow_name, Parameter &parameter);

  void copy_to_gpu(SubsonicInflow *d_inflow, const Parameter &parameter);

  int label = 7;
  real total_pressure = 101325;
  real total_temperature = -1;
  real u = 1, v = 0, w = 0;
  real *sv = nullptr;
  real mut = 0;
};

/**
 * @brief Back pressure boundary condition, default label is 9
 * @details This boundary condition specifies the back pressure at the outlet.
 * @note This boundary condition is only used in the subsonic flow simulation, which you make sure that the outflow is subsonic.
 * All quantities except pressure are extrapolated from the interior, while the pressure is specified by the user as the back pressure.
 */
struct BackPressure {
  explicit BackPressure(const std::string &name, Parameter &parameter);

  int label = 9;
  real pressure = -1;
};

struct Periodic {
  explicit Periodic(const std::string &name, const Parameter &parameter);

  int label = 8;
};
}

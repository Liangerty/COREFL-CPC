#pragma once

#include <curand_kernel.h>
#include "BoundCond.h"
#include "Mesh.h"
#include "Field.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct BCInfo {
  int label = 0;
  int n_boundary = 0;
  int2 *boundary = nullptr;
};

void read_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

void read_lst_profile(const Boundary &boundary, const std::string &file, const Block &block, const Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

void read_dat_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

struct DBoundCond {
  DBoundCond() = default;

  void initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter);

  void link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const;

  template<MixtureModel mix_model> void apply_boundary_conditions(const Block &block, Field &field, DParameter *param,
    int step = -1) const;

  template<MixtureModel mix_model>
  void nonReflectingBoundary(const Block &block, Field &field, DParameter *param) const;

  void write_df(Parameter &parameter, const Mesh &mesh) const;

  // There may be time-dependent BCs, which need to be updated at each time step.
  // E.g., the turbulent library method, we need to update the profile and fluctuation.
  // E.g., the NSCBC
  // Therefore, this function may be extended in the future to be called "time-dependent bc update".
  // void time_dependent_bc_update(const Mesh &mesh, std::vector<Field> &field, DParameter *param, Parameter &parameter) const;

  int n_wall = 0, n_symmetry = 0, n_inflow = 0, n_outflow = 0, n_farfield = 0, n_subsonic_inflow = 0, n_back_pressure =
          0, n_periodic = 0;
  BCInfo *wall_info = nullptr;
  BCInfo *symmetry_info = nullptr;
  BCInfo *inflow_info = nullptr;
  BCInfo *outflow_info = nullptr;
  BCInfo *farfield_info = nullptr;
  BCInfo *subsonic_inflow_info = nullptr;
  BCInfo *back_pressure_info = nullptr;
  BCInfo *periodic_info = nullptr;
  Wall *wall = nullptr;
  Symmetry *symmetry = nullptr;
  Inflow *inflow = nullptr;
  Outflow *outflow = nullptr;
  FarField *farfield = nullptr;
  SubsonicInflow *subsonic_inflow = nullptr;
  BackPressure *back_pressure = nullptr;
  Periodic *periodic = nullptr;

  // Non-reflecting boundary
  std::vector<int> nonReflectingBCs{};

  // Profiles
  // For now, please make sure that all profiles are in the same plane, that the plane is not split into several parts.
  // There may be inflow with values of ghost grids also given.
  std::vector<ggxl::VectorField3D<real>> profile_hPtr_withGhost = {};
  ggxl::VectorField3D<real> *profile_dPtr_withGhost = nullptr;
  // Fluctuation profiles, with real part and imaginary part given for basic variables
  ggxl::VectorField3D<real> *fluctuation_dPtr = nullptr;

  // Digital filter related
  int n_df_face = 0;
  std::vector<int> df_label = {};
  std::vector<int> df_related_block = {};
  constexpr static int DF_N = 50;
  // Random values for digital filter.
  // E.g., the dimensions are often like (ny,nz,3), where 3 is for 3 components of velocity.
  ggxl::VectorField2D<real> *random_values_hPtr = nullptr;
  ggxl::VectorField2D<real> *random_values_dPtr = nullptr;
  ggxl::VectorField1D<real> *df_lundMatrix_dPtr = nullptr; // Lund matrix for digital filter
  // (0:my-1, 0:2*DF_N, 0:2): my*(2N+1)*3, the second index jj corresponds to jj-N
  ggxl::VectorField2D<real> *df_by_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_bz_dPtr = nullptr;
  ggxl::VectorField2D<curandState> *rng_states_hPtr = nullptr; // Random number generator states for digital filter
  ggxl::VectorField2D<curandState> *rng_states_dPtr = nullptr; // Random number generator states for digital filter
  ggxl::VectorField2D<real> *df_fy_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_old_hPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_old_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_new_hPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_new_dPtr = nullptr;

  ggxl::VectorField2DHost<curandState> *df_rng_state_cpu = nullptr;
  ggxl::VectorField2DHost<real> *df_velFluc_cpu = nullptr;

  curandState *rng_d_ptr = nullptr;

private:
  void initialize_digital_filter(Parameter &parameter, Mesh &mesh);

  void initialize_df_memory(const Mesh &mesh, const std::vector<int> &N1, const std::vector<int> &N2);

  void get_digital_filter_lund_matrix(Parameter &parameter, const std::vector<int> &N1,
    const std::vector<std::vector<real>> &scaled_y) const;

  void get_digital_filter_convolution_kernel(Parameter &parameter, const std::vector<int> &N1,
    const std::vector<std::vector<real>> &y_scaled, real dz) const;

  void generate_random_numbers(int iFace, int my, int mz, int ngg) const;

  void apply_convolution(int iFace, int my, int mz, int ngg) const;

  void initialize_profile_and_rng(Parameter &parameter, Mesh &mesh, const Species &species, std::vector<Field> &field);

  void compute_fluctuations(const DParameter *param, DZone *zone, const Inflow *inflowHere, int iFace, int my, int mz,
    int ngg) const;
};

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, int n_bc, int **sep, int blk_idx, int n_block,
  BCInfo *bc_info);

void link_boundary_and_condition(const std::vector<Boundary> &boundary, const BCInfo *bc, int n_bc, int **sep,
  int i_zone);

__global__ void initialize_rng(curandState *rng_states, int size, int64_t time_stamp);

__global__ void initialize_rest_rng(ggxl::VectorField2D<curandState> *rng_states, int iFace, int64_t time_stamp, int dy,
  int dz, int ngg, int my, int mz);

__global__ void initialize_rng(DZone *zone, int n_rand);

void write_rng(const Mesh &mesh, Parameter &parameter, std::vector<Field> &field);
} // cfd

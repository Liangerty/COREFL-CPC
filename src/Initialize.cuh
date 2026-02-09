#pragma once

#include "Parameter.h"
#include "Field.h"
#include <mpi.h>

namespace cfd {
template<MixtureModel mix_model> void initialize_basic_variables(Parameter &parameter, const Mesh &mesh,
  std::vector<Field> &field, Species &species);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

template<MixtureModel mix_model> void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species);

template<MixtureModel mix_model> void read_flowfield_with_same_block(Parameter &parameter, const Mesh &mesh,
  std::vector<Field> &field, Species &species, const std::vector<int> &blk_order, MPI_Offset offset_data,
  const std::vector<int> &mx, const std::vector<int> &my, const std::vector<int> &mz, int n_var_old,
  const std::vector<int> &index_order, MPI_File &fp, std::array<int, 2> &old_data_info);

template<MixtureModel mix_model> void read_flowfield_by_0Order_interpolation(Parameter &parameter, const Mesh &mesh,
  std::vector<Field> &field, Species &species, const std::vector<int> &blk_order, MPI_Offset offset_data,
  const std::vector<int> &mx, const std::vector<int> &my, const std::vector<int> &mz, int n_var_old,
  const std::vector<int> &index_order, MPI_File &fp, std::array<int, 2> &old_data_info);

template<MixtureModel mix_model> void read_2D_for_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species);

void initialize_mixing_layer(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, const Species &species);

__global__ void initialize_mixing_layer_with_profile(ggxl::VectorField3D<real> *profile_dPtr, int profile_idx,
  DZone *zone, int n_scalar);

__global__ void initialize_mixing_layer_with_info(DZone *zone, const real *var_info, int n_spec, real delta_omega,
  int n_turb, int n_fl, int n_ps);

/**
 * @brief To relate the order of variables from the flowfield files to bv, yk, turbulent arrays
 * @param parameter the parameter object
 * @param var_name the array which contains all variables from the flowfield files
 * @param species information about species
 * @param old_data_info the information about the previous simulation, the first one tells if species info exists, the second one tells if turbulent var exists
 * @return an array of orders. 0~5 means density, u, v, w, p, T; 6~5+ns means the species order, 6+ns~... means other variables such as mut...
 */
template<MixtureModel mix_model> std::vector<int> identify_variable_labels(Parameter &parameter,
  std::vector<std::string> &var_name, Species &species, std::array<int, 2> &old_data_info);

void initialize_spec_from_inflow(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

void initialize_turb_from_inflow(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

void initialize_mixture_fraction_from_species(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species);

void expand_2D_to_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field);

// Several initial conditions for some canonical problems
void initialize_sinWaveProp(const Mesh &mesh, std::vector<Field> &field);
}

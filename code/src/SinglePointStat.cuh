#pragma once

#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"
#include <mpi.h>

namespace cfd {
__global__ void collect_singlePointStat_1ghostLayer_Step1(DZone *zone, DParameter *param);

__global__ void collect_singlePointStat_1ghostLayer_Step2(DZone *zone, DParameter *param);

__global__ void collect_single_point_additional_statistics(DZone *zone, DParameter *param);

class SinglePointStat {
public:
  explicit SinglePointStat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field,
                           const Species &_species);

  void initialize_statistics_collector();

  void collect_data(DParameter *param);

  void export_statistical_data();

private:
  int n_reyAve = 2;
  int n_reyAveScalar = 0;
  int n_favAve = 4;
  int n_rey2nd = 2;
  int n_fav2nd = 7;
  std::vector<std::string> reyAveVar = {"rho", "p"};
  std::vector<int> reyAveVarIndex = {0, 4};
  std::vector<std::string> reyAveScalar = {};
  std::vector<int> reyAveScalarIndex = {};
  std::vector<int> counter_rey1stScalar;
  std::vector<std::string> favAveVar = {"rhoU", "rhoV", "rhoW", "rhoT"};
  std::vector<std::string> rey2ndVar = {"RhoRho", "pp"};
  std::vector<int> counter_rey2nd;
  std::vector<std::string> fav2ndVar = {"rhoUU", "rhoVV", "rhoWW", "rhoUV", "rhoUW", "rhoVW", "rhoTT"};
  std::vector<int> counter_fav2nd;

  // available statistics
  std::vector<int> species_stat_index;
  int n_species_stat = 0;
  int n_ps = 0;
  bool tke_budget = false;
  bool scalar_fluc_budget = false;
  bool species_velocity_correlation = false;
  bool species_dissipation_rate = false;
  std::vector<int> counter_scalar_fluc_budget;
  std::vector<int> counter_species_velocity_correlation;
  std::vector<int> counter_species_dissipation_rate;
  MPI_Offset offset_tke_budget{0};
  MPI_Offset offset_scalar_fluc_budget{0};
  MPI_Offset offset_species_velocity_correlation{0};
  MPI_Offset offset_species_dissipation_rate{0};

  int myid{0};
  int ngg = 1;
  // Data to be bundled
  Parameter &parameter;
  const Mesh &mesh;
  std::vector<Field> &field;
  const Species &species;

  MPI_Offset offset_unit[4] = {0, 0, 0, 0};
  std::vector<MPI_Datatype> ty_1gg, ty_0gg;
public:
  std::vector<int> counter_rey1st;
  std::vector<int> counter_fav1st;
  std::vector<int> counter_tke_budget;

private:
  void init_stat_name();

  void compute_offset_for_export_data();

  void read_previous_statistical_data();
};
} // cfd

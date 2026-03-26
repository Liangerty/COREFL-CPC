#pragma once

#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"
#include <mpi.h>

namespace cfd {
__global__ void collect_singlePointStat_1ghostLayer_Step1(DZone *zone, DParameter *param);

__global__ void collect_singlePointStat_spanAvg_Step1(DZone *zone, DParameter *param);

__global__ void collect_singlePointStat_1ghostLayer_Step2(DZone *zone, DParameter *param);

__global__ void collect_single_point_additional_statistics(DZone *zone, DParameter *param);

__global__ void compute_statistical_data_spanwise_average(DZone *zone, const DParameter *param, int N);

class SinglePointStat {
public:
  explicit SinglePointStat(Parameter &_parameter, const Mesh &_mesh, std::vector<Field> &_field,
    const Species &_species);

  void initialize_statistics_collector();

  void collect_data(DParameter *param);

  void export_statistical_data(DParameter *param);

private:
  struct ThicknessRecord {
    real x = 0;
    real yc = 0;
    real delta_theta = 0;
    real delta_omega = 0;
    real delta_vis = 0;
  };

  struct ThicknessReferenceState {
    real u1 = 0;
    real u2 = 0;
    real rho_ref = 1;
    real convective_velocity = 0;
    real u11 = 0;
    real u22 = 0;
    real delta_u = 0;
    bool upper_u_faster = true;
  };

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
  bool if_collect_spec_favreAvg = true;
  bool if_collect_2nd_moments = true;
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
  bool perform_spanwise_average = false;

  MPI_Offset offset_unit[4] = {0, 0, 0, 0};
  std::vector<MPI_Datatype> ty_1gg, ty_0gg;

  // plot-related variables
  MPI_Offset offset_header{0};
  int n_plot{0};
  MPI_Offset *offset_minmax_var = nullptr;
  MPI_Offset *offset_var = nullptr;

public:
  std::vector<int> counter_rey1st;
  std::vector<int> counter_fav1st;
  std::vector<int> counter_tke_budget;

private:
  void init_stat_name();

  void compute_offset_for_export_data();

  void read_previous_statistical_data();

  void prepare_for_statistical_data_plot(const Species &species);

  ThicknessReferenceState get_thickness_reference_state() const;

  void compute_thickness_for_block(int blk, const ThicknessReferenceState &ref,
                                   std::vector<ThicknessRecord> &records) const;

  void write_thickness_file(const std::vector<ThicknessRecord> &local_records) const;

  void plot_statistical_data(DParameter *param) const;
};
} // cfd

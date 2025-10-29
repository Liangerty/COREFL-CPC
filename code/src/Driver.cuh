#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "gxl_lib/Time.h"
#include "ChemData.h"
#include "Field.h"
#include "BoundCond.cuh"
#include "ParameterSet.h"
#include "SinglePointStat.cuh"

namespace cfd {

template<MixtureModel mix_model>
struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_);

  void initialize_computation();

  void deallocate();

public:
  int myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  Parameter &parameter;
  // ParameterSet ps;
  Species spec;
  Reaction reac;
  std::vector<Field> field; // The flowfield data of the simulation. Every block is a "Field" object
  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};
  // Statistical data
  SinglePointStat stat_collector;
  //  StatisticsCollector stat_collector;
};

template<MixtureModel mix_model>
void Driver<mix_model>::deallocate() {
  for (auto& f:field){
    f.deallocate_memory(parameter);
  }
}

void write_reference_state(Parameter &parameter, const Species &species);

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, int n_point_times3);

__global__ void transfer_counter_to_device(DParameter *param, int count_rey, int count_fav, int count_tkeBudget);

} // cfd
#pragma once

#include "Define.h"
#include <vector>
#include <mpi.h>
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"
#include <cuda_runtime.h>

namespace cfd {
template<MixtureModel mix_model> void data_communication(const Mesh &mesh, std::vector<Field> &field,
  const Parameter &parameter, int step, DParameter *param);

template<MixtureModel mix_model> __global__ void inner_communication(DZone *zone, DZone *tar_zone, int i_face,
  DParameter *param);

template<MixtureModel mix_model> void parallel_communication(const Mesh &mesh, std::vector<Field> &field, int step,
  const Parameter &parameter, DParameter *param);

__global__ void setup_data_to_be_sent(const DZone *zone, int i_face, real *data, const DParameter *param);

template<MixtureModel mix_model> __global__ void assign_data_received(DZone *zone, int i_face, const real *data,
  DParameter *param);

__global__ void setup_data_to_be_sent(const DZone *zone, int i_face, real *data, const DParameter *param, int task);

__global__ void assign_data_received(DZone *zone, int i_face, const real *data, DParameter *param, int task);

void exchange_value(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter, DParameter *param,
  int task);

__global__ void inner_exchange(DZone *zone, DZone *tar_zone, int i_face, DParameter *param, int task);

void parallel_exchange(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter, DParameter *param,
  int task);

__global__ void periodic_exchange(DZone *zone, DParameter *param, int task, int i_face);
}

#pragma once

#include "Define.h"
#include "Driver.cuh"

namespace cfd {
template<MixtureModel mix_model> void compute_cn(std::vector<Field> &field, DParameter *param, Parameter &parameter,
  const Mesh &mesh, int n_block, dim3 tpb, dim3 *bpg);

template<MixtureModel mix_model> void wu_splitting(Driver<mix_model> &driver);
}

#pragma once
#include "DataCommunication.cuh"
#include "Define.h"
#include "FiniteRateChem.cuh"
#include "IOManager.h"
#include "RK.cuh"
#include "SchemeSelector.cuh"
#include "TimeAdvanceFunc.cuh"

namespace cfd {
template<MixtureModel mix_model>
void RK_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  DBoundCond &bound_cond, int n_block, int step, dim3 tpb, dim3 *bpg, real dt);

template<MixtureModel mix_model>
void reaction_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  int n_block, dim3 tpb, dim3 *bpg, real dt);

template<MixtureModel mix_model>
void strang_splitting(Driver<mix_model> &driver);

template<MixtureModel mix_model>
void compute_cn(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  int n_block, dim3 tpb, dim3 *bpg);

template<MixtureModel mix_model>
void wu_reaction_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  int n_block, dim3 tpb, dim3 *bpg, real dt);

template<MixtureModel mix_model>
void wu_splitting(Driver<mix_model> &driver);
}

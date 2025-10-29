#pragma once
#include <random>
#include "DParameter.cuh"
#include "Field.h"

namespace cfd {
void compute_fluctuation(const Block &block, DZone *zone, DParameter *param, int form, Parameter &parameter,
  std::uniform_real_distribution<real> &dist, std::mt19937 &gen);

__global__ void ferrer_fluctuation(DZone *zone, DParameter *param, real eps1, real eps2);

__global__ void update_values_with_fluctuations(DZone *zone, DParameter *param);
}

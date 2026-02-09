#pragma once

#include "Mesh.h"

namespace cfd {
void compute_jac_metric(int myid, Block &block, Parameter &parameter);
} // namespace cfd

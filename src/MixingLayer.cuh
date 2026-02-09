#pragma once

#include "Define.h"
#include <vector>

namespace cfd{
class Parameter;
class Mesh;
struct Field;
struct Species;

void
get_mixing_layer_info(Parameter &parameter, const Species &species, std::vector<real> &var_info);
}
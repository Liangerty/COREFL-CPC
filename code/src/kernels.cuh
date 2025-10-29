#pragma once

#include "Define.h"

namespace cfd{
struct DParameter;
__global__ void modify_cfl(DParameter *param, real cfl);
}

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using real = double;
using uint = unsigned int;

// The therm.dat may have only 2 ranges, or multiple temperature ranges.
// When standard Chemkin is used with 2 temperature ranges, we define it as Combustion2Part.
// When the therm.dat has multiple temperature ranges, we define it as HighTempMultiPart.
// #define Combustion2Part

enum class MixtureModel {
  Air, Mixture,
  // Species mixing
};

enum class OutputTimeChoice {
  Instance,
  // Output the instant values, which would overwrite its previous values
  TimeSeries,
  // Output the values as a time series, which would create new files with time stamp
};

namespace cfd {
namespace rok4e {
constexpr real gamma = 0.572816062482135;
constexpr real alpha21 = 0.432364435748567;
constexpr real alpha31 = -0.51421131687617;
constexpr real alpha32 = 1.38227114461736;
constexpr real gamma21Gamma = -0.602765307997356 / gamma;
constexpr real gamma31Gamma = -1.389195789724843 / gamma;
constexpr real gamma32Gamma = 1.072950969011413 / gamma;
constexpr real gamma41Gamma = 0.992356412977094 / gamma;
constexpr real gamma42Gamma = -1.390032613873701 / gamma;
constexpr real gamma43Gamma = -0.440875890223325 / gamma;
constexpr real b1 = 0.194335256262729, b2 = 0.483167813989227, b4 = 0.322496929748044;
constexpr real e1 = -0.217819895945721 - b1, e2 = 1.03130847478467 - b2, e3 = 0.186511421161047, e4 = -b4;
// constexpr real RTol = 1e-4, ATol = 1e-8;
constexpr real RTol = 1e-9, ATol = 1e-13;
}
}
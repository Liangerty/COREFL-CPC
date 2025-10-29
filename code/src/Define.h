#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using real = double;
using uint = unsigned int;

// The therm.dat may have only 2 ranges, or multiple temperature ranges.
// When standard Chemkin is used with 2 temperature ranges, we define it as Combustion2Part.
// When the therm.dat has multiple temperature ranges, we define it as HighTempMultiPart.
#define Combustion2Part

enum class MixtureModel{
  Air,
  Mixture,  // Species mixing
};

enum class OutputTimeChoice{
  Instance,   // Output the instant values, which would overwrite its previous values
  TimeSeries, // Output the values as a time series, which would create new files with time stamp
};

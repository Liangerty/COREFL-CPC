#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define __host__
#endif

using real = double;
using uint = unsigned int;

// The therm.dat may have only 2 ranges, or multiple temperature ranges.
// When standard Chemkin is used with 2 temperature ranges, we define it as Combustion2Part.
// When the therm.dat has multiple temperature ranges, we define it as HighTempMultiPart.
#define Combustion2Part

enum class TurbulenceMethod{
  Laminar,
  RANS,
  LES,
//  ILES,
//  DNS
};

enum class MixtureModel{
  Air,
  Mixture,  // Species mixing
  FR,       // Finite Rate
  MixtureFraction,  // Species + mixture fraction + mixture fraction variance are solved.
  FL,       // Flamelet Model
};

enum class OutputTimeChoice{
  Instance,   // Output the instant values, which would overwrite its previous values
  TimeSeries, // Output the values as a time series, which would create new files with time stamp
};

struct reconstruct_bv{};
struct reconstruct_cv{};

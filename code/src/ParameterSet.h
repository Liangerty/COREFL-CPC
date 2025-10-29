#pragma once

namespace cfd {
class Parameter;

struct ParameterSet {
  explicit ParameterSet() = default;

  void set_parameters(const Parameter &param);

  // Members used in the simulation
};
} // cfd

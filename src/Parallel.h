#pragma once

namespace cfd {
/**
 * \brief The class controlling the MPI message of current simulation
 */
class MpiParallel {
public:
  MpiParallel() = default;

  MpiParallel(const MpiParallel &) = delete;

  MpiParallel(MpiParallel &&) = delete;

  MpiParallel &operator=(const MpiParallel &) = delete;

  MpiParallel operator=(MpiParallel &&) = delete;

  static double get_wall_time();

  static void barrier();

  static void exit();

  ~MpiParallel();
};
}

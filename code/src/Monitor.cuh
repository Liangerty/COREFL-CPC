#pragma once

#include <mpi.h>
#include "Parameter.h"
#include "gxl_lib/Array.cuh"

namespace cfd {
struct Field;
class Mesh;

struct DeviceMonitorData {
  int n_bv{0}, n_sv{0}, n_var{0};
  int *bv_label{nullptr};
  int *sv_label{nullptr};
  int *bs_d{nullptr}, *is_d{nullptr}, *js_d{nullptr}, *ks_d{nullptr};
  int *disp{nullptr}, *n_point{nullptr};
  ggxl::Array3D<real> data;
};

class BlockMonitor {
public:
  explicit BlockMonitor(const Parameter &parameter, const Mesh &mesh_);

  void output_data(const Parameter &parameter, const std::vector<Field> &field, real t) const;

private:
  int n_bv{0}, n_sv{0}, n_ov{0}, n_var{0};
  std::vector<int> bv_label, sv_label, ov_label;

  int n_block_mon{0};
  int n_group{0};
  std::vector<std::string> group_name;
  std::vector<std::array<int, 7>> group_range;

  MPI_Datatype *ty;

  // Utility functions
  void setup_labels_to_monitor(const Parameter &parameter);
};

class Monitor {
public:
  explicit Monitor(const Parameter &parameter, const Species &species, const Mesh &mesh_);

  void monitor_point(int step, real physical_time, const std::vector<Field> &field);

  void output_point_monitors();

  void output_block_monitors(const Parameter &parameter, const std::vector<Field> &field, real t) const {
    block_monitor.output_data(parameter, field, t);
  }

  ~Monitor();

private:
  BlockMonitor block_monitor;
  int output_file{0};
  int step_start{0};
  int counter_step{0};
  int n_block{0};
  int n_bv{0}, n_sv{0}, n_var{0};
  std::vector<int> bs_h, is_h, js_h, ks_h;
  int n_point_total{0};
  std::vector<int> n_point;
  std::vector<int> disp;
  ggxl::Array3DHost<real> mon_var_h;
  DeviceMonitorData *h_ptr, *d_ptr{nullptr};
  std::vector<FILE *> files;

  const Mesh &mesh;

private:
  // Utility functions
  std::vector<std::string> setup_labels_to_monitor(const Parameter &parameter, const Species &species);
};

struct DZone;
__global__ void record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos,
  real physical_time);
} // cfd

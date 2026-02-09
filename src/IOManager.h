#pragma once

#include "Define.h"
#include "FieldIO.h"
#include "BoundaryIO.h"
#include "Monitor.cuh"
#include "Parallel.h"
#include "SinglePointStat.cuh"

namespace cfd {
__global__ void transfer_counter_to_device(DParameter *param, int count_rey, int count_fav, int count_tkeBudget);

template<MixtureModel mix_model>
struct IOManager {
  Parameter &parameter;
  const Species &species;
  const Mesh &mesh;
  FieldIO<mix_model, OutputTimeChoice::Instance> field_io;
  BoundaryIO<mix_model, OutputTimeChoice::Instance> boundary_io;
  FieldIO<mix_model, OutputTimeChoice::TimeSeries> time_series_io;
  Monitor monitor;
  int if_monitor_points;
  int if_monitor_blocks;
  int monitor_block_frequency;
  int n_block;
  int output_file;
  int output_time_series;
  SinglePointStat stat_collector;
  bool if_collect_statistics;
  int collect_statistics_iter_start;
  int n_rand;

  explicit IOManager(int _myid, const Mesh &_mesh, std::vector<Field> &_field, Parameter &_parameter,
    const Species &spec, int ngg_out);

  void initialize(DParameter *param);

  void manage_output(int step, real physical_time, std::vector<Field> &field, bool finished, DParameter *param, DBoundCond& bound_cond);

  void print_field(int step, const Parameter &parameter, real physical_time = 0);
};

template<MixtureModel mix_model>
void IOManager<mix_model>::print_field(int step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
  boundary_io.print_boundary();
}

template<MixtureModel mix_model> IOManager<mix_model>::IOManager(int _myid, const Mesh &_mesh,
  std::vector<Field> &_field, Parameter &_parameter, const Species &spec, int ngg_out) :
  parameter(_parameter), species(spec), field_io(_myid, _mesh, _field, _parameter, spec, ngg_out), mesh(_mesh),
  boundary_io(_parameter, _mesh, spec, _field), time_series_io(_myid, _mesh, _field, _parameter, spec, ngg_out),
  monitor(_parameter, _mesh), if_monitor_points{_parameter.get_int("if_monitor_points")},
  if_monitor_blocks{_parameter.get_int("if_monitor_blocks")},
  monitor_block_frequency(_parameter.get_int("monitor_block_frequency")), n_block(_mesh.n_block),
  output_file{_parameter.get_int("output_file")}, output_time_series(_parameter.get_int("output_time_series")),
  stat_collector(parameter, _mesh, _field, spec), if_collect_statistics(_parameter.get_bool("if_collect_statistics")),
  collect_statistics_iter_start(_parameter.get_int("start_collect_statistics_iter")),
  n_rand(_parameter.get_int("random_number_per_point")) {}

template<MixtureModel mix_model> void IOManager<mix_model>::initialize(DParameter *param) {
  field_io.initialize();
  boundary_io.initialize();
  time_series_io.initialize();
  monitor.initialize(parameter, species);
  if (parameter.get_bool("steady") == 0 && parameter.get_bool("if_collect_statistics")) {
    stat_collector.initialize_statistics_collector();
    cudaDeviceSynchronize();
    MpiParallel::barrier();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error in proc %d after initialize_statistics_collector: %s\n", parameter.get_int("myid"),
             cudaGetErrorString(err));
      MpiParallel::exit();
    }
    // transfer the count of steps that have been collected to the parameter.
    transfer_counter_to_device<<<1, 1>>>(param, stat_collector.counter_rey1st[0], stat_collector.counter_fav1st[0],
                                         stat_collector.counter_tke_budget[0]);
  }
}

template<MixtureModel mix_model> void IOManager<mix_model>::manage_output(int step, real physical_time,
  std::vector<Field> &field, bool finished, DParameter *param, DBoundCond& bound_cond) {
  bool update_copy{false};
  if (if_monitor_points) {
    monitor.monitor_point(step, physical_time, field);
  }
  if (if_monitor_blocks && step % monitor_block_frequency == 0) {
    // ReSharper disable CppDFAConstantConditions
    if (!update_copy) {
      // ReSharper restore CppDFAConstantConditions
      for (int b = 0; b < n_block; ++b) {
        field[b].copy_data_from_device(parameter);
      }
      update_copy = true;
    }
    monitor.output_block_monitors(parameter, field, physical_time, step);
  }
  if (if_collect_statistics && step > collect_statistics_iter_start) {
    stat_collector.collect_data(param);
  }
  if (step % output_file == 0 || finished) {
    if (!update_copy) {
      for (int b = 0; b < n_block; ++b) {
        field[b].copy_data_from_device(parameter);
      }
      update_copy = true;
    }
    field_io.print_field(step, physical_time);
    boundary_io.print_boundary();
    if (n_rand > 0)
      write_rng(mesh, parameter, field);
    if (if_collect_statistics && step > collect_statistics_iter_start)
      stat_collector.export_statistical_data();
    // post_process(driver);
    if (if_monitor_points)
      monitor.output_point_monitors();
    // if (parameter.get_bool("sponge_layer")) {
    //   output_sponge_layer(parameter, field, mesh, driver.spec);
    // }
    if (parameter.get_bool("use_df")) {
      bound_cond.write_df(parameter, mesh);
    }
  }
  if (output_time_series > 0 && step % output_time_series == 0) {
    if (!update_copy) {
      for (int b = 0; b < n_block; ++b) {
        field[b].copy_data_from_device(parameter);
      }
      update_copy = true;
    }
    time_series_io.print_field(step, physical_time);
  }
}

template<MixtureModel mix_model>
struct TimeSeriesIOManager {
  FieldIO<mix_model, OutputTimeChoice::TimeSeries> field_io;

  explicit TimeSeriesIOManager(int _myid, const Mesh &_mesh, std::vector<Field> &_field,
    const Parameter &_parameter, const Species &spec, int ngg_out);

  void print_field(int step, const Parameter &parameter, real physical_time);
};

template<MixtureModel mix_model>
TimeSeriesIOManager<mix_model>::TimeSeriesIOManager(int _myid, const Mesh &_mesh,
  std::vector<Field> &_field, const Parameter &_parameter, const Species &spec, int ngg_out) :
  field_io(_myid, _mesh, _field, _parameter, spec, ngg_out)/*, boundary_io(_parameter, _mesh, spec, _field)*/ {}

template<MixtureModel mix_model>
void
TimeSeriesIOManager<mix_model>::print_field(int step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
}
}

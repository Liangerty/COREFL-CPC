#pragma once

#include "Define.h"
#include "FieldIO.h"
#include "BoundaryIO.h"

namespace cfd {
template<MixtureModel mix_model>
struct IOManager {
  FieldIO<mix_model, OutputTimeChoice::Instance> field_io;
  BoundaryIO<mix_model, OutputTimeChoice::Instance> boundary_io;

  explicit IOManager(int _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
                     const Species &spec, int ngg_out);

  void print_field(int step, const Parameter &parameter, real physical_time = 0);
};

template<MixtureModel mix_model>
void IOManager<mix_model>::print_field(int step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
  boundary_io.print_boundary();
}

template<MixtureModel mix_model>
IOManager<mix_model>::IOManager(int _myid, const Mesh &_mesh, std::vector<Field> &_field,
                                      const Parameter &_parameter, const Species &spec, int ngg_out):
    field_io(_myid, _mesh, _field, _parameter, spec, ngg_out), boundary_io(_parameter, _mesh, spec, _field) {

}

template<MixtureModel mix_model>
struct TimeSeriesIOManager {
  FieldIO<mix_model, OutputTimeChoice::TimeSeries> field_io;

  explicit TimeSeriesIOManager(int _myid, const Mesh &_mesh, std::vector<Field> &_field,
                               const Parameter &_parameter,
                               const Species &spec, int ngg_out);

  void print_field(int step, const Parameter &parameter, real physical_time);
};

template<MixtureModel mix_model>
TimeSeriesIOManager<mix_model>::TimeSeriesIOManager(int _myid, const Mesh &_mesh,
                                                                 std::vector<Field> &_field,
                                                                 const Parameter &_parameter, const Species &spec,
                                                                 int ngg_out):
    field_io(_myid, _mesh, _field, _parameter, spec, ngg_out)/*, boundary_io(_parameter, _mesh, spec, _field)*/ {

}

template<MixtureModel mix_model>
void
TimeSeriesIOManager<mix_model>::print_field(int step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
}

}
#pragma once

#include <mpi.h>
#include "../Parameter.h"
#include "../Mesh.h"
#include "../Field.h"

namespace cfd {
struct TkeBudget {
  constexpr static int n_collect = 7;
  static constexpr std::array<std::string_view, n_collect> collect_name = {"Tau11", "Tau12", "Tau13", "Tau22", "Tau23",
                                                                           "Tau33", "TauIjDUiDXj"};
  constexpr static int ngg = 1;
};

MPI_Offset create_tke_budget_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead);

std::vector<int>
read_tke_budget_file(Parameter &parameter, const Mesh &mesh, int n_block_ahead, std::vector<Field> &field);

__device__ void collect_tke_budget(DZone *zone, DParameter *param, int i, int j, int k);

void export_tke_budget_file(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, MPI_Offset offset_ahead,
                            const std::vector<int> &counter, const std::vector<MPI_Datatype> &tys);

}
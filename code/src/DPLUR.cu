#include "DPLUR.cuh"
#include "BoundCond.cuh"

namespace cfd {
__global__ void convert_dq_back_to_dqDt(DZone *zone, const DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt_local = zone->dt_local(i, j, k);
  auto &dq = zone->dq;
  for (int l = 0; l < param->n_var; ++l) {
    dq(i, j, k, l) /= dt_local;
  }
}

__global__ void set_dq_to_0(const DParameter *param, DZone *zone, int i_face) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  int idx[3];
  idx[0] = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);

  int ngg[3]{0, 0, 0};
  if (param->viscous_scheme == 2) {
    for (int l = 0; l < 3; ++l) {
      if (l != b.face) {
        ngg[l] = zone->ngg;
        idx[l] += zone->ngg;
      }
    }
  }
  if (idx[0] > range_end[0] - ngg[0] || idx[1] > range_end[1] - ngg[1] || idx[2] > range_end[2] - ngg[2]) return;

  for (int l = 0; l < param->n_var; ++l) {
    zone->dq(idx[0], idx[1], idx[2], l) = 0;
  }
}

void set_wall_dq_to_0(const Block &block, const DParameter *param, DZone *zone, const DBoundCond &bound_cond,
                      bool ngg_extended) {
  for (size_t l = 0; l < bound_cond.n_wall; l++) {
    const auto nb = bound_cond.wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = bound_cond.wall_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        if (ngg_extended) {
          if (j != hf.face) {
            n_point -= 2 * ngg;
          }
        }
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      set_dq_to_0<<<BPG, TPB>>>(param, zone, i_face);
    }
  }
}
}

#pragma once

#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"
#include "DataCommunication.cuh"

namespace cfd {
template<MixtureModel mix_model>
void compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param, int n_var, const Parameter &parameter) {
  switch (parameter.get_int("inviscid_type")) {
    case 0: // Compute the term with primitive reconstruction methods. (MUSCL/NND/1stOrder + LF/AUSM+/HLLC)
      compute_convective_term_pv<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 3: // Compute the term with WENO-Z-5
      compute_convective_term_weno<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 6:
      compute_convective_term_hybrid_ud_weno<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 2:  // Roe scheme
    default: // Roe scheme
      Roe_compute_inviscid_flux<mix_model>(block, zone, param, n_var, parameter);
      break;
  }
}

//
// template<MixtureModel mix_model>
// void compute_viscous_flux(const Block &block, DZone *zone, DParameter *param, const Parameter &parameter) {
//   const int viscous_order = parameter.get_int("viscous_order");
//
//   if (viscous_order == 0)
//     return;
//
//   const auto mx = block.mx, my = block.my, mz = block.mz;
//   const int dim{mz == 1 ? 2 : 3};
//
//   dim3 tpb = {32, 8, 2};
//   if (dim == 2)
//     tpb = {32, 16, 1};
//   dim3 BPG{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
//
//   if (viscous_order == 2) {
//     auto bpg = dim3(mx/*+1-1*/ / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
//     compute_fv_2nd_order<mix_model><<<bpg, tpb>>>(zone, param);
//     compute_dFv_dx<<<BPG, tpb>>>(zone, param);
//
//     bpg = dim3((mx - 1) / tpb.x + 1, my/*+1-1*/ / tpb.y + 1, (mz - 1) / tpb.z + 1);
//     compute_gv_2nd_order<mix_model><<<bpg, tpb>>>(zone, param);
//     compute_dGv_dy<<<BPG, tpb>>>(zone, param);
//
//     if (dim == 3) {
//       dim3 TPB = {32, 8, 2};
//       bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
//       compute_hv_2nd_order<mix_model><<<bpg, TPB>>>(zone, param);
//
//       compute_dHv_dz<<<BPG, tpb>>>(zone, param);
//     }
//   } else if (viscous_order == 8) {
//     auto bpg = dim3((mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
//     compute_viscous_flux_collocated<mix_model><<<bpg, tpb>>>(zone, param);
//     cudaDeviceSynchronize();
//     // After computing the values on the nodes, we exchange the fv, gv, hv to acquire values on ghost grids.
//     exchange_value(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter, int step,
//   DParameter *param, int task);
//
//     compute_dFv_dx<<<BPG, tpb>>>(zone, param);
//   }
// }
}

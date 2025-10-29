// #include "InviscidScheme.cuh"
// #include "Mesh.h"
// #include "Parameter.h"
// #include "Field.h"
// #include "DParameter.cuh"
// #include "Constants.h"
//
// namespace cfd {
// template<MixtureModel mix_model>
// __global__ void
// __launch_bounds__(64, 8)
// compute_convective_term_weno_x(DZone *zone, DParameter *param) {
//   const int i = static_cast<int>((blockDim.x - 1) * blockIdx.x + threadIdx.x) - 1;
//   const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
//   const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
//   const int max_extent = zone->mx;
//   if (i >= max_extent) return;
//
//   const int tid = static_cast<int>(threadIdx.x);
//   const int block_dim = static_cast<int>(blockDim.x);
//   const auto ngg{zone->ngg};
//   const auto n_var{param->n_var};
//   const int n_point = block_dim + 2 * ngg - 1;
//   const auto n_scalar{param->n_scalar};
//
//   extern __shared__ real s[];
//   real *metric = s;
//   real *jac = &metric[n_point * 3];
//   // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
//   real *pv = &jac[n_point];
//   real *cGradK = &pv[n_point * n_var];
//   real *rhoE = &cGradK[n_point];
//   real *uk = &rhoE[n_point];
//   real *fc = &uk[n_point];
//   real *f_1st = nullptr;
//   if (param->positive_preserving)
//     f_1st = &fc[block_dim * n_var];
//
//   const int i_shared = tid - 1 + ngg;
//   metric[i_shared * 3] = zone->metric(i, j, k)(1, 1);
//   metric[i_shared * 3 + 1] = zone->metric(i, j, k)(1, 2);
//   metric[i_shared * 3 + 2] = zone->metric(i, j, k)(1, 3);
//   jac[i_shared] = zone->jac(i, j, k);
//   for (auto l = 0; l < 5; ++l) {
//     pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
//   }
//   for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//     pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
//   }
//   uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
//                  metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
//                  metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
//   rhoE[i_shared] = zone->cv(i, j, k, 4);
//   if constexpr (mix_model != MixtureModel::Air)
//     cGradK[i_shared] = zone->acoustic_speed(i, j, k);
//   else
//     cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//   cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
//                            metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
//                            metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);
//
//   // ghost cells
//   if (tid < ngg - 1) {
//     const int gi = i - (ngg - 1);
//
//     metric[tid * 3] = zone->metric(gi, j, k)(1, 1);
//     metric[tid * 3 + 1] = zone->metric(gi, j, k)(1, 2);
//     metric[tid * 3 + 2] = zone->metric(gi, j, k)(1, 3);
//     jac[tid] = zone->jac(gi, j, k);
//     for (auto l = 0; l < 5; ++l) {
//       pv[tid * n_var + l] = zone->bv(gi, j, k, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[tid * n_var + 5 + l] = zone->sv(gi, j, k, l);
//     }
//     uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
//               metric[tid * 3 + 1] * pv[tid * n_var + 2] +
//               metric[tid * 3 + 2] * pv[tid * n_var + 3];
//     rhoE[tid] = zone->cv(gi, j, k, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[tid] = zone->acoustic_speed(gi, j, k);
//     else
//       cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
//     cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
//                         metric[tid * 3 + 1] * metric[tid * 3 + 1] +
//                         metric[tid * 3 + 2] * metric[tid * 3 + 2]);
//   }
//   if (tid > block_dim - ngg - 1 || i > max_extent - ngg - 1) {
//     const int iSh = tid + 2 * ngg - 1;
//     const int gi = i + ngg;
//     metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
//     metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
//     metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
//     jac[iSh] = zone->jac(gi, j, k);
//     for (auto l = 0; l < 5; ++l) {
//       pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
//     }
//     uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//               metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//               metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//     rhoE[iSh] = zone->cv(gi, j, k, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[iSh] = zone->acoustic_speed(gi, j, k);
//     else
//       cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
//     cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                         metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                         metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//   }
//   if (i == max_extent - 1 && tid < ngg - 1) {
//     const int n_more_left = ngg - 1 - tid - 1;
//     for (int m = 0; m < n_more_left; ++m) {
//       const int iSh = tid + m + 1;
//       const int gi = i - (ngg - 1 - m - 1);
//
//       metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
//       metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
//       metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
//       jac[iSh] = zone->jac(gi, j, k);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(gi, j, k, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(gi, j, k);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//     const int n_more_right = ngg - 1 - tid;
//     for (int m = 0; m < n_more_right; ++m) {
//       const int iSh = i_shared + m + 1;
//       const int gi = i + (m + 1);
//
//       metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
//       metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
//       metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
//       jac[iSh] = zone->jac(gi, j, k);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(gi, j, k, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(gi, j, k, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(gi, j, k, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(gi, j, k);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//   }
//   __syncthreads();
//
//   // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//   // if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//   hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // } else if (sch == 52 || sch == 72) {
//   // hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // }
//   __syncthreads();
//
//   // if (param->positive_preserving) {
//   //   real dt{0};
//   //   if (param->dt > 0)
//   //     dt = param->dt;
//   //   else
//   //     dt = zone->dt_local(i, j, k);
//   //   positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, i, max_extent, cv, jac);
//   // }
//   // __syncthreads();
//
//   if (tid > 0 && i >= zone->iMin && i <= zone->iMax) {
//     for (int l = 0; l < n_var; ++l) {
//       zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//     }
//   }
// }
//
// template<MixtureModel mix_model>
// __global__ void
// __launch_bounds__(64, 8)
// compute_convective_term_weno_y(DZone *zone, DParameter *param) {
//   const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
//   const int j = static_cast<int>((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
//   const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
//   const int max_extent = zone->my;
//   if (j >= max_extent) return;
//
//   const int block_dim = static_cast<int>(blockDim.y);
//   const auto ngg{zone->ngg};
//   const int n_point = block_dim + 2 * ngg - 1;
//   const auto n_var{param->n_var};
//   const auto n_scalar{param->n_scalar};
//
//   extern __shared__ real s[];
//   real *metric = s;
//   real *jac = &metric[n_point * 3];
//   // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
//   real *pv = &jac[n_point];
//   real *cGradK = &pv[n_point * n_var];
//   real *rhoE = &cGradK[n_point];
//   real *uk = &rhoE[n_point];
//   real *fc = &uk[n_point];
//
//   const int tid = static_cast<int>(threadIdx.y);
//
//   const int i_shared = tid - 1 + ngg;
//   metric[i_shared * 3] = zone->metric(i, j, k)(2, 1);
//   metric[i_shared * 3 + 1] = zone->metric(i, j, k)(2, 2);
//   metric[i_shared * 3 + 2] = zone->metric(i, j, k)(2, 3);
//   jac[i_shared] = zone->jac(i, j, k);
//   for (auto l = 0; l < 5; ++l) {
//     pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
//   }
//   for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//     pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
//   }
//   uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
//                  metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
//                  metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
//   rhoE[i_shared] = zone->cv(i, j, k, 4);
//   if constexpr (mix_model != MixtureModel::Air)
//     cGradK[i_shared] = zone->acoustic_speed(i, j, k);
//   else
//     cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//   cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
//                            metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
//                            metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);
//
//   // ghost cells
//   if (tid < ngg - 1) {
//     const int gj = j - (ngg - 1);
//
//     metric[tid * 3] = zone->metric(i, gj, k)(2, 1);
//     metric[tid * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//     metric[tid * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//     jac[tid] = zone->jac(i, gj, k);
//     for (auto l = 0; l < 5; ++l) {
//       pv[tid * n_var + l] = zone->bv(i, gj, k, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[tid * n_var + 5 + l] = zone->sv(i, gj, k, l);
//     }
//     uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
//               metric[tid * 3 + 1] * pv[tid * n_var + 2] +
//               metric[tid * 3 + 2] * pv[tid * n_var + 3];
//     rhoE[tid] = zone->cv(i, gj, k, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[tid] = zone->acoustic_speed(i, gj, k);
//     else
//       cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//     cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
//                         metric[tid * 3 + 1] * metric[tid * 3 + 1] +
//                         metric[tid * 3 + 2] * metric[tid * 3 + 2]);
//   }
//   if (tid > block_dim - ngg - 1 || j > max_extent - ngg - 1) {
//     const int iSh = tid + 2 * ngg - 1;
//     const int gj = j + ngg;
//     metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//     metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//     metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//     jac[iSh] = zone->jac(i, gj, k);
//     for (auto l = 0; l < 5; ++l) {
//       pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
//     }
//     uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//               metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//               metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//     rhoE[iSh] = zone->cv(i, gj, k, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[iSh] = zone->acoustic_speed(i, gj, k);
//     else
//       cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//     cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                         metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                         metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//   }
//   if (j == max_extent - 1 && tid < ngg - 1) {
//     const int n_more_left = ngg - 1 - tid - 1;
//     for (int m = 0; m < n_more_left; ++m) {
//       const int iSh = tid + m + 1;
//       const int gj = j - (ngg - 1 - m - 1);
//
//       metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//       metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//       metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//       jac[iSh] = zone->jac(i, gj, k);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(i, gj, k, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(i, gj, k);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//     const int n_more_right = ngg - 1 - tid;
//     for (int m = 0; m < n_more_right; ++m) {
//       const int iSh = i_shared + m + 1;
//       const int gj = j + (m + 1);
//
//       metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
//       metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
//       metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
//       jac[iSh] = zone->jac(i, gj, k);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(i, gj, k, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(i, gj, k, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(i, gj, k, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(i, gj, k);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//   }
//   __syncthreads();
//
//   // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//   // if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//   hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // } else if (sch == 52 || sch == 72) {
//   // hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // }
//   __syncthreads();
//
//   // if (param->positive_preserving) {
//   //   real dt{0};
//   //   if (param->dt > 0)
//   //     dt = param->dt;
//   //   else
//   //     dt = zone->dt_local(i, j, k);
//   //   positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, j, max_extent, cv, jac);
//   // }
//   // __syncthreads();
//
//   if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
//     for (int l = 0; l < n_var; ++l) {
//       zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//     }
//   }
// }
//
// template<MixtureModel mix_model>
// __global__ void
// __launch_bounds__(64, 8)
// compute_convective_term_weno_z(DZone *zone, DParameter *param) {
//   const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
//   const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
//   const int k = static_cast<int>((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
//   const int max_extent = zone->mz;
//   if (k >= max_extent) return;
//
//   const int block_dim = static_cast<int>(blockDim.z);
//   const auto ngg{zone->ngg};
//   const int n_point = block_dim + 2 * ngg - 1;
//   const auto n_var{param->n_var};
//   const auto n_scalar{param->n_scalar};
//
//   extern __shared__ real s[];
//   real *metric = s;
//   real *jac = &metric[n_point * 3];
//   // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
//   real *pv = &jac[n_point];
//   real *cGradK = &pv[n_point * n_var];
//   real *rhoE = &cGradK[n_point];
//   real *uk = &rhoE[n_point];
//   real *fc = &uk[n_point];
//
//   const int tid = static_cast<int>(threadIdx.z);
//   const int i_shared = tid - 1 + ngg;
//   metric[i_shared * 3] = zone->metric(i, j, k)(3, 1);
//   metric[i_shared * 3 + 1] = zone->metric(i, j, k)(3, 2);
//   metric[i_shared * 3 + 2] = zone->metric(i, j, k)(3, 3);
//   jac[i_shared] = zone->jac(i, j, k);
//   for (auto l = 0; l < 5; ++l) {
//     pv[i_shared * n_var + l] = zone->bv(i, j, k, l);
//   }
//   for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//     pv[i_shared * n_var + 5 + l] = zone->sv(i, j, k, l);
//   }
//   uk[i_shared] = metric[i_shared * 3] * pv[i_shared * n_var + 1] +
//                  metric[i_shared * 3 + 1] * pv[i_shared * n_var + 2] +
//                  metric[i_shared * 3 + 2] * pv[i_shared * n_var + 3];
//   rhoE[i_shared] = zone->cv(i, j, k, 4);
//   if constexpr (mix_model != MixtureModel::Air)
//     cGradK[i_shared] = zone->acoustic_speed(i, j, k);
//   else
//     cGradK[i_shared] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
//   cGradK[i_shared] *= sqrt(metric[i_shared * 3] * metric[i_shared * 3] +
//                            metric[i_shared * 3 + 1] * metric[i_shared * 3 + 1] +
//                            metric[i_shared * 3 + 2] * metric[i_shared * 3 + 2]);
//
//   // ghost cells
//   if (tid < ngg - 1) {
//     const int gk = k - (ngg - 1);
//
//     metric[tid * 3] = zone->metric(i, j, gk)(3, 1);
//     metric[tid * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//     metric[tid * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//     jac[tid] = zone->jac(i, j, gk);
//     for (auto l = 0; l < 5; ++l) {
//       pv[tid * n_var + l] = zone->bv(i, j, gk, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[tid * n_var + 5 + l] = zone->sv(i, j, gk, l);
//     }
//     uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
//               metric[tid * 3 + 1] * pv[tid * n_var + 2] +
//               metric[tid * 3 + 2] * pv[tid * n_var + 3];
//     rhoE[tid] = zone->cv(i, j, gk, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[tid] = zone->acoustic_speed(i, j, gk);
//     else
//       cGradK[tid] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//     cGradK[tid] *= sqrt(metric[tid * 3] * metric[tid * 3] +
//                         metric[tid * 3 + 1] * metric[tid * 3 + 1] +
//                         metric[tid * 3 + 2] * metric[tid * 3 + 2]);
//   }
//   if (tid > block_dim - ngg - 1 || k > max_extent - ngg - 1) {
//     const int iSh = tid + 2 * ngg - 1;
//     const int gk = k + ngg;
//
//     metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//     metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//     metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//     jac[iSh] = zone->jac(i, j, gk);
//     for (auto l = 0; l < 5; ++l) {
//       pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
//     }
//     for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//       pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
//     }
//     uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//               metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//               metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//     rhoE[iSh] = zone->cv(i, j, gk, 4);
//     if constexpr (mix_model != MixtureModel::Air)
//       cGradK[iSh] = zone->acoustic_speed(i, j, gk);
//     else
//       cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//     cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                         metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                         metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//   }
//   if (k == max_extent - 1 && tid < ngg - 1) {
//     const int n_more_left = ngg - 1 - tid - 1;
//     for (int m = 0; m < n_more_left; ++m) {
//       const int iSh = tid + m + 1;
//       const int gk = k - (ngg - 1 - m - 1);
//
//       metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//       metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//       metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//       jac[iSh] = zone->jac(i, j, gk);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(i, j, gk, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(i, j, gk);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//     const int n_more_right = ngg - 1 - tid;
//     for (int m = 0; m < n_more_right; ++m) {
//       const int iSh = i_shared + m + 1;
//       const int gk = k + (m + 1);
//
//       metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
//       metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
//       metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
//       jac[iSh] = zone->jac(i, j, gk);
//       for (auto l = 0; l < 5; ++l) {
//         pv[iSh * n_var + l] = zone->bv(i, j, gk, l);
//       }
//       for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
//         pv[iSh * n_var + 5 + l] = zone->sv(i, j, gk, l);
//       }
//       uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
//                 metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
//                 metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
//       rhoE[iSh] = zone->cv(i, j, gk, 4);
//       if constexpr (mix_model != MixtureModel::Air)
//         cGradK[iSh] = zone->acoustic_speed(i, j, gk);
//       else
//         cGradK[iSh] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
//       cGradK[iSh] *= sqrt(metric[iSh * 3] * metric[iSh * 3] +
//                           metric[iSh * 3 + 1] * metric[iSh * 3 + 1] +
//                           metric[iSh * 3 + 2] * metric[iSh * 3 + 2]);
//     }
//   }
//   __syncthreads();
//
//   // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
//   // if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
//   hybrid_weno_part_cp(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // } else if (sch == 52 || sch == 72) {
//   // hybrid_weno_part<mix_model>(pv, rhoE, i_shared, param, metric, jac, uk, cGradK, &fc[tid * n_var]);
//   // }
//   __syncthreads();
//
//   // if (param->positive_preserving) {
//   //   real dt{0};
//   //   if (param->dt > 0)
//   //     dt = param->dt;
//   //   else
//   //     dt = zone->dt_local(i, j, k);
//   //   positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, k, max_extent, cv, jac);
//   // }
//   // __syncthreads();
//
//   if (tid > 0 && k >= zone->kMin && k <= zone->kMax) {
//     for (int l = 0; l < n_var; ++l) {
//       zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
//     }
//   }
// }
//
// template<MixtureModel mix_model>
// void compute_convective_term_weno_new(const Block &block, DZone *zone, DParameter *param, int n_var,
//   const Parameter &parameter) {
//   // The implementation of classic WENO.
//   const int extent[3]{block.mx, block.my, block.mz};
//
//   constexpr int block_dim = 64;
//   const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
//   auto shared_mem = (block_dim * n_var                              // fc
//                      + n_computation_per_block * n_var              // pv[5]+sv[n_scalar]
//                      + n_computation_per_block * 7) * sizeof(real); // metric[3]+jacobian+rhoE+Uk+speed_of_sound
//   if (parameter.get_bool("positive_preserving")) {
//     shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
//   }
//
//   dim3 TPB(block_dim, 1, 1);
//   dim3 BPG((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
//   compute_convective_term_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
//
//   TPB = dim3(1, block_dim, 1);
//   BPG = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
//   compute_convective_term_weno_y<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
//
//   if (extent[2] > 1) {
//     TPB = dim3(1, 1, 64);
//     BPG = dim3(extent[0], extent[1], (extent[2] - 1) / (64 - 1) + 1);
//     compute_convective_term_weno_z<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
//   }
// }
//
// template void
// compute_convective_term_weno_new<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param, int n_var,
//   const Parameter &parameter);
//
// template void
// compute_convective_term_weno_new<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
//   int n_var, const Parameter &parameter);
//
// template void
// compute_convective_term_weno_new<MixtureModel::MixtureFraction>(const Block &block, DZone *zone, DParameter *param,
//   int n_var, const Parameter &parameter);
//
// template void
// compute_convective_term_weno_new<MixtureModel::FR>(const Block &block, DZone *zone, DParameter *param, int n_var,
//   const Parameter &parameter);
//
// template void
// compute_convective_term_weno_new<MixtureModel::FL>(const Block &block, DZone *zone, DParameter *param, int n_var,
//   const Parameter &parameter);
// }

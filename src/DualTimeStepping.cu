#include "DualTimeStepping.cuh"
#include "TimeAdvanceFunc.cuh"
#include "Parallel.h"
#include "IOManager.h"
#include "Monitor.cuh"
#include <random>
#include "ViscousScheme.cuh"
#include "FiniteRateChem.cuh"
#include "InviscidScheme.cuh"
#include "Fluctuation.cuh"
#include "DataCommunication.cuh"
#include "DPLUR.cuh"
#include "Residual.cuh"
#include "PostProcess.h"

namespace cfd {
template<MixtureModel mix_model> void dual_time_stepping(Driver<mix_model> &driver) {
  auto &parameter{driver.parameter};
  const real dt = parameter.get_real("dt");
  const real diag_factor = 1.5 / dt;
  if (driver.myid == 0) {
    printf("Unsteady flow simulation with 2rd order dual-time stepping for time advancing.\n");
    printf("The physical time step is %e.\n", dt);
  }

  dim3 tpb{8, 8, 4};
  auto &mesh{driver.mesh};
  const int n_block{mesh.n_block};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (int b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  std::vector<Field> &field{driver.field};
  const int n_var{parameter.get_int("n_var")};
  const int ng_1 = 2 * mesh[0].ngg - 1;
  DParameter *param{driver.param};
  for (auto b = 0; b < n_block; ++b) {
    // Store the initial value of the flow field
    store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
    // Compute the conservative variables from basic variables
    // In unsteady simulations, because of the upwind high-order method to be used;
    // we need the conservative variables.
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);

    // Initialize qn1. This should be a condition if we read from previous or initialize with the current cv.
    // We currently initialize with the cv.
    cudaMemcpy(field[b].h_ptr->qn1.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
               cudaMemcpyDeviceToDevice);
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }

  // IOManager<mix_model> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  // TimeSeriesIOManager<mix_model> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  const int output_time_series = parameter.get_int("output_time_series");
  auto& io_manager = driver.io_manager;

  // Monitor monitor(parameter, driver.spec, mesh);
  const int if_monitor_points{parameter.get_int("if_monitor_points")};
  const int if_monitor_blocks{parameter.get_int("if_monitor_blocks")};

  int step{parameter.get_int("step")};
  const int total_step{parameter.get_int("total_step") + step};
  const int output_screen = parameter.get_int("output_screen");
  const real total_simulation_time{parameter.get_real("total_simulation_time")};
  const int output_file = parameter.get_int("output_file");
  int inner_iteration = parameter.get_int("inner_iteration");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  const bool if_collect_statistics{parameter.get_bool("if_collect_statistics")};
  const int collect_statistics_iter_start{parameter.get_int("start_collect_statistics_iter")};
  // auto &stat_collector{driver.stat_collector};

  bool monitor_inner_iteration;
  std::array<real, 4> res_scale_inner{1, 1, 1, 1};
  const int iteration_adjust_step{parameter.get_int("iteration_adjust_step")};
  const auto need_physical_time{parameter.get_bool("need_physical_time")};
  const auto hybrid_inviscid_scheme{parameter.get_string("hybrid_inviscid_scheme")};

  const int n_rand = parameter.get_int("random_number_per_point");
  const int fluctuation_form = parameter.get_int("fluctuation_form");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<real> dist(-1, 1);

  while (!finished) {
    ++step;

    if (need_physical_time) {
      update_physical_time<<<1, 1>>>(param, physical_time);
    }

    // Start a single iteration
    // First, store the value of last step if we need to compute residual
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    // For every time step, we need to compute the qn_star and replace the qn1 with the current cv.
    for (auto b = 0; b < n_block; ++b) {
      compute_qn_star<<<bpg[b], tpb>>>(field[b].d_ptr, n_var, dt);
      cudaMemcpy(field[b].h_ptr->qn1.data(), field[b].h_ptr->cv.data(),
                 field[b].h_ptr->cv.size() * sizeof(real) * n_var, cudaMemcpyDeviceToDevice);
    }

    if (step % iteration_adjust_step == 0) {
      monitor_inner_iteration = true;
    } else {
      monitor_inner_iteration = false;
    }

    // Calculate the shock sensor, and save the results to zone->shock_sensor(i,j,k).
    if (hybrid_inviscid_scheme != "NO") {
      for (auto b = 0; b < n_block; ++b) {
        compute_shock_sensor<<<bpg[b], tpb>>>(field[b].d_ptr, param);
      }
    }

    // Compute the fluctuation values. This is updated every step, not every sub-iter in RK
    if (fluctuation_form > 0) {
      for (int b = 0; b < n_block; ++b) {
        compute_fluctuation(mesh[b], field[b].d_ptr, param, fluctuation_form, parameter, dist, gen);
      }
    }

    // dual-time stepping inner iteration
    for (int iter = 1; iter <= inner_iteration; ++iter) {
      for (auto b = 0; b < n_block; ++b) {
        if (monitor_inner_iteration) {
          store_last_iter<<<bpg[b], tpb>>>(field[b].d_ptr);
        }

        // Set dq to 0
        cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));
      }

      compute_viscous_flux<mix_model>(mesh, field, param, parameter);

      for (auto b = 0; b < n_block; ++b) {
        // Second, for each block, compute the residual dq
        // First, compute the source term, because properties such as mut are updated here.
        if (parameter.get_int("reaction") == 1) {
          finite_rate_chemistry<<<bpg[b], tpb>>>(field[b].d_ptr, param);
        }
        compute_inviscid_flux<mix_model>(mesh[b], field[b].d_ptr, param, n_var, parameter);
        // compute_viscous_flux<mix_model>(mesh[b], field[b].d_ptr, param, parameter);

        // compute the local time step
        if constexpr (mix_model != MixtureModel::Air) {
          local_time_step_without_reaction<<<bpg[b], tpb>>>(field[b].d_ptr, param);
        } else {
          local_time_step<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);
        }

        // Implicit treat
        dual_time_stepping_implicit_treat<mix_model>(mesh[b], param, field[b].d_ptr, field[b].h_ptr, parameter,
                                                     driver.bound_cond, diag_factor);

        // update basic and conservative variables
        update_cv_and_bv<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

        if (parameter.get_bool("limit_flow"))
          limit_flow<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

        // Apply boundary conditions
        // Attention: "driver" is a template class, when a template class calls a member function of another template,
        // the compiler will not treat the called function as a template function,
        // so we need to explicitly specify the "template" keyword here.
        // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
        driver.bound_cond.template apply_boundary_conditions<mix_model>(mesh[b], field[b], param, 0);

        update_values_with_fluctuations<<<bpg[b], tpb>>>(field[b].d_ptr, param);
      }
      // Third, transfer data between and within processes
      data_communication<mix_model>(mesh, field, parameter, step, param);

      if (mesh.dimension == 2) {
        for (auto b = 0; b < n_block; ++b) {
          const auto mx{mesh[b].mx}, my{mesh[b].my};
          dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
          eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
        }
      }

      // update physical properties such as Mach number, transport coefficients et, al.
      for (auto b = 0; b < n_block; ++b) {
        const int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
        update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
      }

      if (monitor_inner_iteration)
        if (inner_converged(mesh, field, parameter, iter, res_scale_inner, driver.myid, step, inner_iteration))
          break;
    }

    // Finally, test if the simulation reaches convergence state
    physical_time += dt;
    parameter.update_parameter("solution_time", physical_time);
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
      if (driver.myid == 0) {
        unsteady_screen_output(step, err_max, driver.time, driver.res, dt, physical_time);
      }
    }
    cudaDeviceSynchronize();
    if (physical_time > total_simulation_time || step == total_step) {
      if (physical_time > total_simulation_time)
        printf("The simulated physical time is %e, which is larger than required physical time %e.\n", physical_time,
               total_simulation_time);
      else if (step == total_step)
        printf("The step %d reaches specified total step.\n", step);
      finished = true;
    }
    io_manager.manage_output(step, physical_time, field, finished, param, driver.bound_cond);
    // bool update_copy{false};
    // if (if_monitor_points) {
    //   monitor.monitor_point(step, physical_time, field);
    // }
    // if (if_monitor_blocks && step % parameter.get_int("monitor_block_frequency") == 0) {
    //   if (!update_copy) {
    //     for (int b = 0; b < n_block; ++b) {
    //       field[b].copy_data_from_device(parameter);
    //     }
    //     update_copy = true;
    //   }
    //   monitor.output_block_monitors(parameter, field, physical_time, step);
    // }
    // if (if_collect_statistics && step > collect_statistics_iter_start) {
    //   //      stat_collector.template collect_data<mix_model>(param);
    //   stat_collector.collect_data(param);
    // }
    // if (step % output_file == 0 || finished) {
    //   if (!update_copy) {
    //     for (int b = 0; b < n_block; ++b) {
    //       field[b].copy_data_from_device(parameter);
    //     }
    //     update_copy = true;
    //   }
    //   ioManager.print_field(step, parameter, physical_time);
    //   if (n_rand > 0)
    //     write_rng(mesh, parameter, field);
    //   if (if_collect_statistics && step > collect_statistics_iter_start)
    //     stat_collector.export_statistical_data();
    //   post_process(driver);
    //   if (if_monitor_points)
    //     monitor.output_point_monitors();
    // }
    // if (output_time_series > 0 && step % output_time_series == 0) {
    //   if (!update_copy) {
    //     for (int b = 0; b < n_block; ++b) {
    //       field[b].copy_data_from_device(parameter);
    //     }
    //     update_copy = true;
    //   }
    //   timeSeriesIOManager.print_field(step, parameter, physical_time);
    // }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      MpiParallel::exit();
    }
  }
  delete[] bpg;
  io_manager.monitor.stop_recording_blocks(parameter);
}

__global__ void compute_qn_star(DZone *zone, int n_var, real dt_global) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real factor = 0.5 * zone->jac(i, j, k) / dt_global;
  for (int l = 0; l < n_var; ++l) {
    zone->qn_star(i, j, k, l) = factor * (4 * zone->cv(i, j, k, l) - zone->qn1(i, j, k, l));
  }
}

template<MixtureModel mixture_model> void dual_time_stepping_implicit_treat(const Block &block, DParameter *param,
  DZone *d_ptr, DZone *h_ptr, Parameter &parameter, DBoundCond &bound_cond, real diag_factor) {
  const int extent[3]{block.mx, block.my, block.mz};
  const int dim{extent[2] == 1 ? 2 : 3};
  dim3 tpb{8, 8, 4};
  if (dim == 2) {
    tpb = {16, 16, 1};
  }
  const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
  compute_modified_rhs<<<bpg, tpb>>>(d_ptr, parameter.get_int("n_var"), parameter.get_real("dt"));

  DPLUR<mixture_model>(block, param, d_ptr, h_ptr, parameter, bound_cond, diag_factor);
}

__global__ void compute_modified_rhs(DZone *zone, int n_var, real dt_global) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &dq = zone->dq;
  auto &q_star = zone->qn_star;
  const real factor = 1.5 * zone->jac(i, j, k) / dt_global;
  for (int l = 0; l < n_var; ++l) {
    dq(i, j, k, l) += q_star(i, j, k, l) - factor * zone->cv(i, j, k, l);
  }
}

bool inner_converged(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter, int iter,
  std::array<real, 4> &res_scale, int myid, int step, int &inner_iter) {
  const int n_block = mesh.n_block;

  std::array<real, 4> res{0, 0, 0, 0};
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  for (int b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    // compute the square of the difference of the basic variables
    compute_square_of_dbv_wrt_last_inner_iter<<<bpg, tpb>>>(field[b].d_ptr);
  }
  constexpr int TPB{128};
  constexpr int n_res_var{4};
  real res_block[n_res_var];
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
                                                TPB * sizeof(real) * n_res_var);
  for (int b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const int size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB, TPB * sizeof(real) * n_res_var>>>(
      field[b].h_ptr->in_last_step.data(), size);
    reduction_of_dv_squared<n_res_var> <<<1, TPB, TPB * sizeof(real) * n_res_var>>>(
      field[b].h_ptr->in_last_step.data(), n_blocks);
    cudaMemcpy(res_block, field[b].h_ptr->in_last_step.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
    for (int l = 0; l < n_res_var; ++l) {
      res[l] += res_block[l];
    }
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    static std::array<double, 4> res_temp;
    for (int i = 0; i < 4; ++i) {
      res_temp[i] = res[i];
    }
    MPI_Allreduce(res_temp.data(), res.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  for (auto &e: res) {
    e = std::sqrt(e / mesh.n_grid_total);
  }

  if (iter == 1) {
    for (int i = 0; i < n_res_var; ++i) {
      res_scale[i] = res[i];
      if (res_scale[i] < 1e-20) {
        res_scale[i] = 1e-20;
      }
    }
    if (myid == 0) {
      printf("******************************\nStep %d.\n", step);
    }
  }

  for (int i = 0; i < 4; ++i) {
    res[i] /= res_scale[i];
  }

  // Find the maximum error of the 4 errors
  real err_max = res[0];
  for (int i = 1; i < 4; ++i) {
    if (std::abs(res[i]) > err_max) {
      err_max = res[i];
    }
  }

  if (myid == 0) {
    if (isnan(err_max)) {
      printf("Nan occurred in iter %d of step %d. Stop simulation.\n", iter, step);
      MpiParallel::exit();
    }
    if (iter % 5 == 0 || err_max < 1e-3)
      printf("iter %d, err_max=%e\n", iter, err_max);
  }

  if (iter == inner_iter) {
    const int INNER_ITER_MAX = parameter.get_int("max_inner_iteration");
    if (err_max > 1e-3 && inner_iter < INNER_ITER_MAX) {
      inner_iter += 5;
      inner_iter = std::min(inner_iter, INNER_ITER_MAX);
      if (myid == 0) {
        printf("Inner iteration step is increased to %d.\n", inner_iter);
      }
    }
  }

  if (err_max < 1e-3) {
    if (const auto minus = inner_iter - iter; minus > 0) {
      inner_iter -= minus;
      if (myid == 0) {
        printf("Inner iteration step is decreased to %d.\n", inner_iter);
      }
    }
    return true;
  }

  return false;
}

__global__ void compute_square_of_dbv_wrt_last_inner_iter(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->in_last_step;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  const real vel = sqrt(
    bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
  bv_last(i, j, k, 1) = (vel - bv_last(i, j, k, 1)) * (vel - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

__global__ void store_last_iter(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  zone->in_last_step(i, j, k, 0) = zone->bv(i, j, k, 0);
  zone->in_last_step(i, j, k, 1) = sqrt(
    zone->bv(i, j, k, 1) * zone->bv(i, j, k, 1) + zone->bv(i, j, k, 2) * zone->bv(i, j, k, 2) +
    zone->bv(i, j, k, 3) * zone->bv(i, j, k, 3));
  zone->in_last_step(i, j, k, 2) = zone->bv(i, j, k, 4);
  zone->in_last_step(i, j, k, 3) = zone->bv(i, j, k, 5);
}

template void dual_time_stepping<MixtureModel::Air>(Driver<MixtureModel::Air> &driver);

template void dual_time_stepping<MixtureModel::Mixture>(Driver<MixtureModel::Mixture> &driver);
}

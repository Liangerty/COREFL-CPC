#include "StrangSplitting.cuh"
#include "FiniteRateChem.cuh"
#include "Residual.cuh"
#include "PostProcess.h"

namespace cfd {
template<MixtureModel mix_model>
void strang_splitting(Driver<mix_model> &driver) {
  if (driver.myid == 0) {
    printf("\n****************************Time advancement starts*****************************\n");
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
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }

  auto &parameter{driver.parameter};
  // auto &ps{driver.ps};

  IOManager<mix_model> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  const int output_time_series = parameter.get_int("output_time_series");

  Monitor monitor(parameter, driver.spec, mesh);
  const int if_monitor_points{parameter.get_int("if_monitor_points")};
  const int if_monitor_blocks{parameter.get_int("if_monitor_blocks")};

  int step{parameter.get_int("step")};
  const int total_step{parameter.get_int("total_step") + step};
  const int output_screen = parameter.get_int("output_screen");
  const real total_simulation_time{parameter.get_real("total_simulation_time")};
  const int output_file = parameter.get_int("output_file");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  real dt{1e+6};
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};
  if (fixed_time_step) {
    dt = parameter.get_real("dt");
  }

  const bool if_collect_statistics{parameter.get_bool("if_collect_statistics")};
  const int collect_statistics_iter_start{parameter.get_int("start_collect_statistics_iter")};
  auto &statistics_collector{driver.stat_collector};
  const int n_rand = parameter.get_int("random_number_per_point");
  const int fluctuation_form = parameter.get_int("fluctuation_form");

  const auto need_physical_time{parameter.get_bool("need_physical_time")};
  const auto have_sponge_layer{parameter.get_bool("sponge_layer")};
  const auto hybrid_inviscid_scheme{parameter.get_string("hybrid_inviscid_scheme")};
  const int monitor_block_frequency = parameter.get_int("monitor_block_frequency");

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

    if (!fixed_time_step) {
      // compute the local time step
      for (auto b = 0; b < n_block; ++b)
        local_time_step_without_reaction<<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // After all processes and all blocks computing dt_local, we compute the global time step.
      dt = global_time_step(mesh, parameter, field);

      update_dt_global<<<1, 1>>>(param, dt);
    }

    // Compute the fluctuation values. This is updated every step, not every sub-iter in RK
    if (fluctuation_form > 0) {
      for (int b = 0; b < n_block; ++b) {
        compute_fluctuation(mesh[b], field[b].d_ptr, param, fluctuation_form, parameter, dist, gen);
      }
    }

    RK_subStep<mix_model>(field, param, parameter, mesh, driver.bound_cond, n_block, step, tpb, bpg, 0.5 * dt);
    reaction_subStep<mix_model>(field, param, parameter, mesh, n_block, tpb, bpg, dt);
    RK_subStep<mix_model>(field, param, parameter, mesh, driver.bound_cond, n_block, step, tpb, bpg, 0.5 * dt);

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
      monitor.output_block_monitors(parameter, field, physical_time);
    }
    if (if_collect_statistics && step > collect_statistics_iter_start) {
      statistics_collector.collect_data(param);
    }
    if (step % output_file == 0 || finished) {
      if (!update_copy) {
        for (int b = 0; b < n_block; ++b) {
          field[b].copy_data_from_device(parameter);
        }
        update_copy = true;
      }
      ioManager.print_field(step, parameter, physical_time);
      if (n_rand > 0)
        write_rng(mesh, parameter, field);
      if (if_collect_statistics && step > collect_statistics_iter_start)
        statistics_collector.export_statistical_data();
      post_process(driver);
      if (if_monitor_points)
        monitor.output_point_monitors();
      if (parameter.get_bool("sponge_layer")) {
        output_sponge_layer(parameter, field, mesh, driver.spec);
      }
      if (parameter.get_bool("use_df")) {
        driver.bound_cond.write_df(parameter, mesh);
      }
    }
    if (output_time_series > 0 && step % output_time_series == 0) {
      if (!update_copy) {
        for (int b = 0; b < n_block; ++b) {
          field[b].copy_data_from_device(parameter);
        }
        update_copy = true;
      }
      timeSeriesIOManager.print_field(step, parameter, physical_time);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Process %d, Error: %s\n", driver.myid, cudaGetErrorString(err));
      MpiParallel::exit();
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model>
void RK_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  DBoundCond &bound_cond, int n_block, int step, dim3 tpb, dim3 *bpg, real dt) {
  // For every iteration, we need to save the conservative variables of the last step.
  const int n_var{parameter.get_int("n_var")};
  for (auto b = 0; b < n_block; ++b) {
    cudaMemcpy(field[b].h_ptr->qn.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
               cudaMemcpyDeviceToDevice);
  }

  // Calculate the shock sensor, and save the results to zone->shock_sensor(i,j,k).
  if (parameter.get_string("hybrid_inviscid_scheme") != "NO") {
    for (auto b = 0; b < n_block; ++b) {
      compute_shock_sensor<<<bpg[b], tpb>>>(field[b].d_ptr, param);
    }
  }

  // Rk inner iteration
  for (int rk = 0; rk < 3; ++rk) {
    for (auto b = 0; b < n_block; ++b) {
      // Set dq to 0
      cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));
    }

    compute_viscous_flux<mix_model>(mesh, field, param, parameter);

    for (auto b = 0; b < n_block; ++b) {
      // Second, for each block, compute the residual dq
      compute_inviscid_flux<mix_model>(mesh[b], field[b].d_ptr, param, n_var, parameter);
      // compute_viscous_flux<mix_model>(mesh[b], field[b].d_ptr, param, parameter);

      // Explicit temporal schemes should not use any implicit treatment.

      // Apply the non-reflecting boundary condition
      bound_cond.nonReflectingBoundary<mix_model>(mesh[b], field[b], param);

      // update basic variables
      update_cv_and_bv_rk<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt, rk);

      // limit unphysical values computed by the program
      if (parameter.get_bool("limit_flow"))
        limit_flow<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // Apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      bound_cond.apply_boundary_conditions<mix_model>(mesh[b], field[b], param, step);

      update_values_with_fluctuations<<<bpg[b], tpb>>>(field[b].d_ptr, param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model>(mesh, field, parameter, step, param);

    const int ng_1 = 2 * mesh[0].ngg - 1;
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
  }
}

__device__ void finite_rate_chemistry(const DParameter *param, real dt, real *rhoY, real t, real &timeScale) {
  const int ns = param->n_spec;
  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const auto imw = param->imw;
  for (int l = 0; l < ns; ++l) {
    c[l] = rhoY[l] * imw[l] * 1e-3;
  }

  // compute the forward reaction rate
  real kf[MAX_REAC_NUMBER];
  forward_reaction_rate_1(t, kf, c);

  real kb[MAX_REAC_NUMBER] = {};
  backward_reaction_rate_1(t, kf, kb, param);

  // compute the rate of progress
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  rate_of_progress_1(kf, kb, c, q, q1, q2);

  // compute the chemical source
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];
  chemical_source_1(q1, q2, omega_d, omega);

  // TODO: only the last substep needs to update the timescale!
  real time_scale{1e+6};
  for (int l = 0; l < ns; ++l) {
    if (rhoY[l] > 1e-25 && abs(omega_d[l]) > 1e-25)
      time_scale = min(time_scale, abs(rhoY[l] / omega_d[l]));
    rhoY[l] += omega[l] * dt;
  }
  timeScale = time_scale;
}

__device__ double2 update_temperature(const DParameter *param, const real *rhoY, real rhoE, real t) {
  // compute the new total energy and temperature, pressure
  // The constant volume reactor should not change the total energy?
  const int ns = param->n_spec;

  real err{1};
  constexpr int max_iter{100};
  constexpr real eps{1e-3};
  int iter = 0;

  real h_i[MAX_SPEC_NUMBER], cp_i[MAX_SPEC_NUMBER];
  real rhoR{0};
  for (int l = 0; l < ns; ++l) {
    rhoR += rhoY[l] * param->gas_const[l];
  }
  while (err > eps && iter++ < max_iter) {
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real rhoCp_tot{0}, rhoh{0};
    for (int l = 0; l < ns; ++l) {
      rhoCp_tot += cp_i[l] * rhoY[l];
      rhoh += h_i[l] * rhoY[l];
    }
    const real rhoE_t = rhoh - rhoR * t;
    const real rhoCv = rhoCp_tot - rhoR;
    const real t1 = t - (rhoE_t - rhoE) / rhoCv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  double2 tp;
  tp.x = t;
  tp.y = rhoR * t;
  return tp;
}

__device__ double2 zeroDReaction(const DParameter *param, real dt, real *rhoY, real rhoE, real t, real &timeScale) {
  finite_rate_chemistry(param, dt, rhoY, t, timeScale);

  const auto tp = update_temperature(param, rhoY, rhoE, t);

  return tp;
}

__global__ void reaction_step(DZone *zone, const DParameter *param, real dt) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;
  auto &bv = zone->bv;
  real rhoY[MAX_SPEC_NUMBER];
  for (int l = 0; l < param->n_spec; ++l) {
    rhoY[l] = cv(i, j, k, l + 5);
  }
  real T{bv(i, j, k, 5)}, p{bv(i, j, k, 4)};
  const real rhoE = cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * (
                      bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
                      bv(i, j, k, 3) * bv(i, j, k, 3));

  auto reaction_timescale = zone->reaction_timeScale(i, j, k);
  if (reaction_timescale < 1e-15)
    reaction_timescale = 0.2 * dt;
  const int n_sub = static_cast<int>(dt / reaction_timescale);
  real dt_reaction = dt / (n_sub + 1);

  // The first n_sub steps are advanced with a fixed dt
  auto &timeScale = zone->reaction_timeScale(i, j, k);
  for (int n = 0; n < n_sub; ++n) {
    // if (print) printf("step %d\n", n);
    const auto [x, y] = zeroDReaction(param, dt_reaction, rhoY, rhoE, T, timeScale);
    T = x, p = y;
  }
  // To make the time interval consistent with dt, the last step is advanced with dt_flow - n_sub * dt_reaction
  dt_reaction = dt - dt_reaction * n_sub;
  if (abs(dt_reaction) > 1e-15) {
    const auto [x, y] = zeroDReaction(param, dt_reaction, rhoY, rhoE, T, timeScale);
    T = x, p = y;
  }

  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = T;
  const real iRho = 1.0 / cv(i, j, k, 0);
  auto &sv = zone->sv;
  for (int l = 0; l < param->n_spec; ++l) {
    cv(i, j, k, l + 5) = rhoY[l];
    sv(i, j, k, l) = rhoY[l] * iRho;
  }
}

template<MixtureModel mix_model>
void reaction_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  int n_block, dim3 tpb, dim3 *bpg, real dt) {
  for (auto b = 0; b < n_block; ++b) {
    if (parameter.get_int("reaction") == 1) {
      reaction_step<<<bpg[b], tpb>>>(field[b].d_ptr, param, dt);
    }
  }
  const int ng_1 = 2 * mesh[0].ngg - 1;
  if (mesh.dimension == 2) {
    for (auto b = 0; b < n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
    }
  }
  for (auto b = 0; b < n_block; ++b) {
    const int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
  }
}

template<MixtureModel mix_model> void compute_cn(std::vector<Field> &field, DParameter *param, Parameter &parameter,
  const Mesh &mesh, int n_block, dim3 tpb, dim3 *bpg) {
  const int n_var{parameter.get_int("n_var")};

  // Calculate the shock sensor, and save the results to zone->shock_sensor(i,j,k).
  if (parameter.get_string("hybrid_inviscid_scheme") != "NO") {
    for (auto b = 0; b < n_block; ++b) {
      compute_shock_sensor<<<bpg[b], tpb>>>(field[b].d_ptr, param);
    }
  }

  for (auto b = 0; b < n_block; ++b) {
    // Set dq to 0
    cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));
  }

  compute_viscous_flux<mix_model>(mesh, field, param, parameter);

  for (auto b = 0; b < n_block; ++b) {
    // Second, for each block, compute the residual dq
    compute_inviscid_flux<mix_model>(mesh[b], field[b].d_ptr, param, n_var, parameter);

    // copy the data of dq, which contains only the transport part of the equation set, to Cn.
    cudaMemcpy(field[b].h_ptr->Cn.data(), field[b].h_ptr->dq.data(), field[b].h_ptr->dq.size() * sizeof(real) * n_var,
               cudaMemcpyDeviceToDevice);
  }
}

__device__ void wu_finite_rate_chemistry(const DParameter *param, DZone *zone, real dt, const real *cn, const real *cnY,
  real &timeScale, int i, int j, int k) {
  const int ns = param->n_spec;
  auto &cv = zone->cv;
  auto &bv = zone->bv;
  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const auto imw = param->imw;
  for (int l = 0; l < ns; ++l) {
    c[l] = cv(i, j, k, l + 5) * imw[l] * 1e-3;
  }

  // compute the forward reaction rate
  const real t = bv(i, j, k, 5);
  real kf[MAX_REAC_NUMBER];
  forward_reaction_rate_1(t, kf, c);

  real kb[MAX_REAC_NUMBER] = {};
  backward_reaction_rate_1(t, kf, kb, param);

  // compute the rate of progress
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  rate_of_progress_1(kf, kb, c, q, q1, q2);

  // compute the chemical source
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];
  chemical_source_1(q1, q2, omega_d, omega);

  // update the conservative variables with chemical source and Cn (constant).
  const real iJac = 1.0 / zone->jac(i, j, k);
  cv(i, j, k, 0) += cn[0] * dt * iJac;
  cv(i, j, k, 1) += cn[1] * dt * iJac;
  cv(i, j, k, 2) += cn[2] * dt * iJac;
  cv(i, j, k, 3) += cn[3] * dt * iJac;
  cv(i, j, k, 4) += cn[4] * dt * iJac;
  real time_scale{1e+6};
  for (int l = 0; l < ns; ++l) {
    if (cv(i, j, k, 5 + l) > 1e-25 && abs(omega_d[l]) > 1e-25)
      time_scale = min(time_scale, abs(cv(i, j, k, 5 + l) / omega_d[l]));
    cv(i, j, k, 5 + l) += (omega[l] + cnY[l] * iJac) * dt;
  }
  timeScale = time_scale;

  // update the basic variables
  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;
  const auto V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  //V^2

  auto &sv = zone->sv;
  real sum_part_den{0};
  for (int l = 0; l < param->n_spec; ++l) {
    if (cv(i, j, k, 5 + l) < 0) {
      // If the species mass fraction is negative, we set it to zero.
      cv(i, j, k, 5 + l) = 0;
    } else
      sum_part_den += cv(i, j, k, l + 5);
  }
  sum_part_den = 1.0 / sum_part_den;
  const auto denComDivDenReal = cv(i, j, k, 0) * sum_part_den;
  for (int l = 0; l < param->n_spec; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * sum_part_den;
    cv(i, j, k, 5 + l) *= denComDivDenReal;
  }
  // For multiple species or RANS methods, there will be scalars to be computed
  for (int l = param->n_spec; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
  }
  compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
}

__device__ void wu_zeroDReaction(const DParameter *param, real dt, DZone *zone, const real *cn,
  const real *cnY, real &timeScale, int i, int j, int k) {
  const int ns = param->n_spec;
  auto &cv = zone->cv;
  auto &bv = zone->bv;
  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const auto imw = param->imw;
  for (int l = 0; l < ns; ++l) {
    c[l] = cv(i, j, k, l + 5) * imw[l] * 1e-3;
  }
  const real t = bv(i, j, k, 5);
  real kf[MAX_REAC_NUMBER];
  real kb[MAX_REAC_NUMBER] = {};
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  real omega[MAX_SPEC_NUMBER * 2];
  real *omega_d = &omega[MAX_SPEC_NUMBER];

  const int hardCodedMech = param->hardCodedMech;

  if (hardCodedMech == 1) {
    // compute the forward reaction rate
    forward_reaction_rate_1(t, kf, c);

    backward_reaction_rate_1(t, kf, kb, param);

    // compute the rate of progress
    rate_of_progress_1(kf, kb, c, q, q1, q2);

    // compute the chemical source
    chemical_source_1(q1, q2, omega_d, omega);
  } else {
    // generic mechanisms
    forward_reaction_rate(t, kf, c, param);

    backward_reaction_rate(t, kf, c, param, kb);

    // compute the rate of progress
    rate_of_progress(kf, kb, c, q, q1, q2, param);

    // compute the chemical source
    chemical_source(q1, q2, omega_d, omega, param);
  }

  // update the conservative variables with chemical source and Cn (constant).
  const real iJac = 1.0 / zone->jac(i, j, k);
  cv(i, j, k, 0) += cn[0] * dt * iJac;
  cv(i, j, k, 1) += cn[1] * dt * iJac;
  cv(i, j, k, 2) += cn[2] * dt * iJac;
  cv(i, j, k, 3) += cn[3] * dt * iJac;
  cv(i, j, k, 4) += cn[4] * dt * iJac;
  real time_scale{1e+6};
  for (int l = 0; l < ns; ++l) {
    if (cv(i, j, k, 5 + l) > 1e-25 && abs(omega_d[l]) > 1e-25)
      time_scale = min(time_scale, abs(cv(i, j, k, 5 + l) / omega_d[l]));
    cv(i, j, k, 5 + l) += (omega[l] + cnY[l] * iJac) * dt;
  }
  timeScale = time_scale;

  // update the basic variables
  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;

  auto &sv = zone->sv;
  real sum_part_den{0};
  for (int l = 0; l < param->n_spec; ++l) {
    if (cv(i, j, k, 5 + l) < 0) {
      // If the species mass fraction is negative, we set it to zero.
      cv(i, j, k, 5 + l) = 0;
    } else
      sum_part_den += cv(i, j, k, l + 5);
  }
  sum_part_den = 1.0 / sum_part_den;
  const auto denComDivDenReal = cv(i, j, k, 0) * sum_part_den;
  for (int l = 0; l < param->n_spec; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * sum_part_den;
    cv(i, j, k, 5 + l) *= denComDivDenReal;
  }
  // For multiple species or RANS methods, there will be scalars to be computed
  for (int l = param->n_spec; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
  }
  compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
}

__global__ void wu_reaction_step(DZone *zone, const DParameter *param, real dt) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cn = zone->Cn;
  real CnY[MAX_SPEC_NUMBER], Cn[5];
  Cn[0] = cn(i, j, k, 0);
  Cn[1] = cn(i, j, k, 1);
  Cn[2] = cn(i, j, k, 2);
  Cn[3] = cn(i, j, k, 3);
  Cn[4] = cn(i, j, k, 4);
  for (int l = 0; l < param->n_spec; ++l) {
    CnY[l] = cn(i, j, k, l + 5);
  }

  auto reaction_timescale = zone->reaction_timeScale(i, j, k);
  if (reaction_timescale < 1e-15)
    reaction_timescale = 0.2 * dt;
  const int n_sub = static_cast<int>(dt / reaction_timescale);
  real dt_reaction = dt / (n_sub + 1);

  // The first n_sub steps are advanced with a fixed dt
  auto &timeScale = zone->reaction_timeScale(i, j, k);
  for (int n = 0; n < n_sub; ++n) {
    wu_zeroDReaction(param, dt_reaction, zone, Cn, CnY, timeScale, i, j, k);
  }
  // To make the time interval consistent with dt, the last step is advanced with dt_flow - n_sub * dt_reaction
  dt_reaction = dt - dt_reaction * n_sub;
  if (abs(dt_reaction) > 1e-15) {
    wu_zeroDReaction(param, dt_reaction, zone, Cn, CnY, timeScale, i, j, k);
  }
}

template<MixtureModel mix_model> void wu_reaction_subStep(std::vector<Field> &field, DParameter *param,
  Parameter &parameter, const Mesh &mesh, DBoundCond &bound_cond, int n_block, dim3 tpb, dim3 *bpg, real dt, int step) {
  for (auto b = 0; b < n_block; ++b) {
    if (parameter.get_int("reaction") == 1) {
      wu_reaction_step<<<bpg[b], tpb>>>(field[b].d_ptr, param, dt);
    }
    bound_cond.apply_boundary_conditions<mix_model>(mesh[b], field[b], param, step);
  }
  data_communication<mix_model>(mesh, field, parameter, step, param);
  const int ng_1 = 2 * mesh[0].ngg - 1;
  if (mesh.dimension == 2) {
    for (auto b = 0; b < n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
    }
  }
  for (auto b = 0; b < n_block; ++b) {
    const int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
  }
}

template<MixtureModel mix_model>
__global__ void update_cv_and_bv_rk_wu(DZone *zone, DParameter *param, real dt, int rk) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;
  auto &qn = zone->qn;
  auto &cn = zone->Cn;

  const real dt_div_jac = dt / zone->jac(i, j, k);
  for (int l = 0; l < param->n_var; ++l) {
    cv(i, j, k, l) = SSPRK3::a[rk] * qn(i, j, k, l) + SSPRK3::b[rk] * cv(i, j, k, l) +
                     SSPRK3::c[rk] * dt_div_jac * (zone->dq(i, j, k, l) - cn(i, j, k, l));
  }
  if (extent[2] == 1) {
    cv(i, j, k, 3) = 0;
  }

  auto &bv = zone->bv;

  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;
  const auto V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  //V^2

  auto &sv = zone->sv;
  real sum_part_den{0};
  for (int l = 0; l < param->n_spec; ++l) {
    if (cv(i, j, k, 5 + l) < 0) {
      // If the species mass fraction is negative, we set it to zero.
      cv(i, j, k, 5 + l) = 0;
    } else
      sum_part_den += cv(i, j, k, l + 5);
  }
  sum_part_den = 1.0 / sum_part_den;
  const auto denComDivDenReal = cv(i, j, k, 0) * sum_part_den;
  for (int l = 0; l < param->n_spec; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * sum_part_den;
    cv(i, j, k, 5 + l) *= denComDivDenReal;
  }
  // For multiple species or RANS methods, there will be scalars to be computed
  for (int l = param->n_spec; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
  }

  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
  } else {
    // Air
    bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * V2);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }
}

template<MixtureModel mix_model>
void wu_RK_subStep(std::vector<Field> &field, DParameter *param, Parameter &parameter, const Mesh &mesh,
  DBoundCond &bound_cond, int n_block, int step, dim3 tpb, dim3 *bpg, real dt) {
  // For every iteration, we need to save the conservative variables of the last step.
  const int n_var{parameter.get_int("n_var")};
  for (auto b = 0; b < n_block; ++b) {
    cudaMemcpy(field[b].h_ptr->qn.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
               cudaMemcpyDeviceToDevice);
  }

  // Calculate the shock sensor, and save the results to zone->shock_sensor(i,j,k).
  // if (parameter.get_string("hybrid_inviscid_scheme") != "NO") {
  //   for (auto b = 0; b < n_block; ++b) {
  //     compute_shock_sensor<<<bpg[b], tpb>>>(field[b].d_ptr, param);
  //   }
  // }

  // Rk inner iteration
  for (int rk = 0; rk < 3; ++rk) {
    for (auto b = 0; b < n_block; ++b) {
      // Set dq to 0
      cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));
    }

    compute_viscous_flux<mix_model>(mesh, field, param, parameter);

    for (auto b = 0; b < n_block; ++b) {
      // Second, for each block, compute the residual dq
      compute_inviscid_flux<mix_model>(mesh[b], field[b].d_ptr, param, n_var, parameter);
      // compute_viscous_flux<mix_model>(mesh[b], field[b].d_ptr, param, parameter);

      // Explicit temporal schemes should not use any implicit treatment.

      // Apply the non-reflecting boundary condition
      bound_cond.nonReflectingBoundary<mix_model>(mesh[b], field[b], param);

      // update basic variables
      update_cv_and_bv_rk_wu<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt, rk);

      // limit unphysical values computed by the program
      if (parameter.get_bool("limit_flow"))
        limit_flow<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // Apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      bound_cond.apply_boundary_conditions<mix_model>(mesh[b], field[b], param, step);

      update_values_with_fluctuations<<<bpg[b], tpb>>>(field[b].d_ptr, param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model>(mesh, field, parameter, step, param);

    const int ng_1 = 2 * mesh[0].ngg - 1;
    if (mesh.dimension == 2) {
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    // for (auto b = 0; b < n_block; ++b) {
    //   const int mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    //   dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    //   update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
    // }
  }
}

template<MixtureModel mix_model> void wu_splitting(Driver<mix_model> &driver) {
  if (driver.myid == 0) {
    printf("\n****************************Time advancement starts*****************************\n");
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
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    MpiParallel::exit();
  }

  auto &parameter{driver.parameter};
  // auto &ps{driver.ps};

  IOManager<mix_model> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  const int output_time_series = parameter.get_int("output_time_series");

  Monitor monitor(parameter, driver.spec, mesh);
  const int if_monitor_points{parameter.get_int("if_monitor_points")};
  const int if_monitor_blocks{parameter.get_int("if_monitor_blocks")};

  int step{parameter.get_int("step")};
  const int total_step{parameter.get_int("total_step") + step};
  const int output_screen = parameter.get_int("output_screen");
  const real total_simulation_time{parameter.get_real("total_simulation_time")};
  const int output_file = parameter.get_int("output_file");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  real dt{1e+6};
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};
  if (fixed_time_step) {
    dt = parameter.get_real("dt");
  }

  const bool if_collect_statistics{parameter.get_bool("if_collect_statistics")};
  const int collect_statistics_iter_start{parameter.get_int("start_collect_statistics_iter")};
  auto &statistics_collector{driver.stat_collector};
  const int n_rand = parameter.get_int("random_number_per_point");
  const int fluctuation_form = parameter.get_int("fluctuation_form");

  const auto need_physical_time{parameter.get_bool("need_physical_time")};
  const auto have_sponge_layer{parameter.get_bool("sponge_layer")};
  const auto hybrid_inviscid_scheme{parameter.get_string("hybrid_inviscid_scheme")};
  const int monitor_block_frequency = parameter.get_int("monitor_block_frequency");

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

    if (!fixed_time_step) {
      // compute the local time step
      for (auto b = 0; b < n_block; ++b)
        local_time_step_without_reaction<<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // After all processes and all blocks computing dt_local, we compute the global time step.
      dt = global_time_step(mesh, parameter, field);

      update_dt_global<<<1, 1>>>(param, dt);
    }

    // Compute the fluctuation values. This is updated every step, not every sub-iter in RK
    if (fluctuation_form > 0) {
      for (int b = 0; b < n_block; ++b) {
        compute_fluctuation(mesh[b], field[b].d_ptr, param, fluctuation_form, parameter, dist, gen);
      }
    }

    compute_cn<mix_model>(field, param, parameter, mesh, n_block, tpb, bpg);
    wu_reaction_subStep<mix_model>(field, param, parameter, mesh, driver.bound_cond, n_block, tpb, bpg, dt, step);
    wu_RK_subStep<mix_model>(field, param, parameter, mesh, driver.bound_cond, n_block, step, tpb, bpg, 0.5 * dt);

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
      monitor.output_block_monitors(parameter, field, physical_time);
    }
    if (if_collect_statistics && step > collect_statistics_iter_start) {
      statistics_collector.collect_data(param);
    }
    if (step % output_file == 0 || finished) {
      if (!update_copy) {
        for (int b = 0; b < n_block; ++b) {
          field[b].copy_data_from_device(parameter);
        }
        update_copy = true;
      }
      ioManager.print_field(step, parameter, physical_time);
      if (n_rand > 0)
        write_rng(mesh, parameter, field);
      if (if_collect_statistics && step > collect_statistics_iter_start)
        statistics_collector.export_statistical_data();
      post_process(driver);
      if (if_monitor_points)
        monitor.output_point_monitors();
      if (parameter.get_bool("sponge_layer")) {
        output_sponge_layer(parameter, field, mesh, driver.spec);
      }
      if (parameter.get_bool("use_df")) {
        driver.bound_cond.write_df(parameter, mesh);
      }
    }
    if (output_time_series > 0 && step % output_time_series == 0) {
      if (!update_copy) {
        for (int b = 0; b < n_block; ++b) {
          field[b].copy_data_from_device(parameter);
        }
        update_copy = true;
      }
      timeSeriesIOManager.print_field(step, parameter, physical_time);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Process %d, Error: %s\n", driver.myid, cudaGetErrorString(err));
      MpiParallel::exit();
    }
  }
  delete[] bpg;
}

template void strang_splitting<MixtureModel::Air>(Driver<MixtureModel::Air> &driver);

template void strang_splitting<MixtureModel::Mixture>(Driver<MixtureModel::Mixture> &driver);

template void wu_splitting<MixtureModel::Air>(Driver<MixtureModel::Air> &driver);

template void wu_splitting<MixtureModel::Mixture>(Driver<MixtureModel::Mixture> &driver);
}

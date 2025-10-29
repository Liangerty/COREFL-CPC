#pragma once

#include "Driver.cuh"
#include "IOManager.h"
#include "TimeAdvanceFunc.cuh"
#include "SchemeSelector.cuh"
#include "ImplicitTreatmentHPP.cuh"
#include "FieldOperation.cuh"
#include "DataCommunication.cuh"
#include "Residual.cuh"
#include "PostProcess.h"

namespace cfd {
template<MixtureModel mix_model>
void steady_simulation(Driver<mix_model> &driver) {
  if (driver.myid == 0) {
    printf("\n****************************Time advancement starts*****************************\n");
  }

  auto &parameter{driver.parameter};
  auto &mesh{driver.mesh};
  std::vector<Field> &field{driver.field};
  DParameter *param{driver.param};

  if (driver.myid == 0) {
    printf("Steady flow simulation.\n");
  }

  bool converged{false};
  int step{parameter.get_int("step")};
  int total_step{parameter.get_int("total_step") + step};
  if (parameter.get_bool("steady_before_transient")) {
    total_step = parameter.get_int("total_step_steady");
  }
  const int n_block{mesh.n_block};
  const int n_var{parameter.get_int("n_var")};
  const int ngg{mesh[0].ngg};
  const int ng_1 = 2 * ngg - 1;
  const int output_screen = parameter.get_int("output_screen");
  const int output_file = parameter.get_int("output_file");

  IOManager<mix_model> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (int b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  for (auto b = 0; b < n_block; ++b) {
    store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
    // Compute the conservative variables from basic variables
    // In steady simulations, we may also need to use the upwind high-order method;
    // we need the conservative variables.
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
  }

  while (!converged) {
    ++step;
    /*[[unlikely]]*/
    if (step > total_step) {
      break;
    }

    // Start a single iteration
    // First, store the value of last step
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    for (auto b = 0; b < n_block; ++b) {
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
      local_time_step<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // implicit treatment if needed
      implicit_treatment<mix_model>(mesh[b], param, field[b].d_ptr, parameter, field[b].h_ptr, driver.bound_cond);

      // update conservative and basic variables
      update_cv_and_bv<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // limit unphysical values computed by the program
      //limit_unphysical_variables<mix_model>(field[b].d_ptr, param, b, step, bpg[b], tpb);
      if (parameter.get_bool("limit_flow"))
        limit_flow<mix_model><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // Apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      driver.bound_cond.template apply_boundary_conditions<mix_model>(mesh[b], field[b], param, step);
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

    // Finally, test if the simulation reaches convergence state
    if (step % output_screen == 0 || step == 1) {
      // if (!update_copy) {
      for (int b = 0; b < n_block; ++b) {
        field[b].copy_data_from_device(parameter);
      }
      // update_copy = true;
      // }
      real err_max = compute_residual(driver, step);
      converged = err_max < parameter.get_real("convergence_criteria");
      if (driver.myid == 0) {
        steady_screen_output(step, err_max, driver.time, driver.res);
      }
    }
    cudaDeviceSynchronize();
    if (step % output_file == 0 || converged) {
      ioManager.print_field(step, parameter);
      post_process(driver);
    }
  }
  delete[] bpg;
}
}

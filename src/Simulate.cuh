#pragma once

#include "Driver.cuh"
#include "SteadySim.cuh"
#include "RK.cuh"
#include "DualTimeStepping.cuh"
#include "StrangSplitting.cuh"

namespace cfd {
template<MixtureModel mix_model> void simulate(Driver<mix_model> &driver) {
  auto &parameter{driver.parameter};
  if (const auto steady{parameter.get_bool("steady")}) {
    // The methods which use only bv do not need to save cv at all, which is the case in steady simulations.
    // In those methods, such as Roe, AUSM..., we do not store the cv variables.
    steady_simulation<mix_model>(driver);
  } else {
    real physical_time = parameter.get_real("solution_time");
    if (const real t0 = parameter.get_real("set_current_physical_time"); t0 > -1e-10) {
      physical_time = t0;
      parameter.update_parameter("solution_time", physical_time);
    }
    const int myid = parameter.get_int("myid");
    if (myid == 0) {
      printf("\n\t-> %10.4es : Current physical time\n", physical_time);
    }
    const real length = parameter.get_real("domain_length");
    real u = parameter.get_real("v_inf");
    if (parameter.get_int("problem_type") == 1) {
      u = parameter.get_real("convective_velocity");
    }
    if (const real u_char = parameter.get_real("characteristic_velocity"); u_char > 0) {
      u = u_char;
    }
    real flowThroughTime = length / u;
    parameter.update_parameter("flow_through_time", flowThroughTime);
    auto n_ftt = parameter.get_real("n_flowThroughTime");
    if (n_ftt > 0) {
      physical_time += n_ftt * flowThroughTime;
      parameter.update_parameter("total_simulation_time", physical_time);
      if (myid == 0) {
        printf("\t-> %10.4es : Flow through time computed with L(%9.3em)/U(%9.3em/s)\n", flowThroughTime, length, u);
        printf("\t-> %10.4e  : Number of flow through time to be computed.\n", n_ftt);
        printf("\t-> %10.4es : Physical time to be simulated.\n", n_ftt * flowThroughTime);
        printf("\t-> %10.4es : The end of the simulation time.\n", physical_time);
      }
    } else {
      // If the flow through time is not set, the total simulation time is added to the solution time.
      physical_time += parameter.get_real("total_simulation_time");
      if (myid == 0) {
        printf("\t-> %10.4es : Physical time to be simulated.\n", parameter.get_real("total_simulation_time"));
        printf("\t-> %10.4es : The end of the simulation time.\n", physical_time);
      }
      parameter.update_parameter("total_simulation_time", physical_time);
    }

    if (parameter.get_int("reaction") == 1) {
      switch (const auto temporal_tag{parameter.get_int("temporal_scheme")}) {
        case 2:
          dual_time_stepping<mix_model>(driver);
          break;
        case 4:
          wu_splitting<mix_model>(driver);
          break;
        case 3:
        default:
          RK3<mix_model>(driver);
          break;
      }
    } else {
      switch (const auto temporal_tag{parameter.get_int("temporal_scheme")}) {
        case 2:
          dual_time_stepping<mix_model>(driver);
          break;
        case 3:
        default:
          RK3<mix_model>(driver);
          break;
      }
    }
  }
  driver.deallocate();
}
}

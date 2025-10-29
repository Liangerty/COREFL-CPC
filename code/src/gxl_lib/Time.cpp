#include "Time.h"
#include <mpi.h>

gxl::Time::Time() {
  start_time       = MPI_Wtime();
  last_record_time = start_time;
}

double gxl::Time::record_current() {
  last_record_time = MPI_Wtime();
  return last_record_time;
}

double gxl::Time::get_elapsed_time() {
  const auto this_time = MPI_Wtime();
  elapsed_time         = this_time - start_time;
  step_time            = this_time - last_record_time;
  last_record_time     = this_time;
  return elapsed_time;
}

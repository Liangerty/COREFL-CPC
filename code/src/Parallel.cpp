#include "Parallel.h"
#include <mpi.h>

double cfd::MpiParallel::get_wall_time() { return MPI_Wtime(); }

void cfd::MpiParallel::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

cfd::MpiParallel::~MpiParallel() { MPI_Finalize(); }

void cfd::MpiParallel::exit() { MPI_Abort(MPI_COMM_WORLD, 1); }

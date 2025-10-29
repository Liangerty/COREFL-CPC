#include "Residual.cuh"

namespace cfd {
void steady_screen_output(int step, real err_max, gxl::Time &time, const std::array<real, 4> &res) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%11.4e\n", step, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,                       V     converged to: %11.4e   \n", step, res[1]);
  printf("  n=%8d,                       p     converged to: %11.4e   \n", step, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}

void
unsteady_screen_output(int step, real err_max, gxl::Time &time, const std::array<real, 4> &res, real dt,
  real solution_time) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%13.7e\t%13.7e\t%11.4e\n", step, dt, solution_time, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,   dt=%13.7e,   V     converged to: %11.4e   \n", step, dt, res[1]);
  printf("  n=%8d,   dt=%13.7e,   p     converged to: %11.4e   \n", step, dt, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("Current physical  time is %16.8es\n", solution_time);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}

__global__ void check_nan(DZone *zone, int blk, int myid, int n_scalar) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  // auto &sv = zone->sv;
  // std::string yks;
  // char* buf[10];
  // for (int l = 0; l < n_scalar; ++l) {
  //   if (isnan(sv(i, j, k, l))) {
  //     sprintf(buf[l], "%e", sv(i, j, k, l));
  //   } else {
  //     sprintf(buf[l], "%f", sv(i, j, k, l));
  //   }
  //   yks += buf[l];
  //   if (l < n_scalar - 1) {
  //     yks += ",";
  //   }
  // }

  // if (isnan(bv(i, j, k, 0)) || isnan(bv(i, j, k, 1)) || isnan(bv(i, j, k, 2)) || isnan(bv(i, j, k, 3)) ||
  //     isnan(bv(i, j, k, 4)) || isnan(bv(i, j, k, 5))) {
  //   printf("P[%d]B[%d](%d, %d, %d), bv = {%e,%e,%e,%e,%e,%e}"
  //          // ", sv=(%s)"
  //          "\n", myid, blk, i, j,
  //          k, bv(i, j, k, 0), bv(i, j, k, 1), bv(i, j, k, 2), bv(i, j, k, 3), bv(i, j, k, 4), bv(i, j, k, 5)
  //          // ,yks.c_str()
  //   );
  // }
}
} // cfd

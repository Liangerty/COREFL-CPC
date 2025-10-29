#include "kernels.cuh"
#include "kernels.h"
#include "Parallel.h"
#include <cstdio>
#include "DParameter.cuh"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace cfd {
void setup_gpu_device(int n_proc, int myid) {
  int deviceCountPerNode{0};
  cudaGetDeviceCount(&deviceCountPerNode);

  // if (deviceCount < n_proc) {
  //   printf("Not enough GPU devices.\n"
  //          "We want %d GPUs but only %d GPUs are available.\n"
  //          " Stop computing.\n", n_proc, deviceCount);
  //   MpiParallel::exit();
  // }
  char hostName[256];
  #ifdef _WIN32
  DWORD size = sizeof(hostName);
  if (!GetComputerName(hostName, &size)) {
    printf("Failed to get host name. Error: %lu\n", GetLastError());
  }
  #else
  if (gethostname(hostName, sizeof(hostName)) != 0) {
    printf("Failed to get host name.\n");
  }
  #endif

  cudaDeviceProp prop{};
  int gpuId = myid % deviceCountPerNode;
  int nodeId = myid / deviceCountPerNode;
  cudaGetDeviceProperties(&prop, gpuId);
  cudaSetDevice(gpuId);
  printf("\tProcess %d, will compute on device [[%d(%s)]] of node [[%d(%s)]].\n", myid, gpuId, prop.name, nodeId,
         hostName);
}

__global__ void modify_cfl(DParameter *param, real cfl) {
  param->cfl = cfl;
}
}

/**
 * @file main.cu
 * @brief Main function for the DNS program.
 * @details This program is designed to run a CFD simulation for various mixture types, including Air as calorically perfect gas, and thermally perfect gas mixture.
 * TODO: Keep only the laminar/DNS part, and delete the flamelet part. Change FR to a choice instead of a template parameter.
 */
#include <cstdio>
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"
#include "Simulate.cuh"

int main(int argc, char *argv[]) {
  cfd::Parameter parameter(&argc, &argv);

  cfd::Mesh mesh(parameter);

  const int species = parameter.get_int("species");
  if (const bool turbulent_laminar = parameter.get_bool("turbulence"); !turbulent_laminar) {
    parameter.update_parameter("turbulence_method", 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (species == 1) {
    // Multiple species
    // Laminar & DNS
    cfd::Driver<MixtureModel::Mixture> driver(parameter, mesh);
    driver.initialize_computation();
    simulate(driver);
  } else {
    // Air simulation
    // Laminar and air
    cfd::Driver<MixtureModel::Air> driver(parameter, mesh);
    driver.initialize_computation();
    simulate(driver);
  }

  if (parameter.get_int("myid") == 0) {
    printf("Yeah, baby, we are ok now\n");
    std::ofstream out("Man, we are Finished.txt");
    out<<"Yeah, baby, we are ok now\n";
    out.close();
  }
  MPI_Finalize();
  return 0;
}

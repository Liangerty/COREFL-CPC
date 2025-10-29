#include "Initialize.cuh"
#include "MixtureFraction.h"
#include "MixingLayer.cuh"
#include "DParameter.cuh"
#include <fstream>

namespace cfd {
void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  // We need to first find out the problem type.
  // For problem_type == 1, a mixing layer is required.
  if (parameter.get_int("problem_type") == 1 && !parameter.get_bool("compatible_mixing_layer")) {
    initialize_mixing_layer(parameter, mesh, field, species);
  } else {
    // For other cases, we initialize by the usual way.
    // First, find out how many groups of initial conditions are needed.
    const int tot_group{parameter.get_int("groups_init")};
    std::vector<Inflow> groups_inflow;
    const std::string default_init = parameter.get_string("default_init");
    Inflow default_inflow(default_init, species, parameter);
    groups_inflow.push_back(default_inflow);

    std::vector<real> xs{}, xe{}, ys{}, ye{}, zs{}, ze{};
    if (tot_group > 1) {
      for (int l = 0; l < tot_group - 1; ++l) {
        auto patch_struct_name = "init_cond_" + std::to_string(l);
        auto &patch_cond = parameter.get_struct(patch_struct_name);
        xs.push_back(std::get<real>(patch_cond.at("x0")));
        xe.push_back(std::get<real>(patch_cond.at("x1")));
        ys.push_back(std::get<real>(patch_cond.at("y0")));
        ye.push_back(std::get<real>(patch_cond.at("y1")));
        zs.push_back(std::get<real>(patch_cond.at("z0")));
        ze.push_back(std::get<real>(patch_cond.at("z1")));
        //groups_inflow.emplace_back(patch_struct_name, species, parameter);
        if (patch_cond.find("name") != patch_cond.cend()) {
          auto name = std::get<std::string>(patch_cond.at("name"));
          groups_inflow.emplace_back(name, species, parameter);
        } else {
          groups_inflow.emplace_back(patch_struct_name, species, parameter);
        }
      }
    }

    // Start to initialize
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      field[blk].initialize_basic_variables(parameter, groups_inflow, xs, xe, ys, ye, zs, ze, species);
    }
  }

  parameter.update_parameter("solution_time", 0.0);

  MPI_Barrier(MPI_COMM_WORLD);
  if (parameter.get_int("myid") == 0) {
    printf("\t->-> %-20s : initialization method.\n", "From start");
    std::ofstream history("history.dat", std::ios::trunc);
    history << "step\tdt(s)\tt(s)\terror_max\n";
    history.close();
  }
}

void initialize_spec_from_inflow(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species) {
  // This can also be implemented like the from_start one, which can have patches.
  // But currently, for easy to implement, initialize the whole flowfield to the inflow composition,
  // which means that other species would have to be computed from boundary conditions.
  // If the need for initialize species in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  const Inflow inflow(default_init, species, parameter);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const int mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    const auto n_spec = parameter.get_int("n_spec");
    const auto mass_frac = inflow.sv;
    auto &yk = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < n_spec; ++l) {
            yk(i, j, k, l) = mass_frac[l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from single species result. The species field is initialized with freestream.\n");
  }
  // If flamelet model, the mixture fraction should also be initialized
  if (parameter.get_int("reaction") == 2) {
    const int i_fl{parameter.get_int("i_fl")};
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      const int mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
      const auto sv_in = inflow.sv;
      auto &sv = field[blk].sv;
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            sv(i, j, k, i_fl) = sv_in[i_fl];
            sv(i, j, k, i_fl + 1) = 0;
          }
        }
      }
    }
  }
}

void initialize_turb_from_inflow(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species) {
  // This can also be implemented like the from_start one, which can have patches.
  // But currently, for easy to implement, initialize the whole flowfield to the main inflow turbulent state.
  // If the need for initialize turbulence in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  const Inflow inflow(default_init, species, parameter);
  const auto n_turb = parameter.get_int("n_turb");
  const auto n_spec = parameter.get_int("n_spec");
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const int mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    auto &sv = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < n_turb; ++l) {
            sv(i, j, k, n_spec + l) = inflow.sv[n_spec + l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from laminar result. The turbulent field is initialized with freestream.\n");
  }
}

void initialize_mixture_fraction_from_species(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  Species &species) {
  // This is called when we need to compute the mixture fraction from a given species field.
  // We need to know the form of coupling functions,
  // the boundary conditions of the 2 streams to know how to compute the mixture fraction
  Inflow *fuel = nullptr, *oxidizer = nullptr;

  // First find and initialize the fuel and oxidizer stream
  auto &bcs = parameter.get_string_array("boundary_conditions");
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto &bc_type = std::get<std::string>(bc.at("type"));
    if (bc_type == "inflow") {
      const auto z = std::get<real>(bc.at("mixture_fraction"));
      if (abs(z - 1) < 1e-10) {
        // fuel
        if (fuel == nullptr) {
          fuel = new Inflow(bc_name, species, parameter);
        } else {
          printf("Two fuel stream! Please check the boundary conditions.\n");
        }
      } else if (abs(z) < 1e-10) {
        // oxidizer
        if (oxidizer == nullptr) {
          oxidizer = new Inflow(bc_name, species, parameter);
        } else {
          printf("Two oxidizer stream! Please check the boundary conditions.\n");
        }
      }
    }
  }
  if (fuel == nullptr || oxidizer == nullptr) {
    printf("Cannot find fuel or oxidizer stream! Please check the boundary conditions.\n");
    exit(1);
  }

  // Next, see which definition of the mixture fraction is used.
  MixtureFraction *mixtureFraction = nullptr;
  if (species.elem_list.find("C") != species.elem_list.end()) {
    mixtureFraction = new BilgerCH(*fuel, *oxidizer, species, parameter.get_int("myid"));
  } else {
    mixtureFraction = new BilgerH(*fuel, *oxidizer, species, parameter.get_int("myid"));
  }

  std::vector<real> yk(species.n_spec, 0);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const int mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    auto &sv = field[blk].sv;
    const auto i_fl = parameter.get_int("i_fl");
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < species.n_spec; ++l) {
            yk[l] = sv(i, j, k, l);
          }
          sv(i, j, k, i_fl) = mixtureFraction->compute_mixture_fraction(yk);
          sv(i, j, k, i_fl + 1) = 0;
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf(
      "Previous results contain only species mass fraction info, the mixture fraction is computed via the Bilger's definition.\n");
  }
}

void expand_2D_to_3D(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field) {
  const int n_scalar{parameter.get_int("n_scalar")};
  for (size_t blk = 0; blk < mesh.n_block; ++blk) {
    const auto mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    for (int l = 0; l < 6; ++l) {
      for (int k = 1; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            field[blk].bv(i, j, k, l) = field[blk].bv(i, j, 0, l);
          }
        }
      }
    }
    for (int l = 0; l < n_scalar; ++l) {
      for (int k = 1; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            field[blk].sv(i, j, k, l) = field[blk].sv(i, j, 0, l);
          }
        }
      }
    }
  }
}

void initialize_mixing_layer(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
  const Species &species) {
  std::vector<real> var_info;
  get_mixing_layer_info(parameter, species, var_info);

  real *var_info_dev;
  cudaMalloc(&var_info_dev, sizeof(real) * var_info.size());
  cudaMemcpy(var_info_dev, var_info.data(), sizeof(real) * var_info.size(), cudaMemcpyHostToDevice);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    dim3 tpb{8, 8, 4};
    if (field[blk].block.mz == 1) {
      tpb = {16, 16, 1};
    }
    const int ngg{field[blk].block.ngg};
    dim3 bpg = {
      (field[blk].block.mx + 2 * ngg - 1) / tpb.x + 1, (field[blk].block.my + 2 * ngg - 1) / tpb.y + 1,
      (field[blk].block.mz + 2 * ngg - 1) / tpb.z + 1
    };
    int n_fl{0};
    if ((species.n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2)
      n_fl = 2;
    initialize_mixing_layer_with_info<<<bpg, tpb>>>(field[blk].d_ptr, var_info_dev, species.n_spec,
                                                    parameter.get_real("delta_omega"), parameter.get_int("n_turb"),
                                                    n_fl, parameter.get_int("n_ps"));
    cudaMemcpy(field[blk].bv.data(), field[blk].h_ptr->bv.data(), sizeof(real) * field[blk].h_ptr->bv.size() * 6,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(field[blk].sv.data(), field[blk].h_ptr->sv.data(),
               sizeof(real) * field[blk].h_ptr->sv.size() * parameter.get_int("n_scalar"), cudaMemcpyDeviceToHost);
  }
}

__global__ void
initialize_mixing_layer_with_info(DZone *zone, const real *var_info, int n_spec, real delta_omega, int n_turb, int n_fl,
  int n_ps) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const auto y = zone->y(i, j, k);
  auto &bv = zone->bv, &sv = zone->sv;

  const real u_upper = var_info[1], u_lower = var_info[8 + n_spec];
  bv(i, j, k, 1) = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / delta_omega);

  if (y >= 0) {
    const auto var = var_info;

    bv(i, j, k, 0) = var[0];
    bv(i, j, k, 2) = var[2];
    bv(i, j, k, 3) = var[3];
    bv(i, j, k, 4) = var[4];
    bv(i, j, k, 5) = var[5];
    for (int l = 0; l < n_spec; ++l) {
      sv(i, j, k, l) = var[6 + l];
    }
    if (n_turb > 0) {
      for (int l = 0; l < n_turb; ++l) {
        sv(i, j, k, n_spec + l) = var[13 + 2 * n_spec + 1 + l];
      }
    }
    if (n_fl > 0) {
      sv(i, j, k, n_spec + n_turb) = var[6 + n_spec];
      sv(i, j, k, n_spec + n_turb + 1) = 0;
    }
    if (n_ps > 0) {
      for (int l = 0; l < n_ps; ++l) {
        sv(i, j, k, n_spec + n_turb + n_fl + l) = var[14 + 2 * n_spec + 4 + 2 * l];
      }
    }
  } else {
    const auto var = &var_info[7 + n_spec];

    bv(i, j, k, 0) = var[0];
    bv(i, j, k, 2) = var[2];
    bv(i, j, k, 3) = var[3];
    bv(i, j, k, 4) = var[4];
    bv(i, j, k, 5) = var[5];
    for (int l = 0; l < n_spec; ++l) {
      sv(i, j, k, l) = var[6 + l];
    }
    if (n_turb > 0) {
      for (int l = 0; l < n_turb; ++l) {
        sv(i, j, k, n_spec + l) = var[13 + 2 * n_spec + n_turb + 1 + l];
      }
    }
    if (n_fl > 0) {
      sv(i, j, k, n_spec + n_turb) = var[13 + n_spec + n_spec];
      sv(i, j, k, n_spec + n_turb + 1) = 0;
    }
    if (n_ps > 0) {
      for (int l = 0; l < n_ps; ++l) {
        sv(i, j, k, n_spec + n_turb + n_fl + l) = var[14 + 2 * n_spec + 4 + 2 * l + 1];
      }
    }
  }
}

__global__ void
initialize_mixing_layer_with_profile(ggxl::VectorField3D<real> *profile_dPtr, int profile_idx, DZone *zone,
  int n_scalar) {
  const int ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  auto &prof = profile_dPtr[profile_idx];
  // Here, we assume the profile is in constant x direction.
  auto &bv = zone->bv, &sv = zone->sv;
  bv(i, j, k, 0) = prof(i, j, 0, 0);
  bv(i, j, k, 1) = prof(i, j, 0, 1);
  bv(i, j, k, 2) = prof(i, j, 0, 2);
  bv(i, j, k, 3) = prof(i, j, 0, 3);
  bv(i, j, k, 4) = prof(i, j, 0, 4);
  bv(i, j, k, 5) = prof(i, j, 0, 5);
  for (int l = 0; l < n_scalar; ++l) {
    sv(i, j, k, l) = prof(i, j, 0, 6 + l);
  }
}
}

#include "ConstVolumeReactor.h"
#include "Constants.h"
#include <chrono>
#include <cmath>
#include "gxl_lib/Math.hpp"

namespace cfd {
void const_volume_reactor(Parameter &parameter) {
#ifdef Combustion2Part
  const Species species(parameter);
  const Reaction reaction(parameter, species);
  const int ns = species.n_spec;

  // *************************Initialize the thermodynamic field and species mass fractions*************************
  const auto condition = parameter.get_struct("constVolumeReactor");
  real T = std::get<real>(condition.at("initial_temperature"));
  real p = std::get<real>(condition.at("initial_pressure"));

  // Initialize the species mass fractions
  std::vector<real> rhoY(ns, 0), yk(ns, 0);
  real imw = 0.0;
  const int mole_mass = std::get<int>(condition.at("mole_mass"));
  if (mole_mass == 0) {
    for (const auto &[name, idx]: species.spec_list) {
      if (condition.find(name) != condition.cend()) {
        yk[idx] = std::get<real>(condition.at(name));
        imw += yk[idx] / species.mw[idx];
      }
    }
    real y_n2 = 1;
    for (int l = 0; l < ns - 1; ++l) {
      y_n2 -= yk[l];
    }
    yk[ns - 1] = y_n2;
  } else {
    // The yk is used as mole fraction first
    for (const auto &[name, idx]: species.spec_list) {
      if (condition.find(name) != condition.cend()) {
        yk[idx] = std::get<real>(condition.at(name));
        imw += yk[idx] * species.mw[idx];
      }
    }
    real x_n2 = 1;
    for (int l = 0; l < ns - 1; ++l) {
      x_n2 -= yk[l];
    }
    yk[ns - 1] = x_n2;
    // convert to mass fraction
    imw = 1.0 / imw;
    for (int i = 0; i < ns; ++i) {
      yk[i] = yk[i] * species.mw[i] * imw;
    }
  }

  const real rho = p / (T * R_u * imw);
  for (int i = 0; i < ns; ++i) {
    rhoY[i] = yk[i] * rho;
  }
  // compute the total energy
  real E = 0;
  // real enthalpy = 0;
  std::vector<real> hk(ns, 0);
  species.compute_enthalpy(T, hk.data());
  for (int i = 0; i < ns; ++i) {
    E += yk[i] * hk[i];
  }
  // enthalpy = E;
  E -= p / rho;
  // *************************End of initialization*************************

  real time = 0.0;
  const real end_time = std::get<real>(condition.at("end_time"));
  const real dt = std::get<real>(condition.at("dt"));

  const int mechanism = std::get<int>(condition.at("mechanism"));
  const int conserve_mass_corefl = std::get<int>(condition.at("conserve_mass_corefl"));
  const int file_frequency = std::get<int>(condition.at("file_frequency"));
  const int screen_frequency = std::get<int>(condition.at("screen_frequency"));
  const int implicit_method = std::get<int>(condition.at("implicit_method"));
  const int advance_scheme = std::get<int>(condition.at("advance_scheme"));

  std::vector<real> omega_d(ns, 0);
  std::vector<real> omega(ns, 0);

  // *************************Prepare for output*************************
  FILE *fp = fopen("const_volume_reactor_output.dat", "w");
  fprintf(fp, "variables=time(s),temperature(K),pressure(Pa)");
  if (mole_mass == 1) {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",X<sub>%s</sub>", species.spec_name[l].c_str());
    }
  } else {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",Y<sub>%s</sub>", species.spec_name[l].c_str());
    }
  }
  fprintf(fp, ",step\n");
  for (int l = 0; l < ns; ++l) {
    yk[l] = rhoY[l] / rho;
  }
  fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
  if (mole_mass == 1) {
    for (int l = 0; l < ns; ++l) {
      const real xl = yk[l] / (species.mw[l] * imw);
      fprintf(fp, ",%.13e", xl);
    }
  } else {
    for (int l = 0; l < ns; ++l) {
      fprintf(fp, ",%.13e", yk[l]);
    }
  }
  fprintf(fp, ",%d\n", 0);
  // *************************End of output file opening*************************

  auto start = std::chrono::high_resolution_clock::now();
  real h = dt, err_last = 0;
  real RTol = 1e-6, ATol = 1e-8;
  int adaptive_time_step = 0, build_arnoldi = 0;
  if (advance_scheme == 1) {
    RTol = std::get<real>(condition.at("RTol"));
    ATol = std::get<real>(condition.at("ATol"));
    adaptive_time_step = std::get<int>(condition.at("adaptive_time_step"));
    build_arnoldi = std::get<int>(condition.at("build_arnoldi"));
  }

  real T_last = T;
  int step = 0;
  const int nr = reaction.n_reac;
  // ***************************Time marching loop*******************************
  while (time < end_time) {
    // The loop to advance the solution
    // Compute the concentration
    std::vector<real> q1(nr, 0), q2(nr, 0);
    compute_src(mechanism, T, species, reaction, rhoY, q1, q2, omega_d, omega);

    bool do_update{true};
    if (advance_scheme == 0) {
      if (implicit_method == 1) {
        // EPI method
        auto jac = compute_chem_src_jacobian(rhoY, species, reaction, q1, q2);
        EPI(jac, species, dt, omega);
      } else if (implicit_method == 2) {
        // DA method
        auto jac = compute_chem_src_jacobian_diagonal(rhoY, species, reaction, q1, q2);
        DA(jac, species, dt, omega);
      }

      // Update species mass fractions
      if (conserve_mass_corefl) {
        real sumRho = 0;
        for (int i = 0; i < ns; ++i) {
          rhoY[i] += omega[i] * dt;
          if (rhoY[i] < 0) {
            rhoY[i] = 0;
          }
          sumRho += rhoY[i];
        }
        for (int i = 0; i < ns; ++i) {
          yk[i] = rhoY[i] / sumRho;
          rhoY[i] = yk[i] * rho;
        }
      } else {
        for (int i = 0; i < ns; ++i) {
          rhoY[i] += omega[i] * dt;
          if (rhoY[i] < 0) {
            rhoY[i] = 0;
          }
        }
        for (int i = 0; i < ns; ++i) {
          yk[i] = rhoY[i] / rho;
        }
      }
    } else if (advance_scheme == 1) {
      // ROK4E
      const real h_step = std::min(h, end_time - time);
      std::vector<real> k1(ns, 0), k2(ns, 0), k3(ns, 0), k4(ns, 0);
      if (build_arnoldi) {
        constexpr int KrylovMaxDim = MAX_SPEC_NUMBER, KrylovMinDim = 4;
        real Q[KrylovMaxDim * MAX_SPEC_NUMBER]{};
        real H[KrylovMaxDim * KrylovMaxDim]{};
        const int krylovDim = buildArnoldi(species, rhoY, omega.data(), Q, H, KrylovMaxDim, mechanism, T, reaction, rho,
                                           E, KrylovMinDim);
        real g[KrylovMaxDim]{};

        // Stage 1
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], omega.data(), ns);
          // g[r] = h_step * gxl::vec_dot(&Q[r * ns], omega.data(), ns);
        }
        real LHS[KrylovMaxDim * KrylovMaxDim]{};
        // LHS = I - h * gamma * H
        real b[KrylovMaxDim]{};
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
          for (int j = 0; j < krylovDim; ++j) {
            if (i == j) {
              LHS[i * krylovDim + j] = 1.0 - h_step * rok4e::gamma * H[i * KrylovMaxDim + j];
            } else {
              LHS[i * krylovDim + j] = -h_step * rok4e::gamma * H[i * KrylovMaxDim + j];
            }
          }
        }
        auto ipiv = lu_decomp(LHS, krylovDim);
        real lhs[KrylovMaxDim * KrylovMaxDim]{};
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, b, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k1[l] = omega[l] - s;
        }

        // Stage 2
        std::vector<real> y_stage(ns, 0);
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * rok4e::alpha21 * k1[l];
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        real T_stage = update_t(T, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int l = 0; l < ns; ++l) {
          omega[l] += rok4e::gamma21Gamma * k1[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], omega.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, b, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k2[l] = omega[l] - s - rok4e::gamma21Gamma * k1[l];
        }
        // for (int l = 0; l < ns; ++l) {
          // k2[l] -= rok4e::gamma21Gamma * k1[l];
        // }

        // Stage 3
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * (rok4e::alpha31 * k1[l] + rok4e::alpha32 * k2[l]);
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        T_stage = update_t(T_stage, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        std::vector<real> f(ns, 0);
        for (int l = 0; l < ns; ++l) {
          f[l] = omega[l] + rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], f.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, g, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k3[l] = f[l] - s - rok4e::gamma31Gamma * k1[l] - rok4e::gamma32Gamma * k2[l];
        }

        // Stage 4
        for (int l = 0; l < ns; ++l) {
          f[l] = omega[l] + rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
        // Project
        for (int r = 0; r < krylovDim; ++r) {
          g[r] = gxl::vec_dot(&Q[r * ns], f.data(), ns);
        }
        for (int i = 0; i < krylovDim; ++i) {
          b[i] = g[i];
        }
        for (int l = 0; l < krylovDim * krylovDim; ++l) {
          lhs[l] = LHS[l];
        }
        lu_to_solution(lhs, g, krylovDim, ipiv);
        for (int i = 0; i < ns; ++i) {
          g[i] -= b[i];
        }
        // Back project
        for (int l = 0; l < ns; ++l) {
          real s = 0;
          for (int r = 0; r < krylovDim; ++r) {
            s += Q[r * ns + l] * g[r];
          }
          k4[l] = f[l] - s - rok4e::gamma41Gamma * k1[l] - rok4e::gamma42Gamma * k2[l] - rok4e::gamma43Gamma * k3[l];
        }
      } else { // First, compute the chemical jacobian matrix
        auto jac = compute_chem_src_jacobian(rhoY, species, reaction, q1, q2);
        // Assemble the LHS matrix and perform the LU decomposition
        // LHS = I - h * gamma * J
        std::vector<real> LHS(ns * ns, 0);
        for (int m = 0; m < ns; ++m) {
          for (int n = 0; n < ns; ++n) {
            if (m == n) {
              LHS[m * ns + n] = 1.0 - h_step * rok4e::gamma * jac[m * ns + n];
            } else {
              LHS[m * ns + n] = -h_step * rok4e::gamma * jac[m * ns + n];
            }
          }
        }
        auto ipiv = lu_decomp(LHS.data(), ns);

        // stage 1
        auto lhs1 = LHS;
        for (int i = 0; i < ns; ++i) {
          k1[i] = omega[i];
        }
        lu_to_solution(lhs1.data(), k1.data(), ns, ipiv);

        // stage 2
        lhs1 = LHS;
        std::vector<real> y_stage(ns, 0);
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * rok4e::alpha21 * k1[l];
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        // solve T with the new y_stage
        real T_stage = update_t(T, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int i = 0; i < ns; ++i) {
          k2[i] = omega[i] + rok4e::gamma21Gamma * k1[i];
        }
        lu_to_solution(lhs1.data(), k2.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k2[l] -= rok4e::gamma21Gamma * k1[l];
        }

        // stage 3
        lhs1 = LHS;
        imw = 0;
        for (int l = 0; l < ns; ++l) {
          y_stage[l] = rhoY[l] + h_step * (rok4e::alpha31 * k1[l] + rok4e::alpha32 * k2[l]);
          yk[l] = y_stage[l] / rho;
          imw += yk[l] / species.mw[l];
        }
        T_stage = update_t(T_stage, imw, species, yk, E);
        compute_src(mechanism, T_stage, species, reaction, y_stage, q1, q2, omega_d, omega);
        for (int l = 0; l < ns; ++l) {
          k3[l] = omega[l] + rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }
        lu_to_solution(lhs1.data(), k3.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k3[l] -= rok4e::gamma31Gamma * k1[l] + rok4e::gamma32Gamma * k2[l];
        }

        // stage 4
        for (int l = 0; l < ns; ++l) {
          k4[l] = omega[l] + rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
        lhs1 = LHS;
        lu_to_solution(lhs1.data(), k4.data(), ns, ipiv);
        for (int l = 0; l < ns; ++l) {
          k4[l] -= rok4e::gamma41Gamma * k1[l] + rok4e::gamma42Gamma * k2[l] + rok4e::gamma43Gamma * k3[l];
        }
      }

      if (adaptive_time_step) {
        // Update rhoY with error control
        std::vector<real> rhoY_new(ns, 0);
        real err{0};
        for (int l = 0; l < ns; ++l) {
          rhoY_new[l] = rhoY[l] + h_step * (rok4e::b1 * k1[l] + rok4e::b2 * k2[l] + rok4e::b4 * k4[l]);
          real dy = h_step * (rok4e::e1 * k1[l] + rok4e::e2 * k2[l] + rok4e::e3 * k3[l] + rok4e::e4 * k4[l]);
          dy /= RTol * rhoY_new[l] + ATol;
          err += dy * dy;
        }
        err = sqrt(err / ns);
        real hStar = h_step;
        if (step > 0) {
          hStar *= std::min(5.0, std::max(0.2, 0.8 * pow(err_last, 0.1) * pow(err + 1e-20, -0.175)));
        } else {
          // step == 0
          if (err > 1) {
            hStar *= 0.2;
          }
        }
        if (err <= 1) {
          // Accept the step
          for (int l = 0; l < ns; ++l) {
            rhoY[l] = rhoY_new[l];
            yk[l] = rhoY_new[l] / rho;
          }
          time += h_step;
          err_last = err;
          h = hStar;
          // if (abs(hStar - h_step) / h_step > 0.01)
            // printf("Step = %d, dt changes from %.3e to %.3e, err = %.3e\n", step, h_step, hStar, err);
        } else {
          // Reject the step
          do_update = false;
          h = hStar;
          // printf("Reject step = %d, dt changes from %.3e to %.3e, err = %.3e\n", step, h_step, hStar, err);
        }
      } else {
        for (int l = 0; l < ns; ++l) {
          rhoY[l] += h_step * (rok4e::b1 * k1[l] + rok4e::b2 * k2[l] + rok4e::b4 * k4[l]);
          yk[l] = rhoY[l] / rho;
        }
        time += h_step;
      }
    } else if (advance_scheme == 3) {
      // RK-3 explicit
      std::vector<real> rhoYN(ns, 0);
      // stage 1
      imw = 0.0;
      for (int l = 0; l < ns; ++l) {
        rhoYN[l] = rhoY[l] + omega[l] * dt;
        // if (rhoYN[l] < 0) {
        //   rhoYN[l] = 0;
        // }
        yk[l] = rhoYN[l] / rho;
        imw += yk[l] / species.mw[l];
      }
      // stage 2
      real T_stage = update_t(T, imw, species, yk, E);
      compute_src(mechanism, T_stage, species, reaction, rhoYN, q1, q2, omega_d, omega);
      imw = 0;
      for (int l = 0; l < ns; ++l) {
        rhoYN[l] = 0.75 * rhoY[l] + 0.25 * (rhoYN[l] + omega[l] * dt);
        // if (rhoYN[l] < 0) {
        //   rhoYN[l] = 0;
        // }
        yk[l] = rhoYN[l] / rho;
        imw += yk[l] / species.mw[l];
      }
      // stage 3
      T_stage = update_t(T_stage, imw, species, yk, E);
      compute_src(mechanism, T_stage, species, reaction, rhoYN, q1, q2, omega_d, omega);
      for (int l = 0; l < ns; ++l) {
        rhoY[l] = (rhoY[l] + 2.0 * (rhoYN[l] + omega[l] * dt)) / 3.0;
        if (rhoY[l] < 0) {
          rhoY[l] = 0;
        }
      }
      for (int i = 0; i < ns; ++i) {
        yk[i] = rhoY[i] / rho;
      }
    }

    // Update imw and T
    if (!do_update) {
      continue;
    }
    imw = 0.0;
    for (int i = 0; i < ns; ++i) {
      imw += yk[i] / species.mw[i];
    }
    T_last = T;
    // T = update_t_with_h(T_last, imw, species, yk, h);
    T = update_t(T_last, imw, species, yk, E);
    p = rho * R_u * imw * T;
    if (advance_scheme == 0 || advance_scheme == 3) {
      time += dt;
    }
    ++step;

    // Output results
    if (mole_mass == 1) {
      // use mole fraction for output
      if (step % screen_frequency == 0) {
        const real xH2 = yk[0] / (species.mw[0] * imw) * 100;
        const real xO2 = yk[2] / (species.mw[2] * imw) * 100;
        const real xH2O = yk[7] / (species.mw[7] * imw) * 100;
        printf("Step: %d, Time: %.3e s, T: %.5f K, P: %.2f Pa, H2%%: %.4e, O2%%: %.4e, H2O%%: %.4e\n", step, time, T, p,
               xH2, xO2, xH2O);
      }
      if (step % file_frequency == 0 || end_time - time < dt) {
        fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
        for (int l = 0; l < ns; ++l) {
          const real xl = yk[l] / (species.mw[l] * imw);
          fprintf(fp, ",%.13e", xl);
        }
        fprintf(fp, ",%d\n", step);
      }
    } else {
      if (step % screen_frequency == 0) {
        printf("Step: %d, Time: %.6e s, T: %.6f K, P: %.2f Pa, H2%%: %.4e, O2%%: %.4e, H2O%%: %.4e\n", step, time, T, p,
               yk[0], yk[2], yk[7]);
      }
      if (step % file_frequency == 0) {
        fprintf(fp, "%.13e,%.13e,%.13e", time, T, p);
        for (int l = 0; l < ns; ++l) {
          fprintf(fp, ",%.13e", yk[l]);
        }
        fprintf(fp, ",%d\n", step);
      }
    }
  }
  // ***************************End of time marching loop*******************************
  fclose(fp);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("Total simulation time is %.6es.\n", duration.count());
#endif
}

#ifdef Combustion2Part
void compute_src(int mechanism, real t, const Species &species, const Reaction &reaction, const std::vector<real> &rhoY,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega) {
  const int ns = species.n_spec;
  std::vector<real> c(ns, 0);
  for (int l = 0; l < ns; ++l) {
    c[l] = rhoY[l] / species.mw[l] * 1e-3; // Convert to mol/cm3
  }

  const int nr = reaction.n_reac;
  if (mechanism == 1) {
    // Hard-coded mechanism of Li's 9s19r H2
    chemical_source_hardCoded1(t, species, c, q1, q2, omega_d, omega);
  } else {
    // Any mechanism provided by the user
    auto kf = forward_reaction_rate(t, species, reaction, c);
    auto kb = backward_reaction_rate(t, species, reaction, kf, c);
    std::vector<real> q(nr, 0);
    rate_of_progress(kf, kb, c, q, q1, q2, species, reaction);
    chemical_source(q1, q2, omega_d, omega, species, reaction);
  }
}

real log10_real(real x) {
  // log10(x) = ln(x) / ln(10)
  constexpr real INV_LN10 = 4.342944819032518e-01;
  return std::log(x) * INV_LN10;
}

real pow10_real(real p) {
  // 10^p = exp(ln(10) * p)
  constexpr real kLn10 = 2.302585092994046; // ln(10) 2.302585092994046
  return std::exp(kLn10 * p);
}

void chemical_source_hardCoded1(real t, const Species &species, const std::vector<real> &c,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega) {
  // Hard-coded chemical source term for Li's 9 species 19 reactions H2 mechanism
  // According to chemistry/2004-Li-IntJ.Chem.Kinet.inp
  // 0:H2, 1:H, 2:O2, 3:O, 4:OH, 5:HO2, 6:H2O2, 7:H2O, 8:N2
  constexpr int ns = 9;

  // G/RT for Kc
  std::vector<real> gibbs_rt(ns, 0);
  compute_gibbs_div_rt(t, species, gibbs_rt);

  // thermodynamic scaling for kc and kb
  const real temp_t = p_atm / R_u * 1e-3 / t; // Unit is mol/cm3
  const real iTemp_t = 1.0 / temp_t;

  // Arrhenius parameters
  const real iT = 1.0 / t;
  const real iRcT = 1.0 / R_c * iT;
  const real logT = std::log(t);

  // Third body concentrations used here
  const real cc = c[0] * 2.5 + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] * 12 + c[8];

  // Production / destruction in mol/(cm3*s)
  real prod0{0}, prod1{0}, prod2{0}, prod3{0}, prod4{0}, prod5{0}, prod6{0}, prod7{0}, prod8{0};
  real dest0{0}, dest1{0}, dest2{0}, dest3{0}, dest4{0}, dest5{0}, dest6{0}, dest7{0}, dest8{0};

  // ----Reaction 0: H + O2 = O + OH
  {
    real kfb = 3.55e+15 * std::exp(-0.41 * logT - 1.66e+4 * iRcT);
    real qfb = kfb * c[1] * c[2];
    q1[0] = qfb;
    dest1 += qfb;
    dest2 += qfb;
    prod3 += qfb;
    prod4 += qfb;
    kfb = kfb * std::exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[3] + gibbs_rt[4]);
    qfb = kfb * c[3] * c[4];
    q2[0] = qfb;
    dest3 += qfb;
    dest4 += qfb;
    prod1 += qfb;
    prod2 += qfb;
  }
  // ----Reaction 1: H2 + O = H + OH
  {
    real kfb = 5.08e+4 * std::exp(2.67 * logT - 6290 * iRcT);
    real qfb = kfb * c[0] * c[3];
    q1[1] = qfb;
    dest0 += qfb;
    dest3 += qfb;
    prod1 += qfb;
    prod4 += qfb;
    kfb *= std::exp(-gibbs_rt[0] - gibbs_rt[3] + gibbs_rt[1] + gibbs_rt[4]);
    qfb = kfb * c[1] * c[4];
    q2[1] = qfb;
    dest1 += qfb;
    dest4 += qfb;
    prod0 += qfb;
    prod3 += qfb;
  }
  // ----Reaction 2: H2 + OH = H2O + H
  {
    real kfb = 2.16e+8 * std::exp(1.51 * logT - 3430 * iRcT);
    real qfb = kfb * c[0] * c[4];
    q1[2] = qfb;
    dest0 += qfb;
    dest4 += qfb;
    prod7 += qfb;
    prod1 += qfb;
    kfb *= std::exp(-gibbs_rt[0] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[1]);
    qfb = kfb * c[7] * c[1];
    q2[2] = qfb;
    dest7 += qfb;
    dest1 += qfb;
    prod0 += qfb;
    prod4 += qfb;
  }
  // ----Reaction 3: O + H2O = OH + OH
  {
    real kf = 2.97e+6 * std::exp(2.02 * logT - 13400 * iRcT);
    real qf = kf * c[3] * c[7];
    q1[3] = qf;
    dest3 += qf;
    dest7 += qf;
    prod4 += 2 * qf;
    kf *= std::exp(-gibbs_rt[3] - gibbs_rt[7] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[3] = qf;
    dest4 += 2 * qf;
    prod3 += qf;
    prod7 += qf;
  }
  // ----Reaction 4: H2 + M = H + H + M
  {
    real kf = 4.58e+19 * std::exp(-1.40 * logT - 1.0438e+5 * iRcT);
    kf *= cc;
    real qf = kf * c[0];
    q1[4] = qf;
    dest0 += qf;
    prod1 += 2 * qf;
    kf *= iTemp_t * std::exp(-gibbs_rt[0] + gibbs_rt[1] + gibbs_rt[1]);
    qf = kf * c[1] * c[1];
    q2[4] = qf;
    dest1 += 2 * qf;
    prod0 += qf;
  }
  // ----Reaction 5: O + O + M = O2 + M
  {
    real kf = 6.16e+15 * sqrt(iT) * cc;
    real qf = kf * c[3] * c[3];
    q1[5] = qf;
    dest3 += 2 * qf;
    prod2 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[3] - gibbs_rt[3] + gibbs_rt[2]);
    qf = kf * c[2];
    q2[5] = qf;
    dest2 += qf;
    prod3 += 2 * qf;
  }
  // ----Reaction 6: O + H + M = OH + M
  {
    real kf = 4.71e+18 * iT * cc;
    real qf = kf * c[1] * c[3];
    q1[6] = qf;
    dest3 += qf;
    dest1 += qf;
    prod4 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[3] - gibbs_rt[1] + gibbs_rt[4]);
    qf = kf * c[4];
    q2[6] = qf;
    dest4 += qf;
    prod3 += qf;
    prod1 += qf;
  }
  // ----Reaction 7: H + OH + M = H2O + M
  {
    real kf = 3.8e+22 * iT * iT * cc;
    real qf = kf * c[1] * c[4];
    q1[7] = qf;
    dest1 += qf;
    dest4 += qf;
    prod7 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[1] - gibbs_rt[4] + gibbs_rt[7]);
    qf = kf * c[7];
    q2[7] = qf;
    dest7 += qf;
    prod1 += qf;
    prod4 += qf;
  }
  // ----Reaction 8: H + O2 (+M) = HO2 (+M)
  {
    const real kf_high = 1.48e+12 * std::exp(0.6 * logT);
    const real kf_low = 6.37e+20 * std::exp(-1.72 * logT - 5.2e+2 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real cc2 = c[0] * 2 + c[1] + c[2] * 0.78 + c[3] + c[4] + c[5] + c[6] + c[7] * 11 + c[8];
      const real reduced_pressure = kf_low * cc2 / kf_high;
      constexpr real logFc = -0.09691001300805639; // log10(0.8) = -9.691001300805639e-02
      constexpr real cT = -0.4 - 0.67 * logFc;
      constexpr real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[1] * c[2];
    q1[8] = qf;
    dest1 += qf;
    dest2 += qf;
    prod5 += qf;
    kf *= temp_t * std::exp(-gibbs_rt[1] - gibbs_rt[2] + gibbs_rt[5]);
    qf = kf * c[5];
    q2[8] = qf;
    dest5 += qf;
    prod1 += qf;
    prod2 += qf;
  }
  // ----Reaction 9: HO2 + H = H2 + O2
  {
    real kf = 1.66e+13 * std::exp(-820 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[9] = qf;
    dest5 += qf;
    dest1 += qf;
    prod0 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[0] + gibbs_rt[2]);
    qf = kf * c[0] * c[2];
    q2[9] = qf;
    dest0 += qf;
    dest2 += qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 10: HO2 + H = OH + OH
  {
    real kf = 7.08e+13 * std::exp(-300 * iRcT);
    real qf = kf * c[5] * c[1];
    q1[10] = qf;
    dest5 += qf;
    dest1 += qf;
    prod4 += 2 * qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[1] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[10] = qf;
    dest4 += 2 * qf;
    prod5 += qf;
    prod1 += qf;
  }
  // ----Reaction 11: HO2 + O = O2 + OH
  {
    real qf = 3.25e+13 * c[5] * c[3];
    q1[11] = qf;
    dest5 += qf;
    dest3 += qf;
    prod2 += qf;
    prod4 += qf;
    const real kb = 3.25e+13 * std::exp(-gibbs_rt[5] - gibbs_rt[3] + gibbs_rt[2] + gibbs_rt[4]);
    qf = kb * c[2] * c[4];
    q2[11] = qf;
    dest2 += qf;
    dest4 += qf;
    prod5 += qf;
    prod3 += qf;
  }
  // ----Reaction 12: HO2 + OH = H2O + O2
  {
    real kf = 2.89e+13 * std::exp(500 * iRcT);
    real qf = kf * c[5] * c[4];
    q1[12] = qf;
    dest5 += qf;
    dest4 += qf;
    prod7 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[4] + gibbs_rt[7] + gibbs_rt[2]);
    qf = kf * c[7] * c[2];
    q2[12] = qf;
    dest7 += qf;
    dest2 += qf;
    prod5 += qf;
    prod4 += qf;
  }
  // ----Reaction 13: HO2 + HO2 = H2O2 + O2
  {
    real kf = 4.20e+14 * std::exp(-11980 * iRcT) + 1.30e+11 * std::exp(1630 * iRcT);
    real qf = kf * c[5] * c[5];
    q1[13] = qf;
    dest5 += 2 * qf;
    prod6 += qf;
    prod2 += qf;
    kf *= std::exp(-gibbs_rt[5] - gibbs_rt[5] + gibbs_rt[6] + gibbs_rt[2]);
    qf = kf * c[6] * c[2];
    q2[13] = qf;
    dest6 += qf;
    dest2 += qf;
    prod5 += 2 * qf;
  }
  // ----Reaction 14: H2O2 (+ M) = OH + OH (+ M)
  {
    const real kf_high = 2.95e+14 * std::exp(-48400 * iRcT);
    const real kf_low = 1.20e+17 * std::exp(-45500 * iRcT);
    real kf;
    if (kf_high < 1e-25 && kf_low < 1e-25) {
      // If both kf_high and kf_low are too small, set kf to zero
      kf = 0;
    } else {
      const real reduced_pressure = kf_low * cc / kf_high;
      constexpr real logFc = -3.010299956639812e-01; // log10(0.5) = -3.010299956639812e-01
      const real cT = -0.4 - 0.67 * logFc;
      const real nT = 0.75 - 1.27 * logFc;
      const real logPr = log10_real(reduced_pressure);
      const real tempo = (logPr + cT) / (nT - 0.14 * (logPr + cT));
      const real p = logFc / (1.0 + tempo * tempo);
      kf = kf_high * reduced_pressure / (1.0 + reduced_pressure) * pow10_real(p);
    }
    real qf = kf * c[6];
    q1[14] = qf;
    dest6 += qf;
    prod4 += 2 * qf;
    kf *= iTemp_t * std::exp(-gibbs_rt[6] + gibbs_rt[4] + gibbs_rt[4]);
    qf = kf * c[4] * c[4];
    q2[14] = qf;
    dest4 += 2 * qf;
    prod6 += qf;
  }
  // ----Reaction 15: H2O2 + H = H2O + OH
  {
    real kf = 2.41e+13 * std::exp(-3970 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[15] = qf;
    dest6 += qf;
    dest1 += qf;
    prod7 += qf;
    prod4 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[7] + gibbs_rt[4]);
    qf = kf * c[7] * c[4];
    q2[15] = qf;
    dest7 += qf;
    dest4 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ---Reaction 16: H2O2 + H = HO2 + H2
  {
    real kf = 4.82e+13 * std::exp(-7950 * iRcT);
    real qf = kf * c[6] * c[1];
    q1[16] = qf;
    dest6 += qf;
    dest1 += qf;
    prod5 += qf;
    prod0 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[1] + gibbs_rt[5] + gibbs_rt[0]);
    qf = kf * c[5] * c[0];
    q2[16] = qf;
    dest5 += qf;
    dest0 += qf;
    prod6 += qf;
    prod1 += qf;
  }
  // ----Reaction 17: H2O2 + O = OH + HO2
  {
    real kf = 9.55e+6 * t * t * std::exp(-3970 * iRcT);
    real qf = kf * c[6] * c[3];
    q1[17] = qf;
    dest6 += qf;
    dest3 += qf;
    prod4 += qf;
    prod5 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[3] + gibbs_rt[4] + gibbs_rt[5]);
    qf = kf * c[4] * c[5];
    q2[17] = qf;
    dest4 += qf;
    dest5 += qf;
    prod6 += qf;
    prod3 += qf;
  }
  // ----Reaction 18: H2O2 + OH = HO2 + H2O
  {
    real kf = 1e+12 + 5.8e+14 * std::exp(-9560 * iRcT);
    real qf = kf * c[6] * c[4];
    q1[18] = qf;
    dest6 += qf;
    dest4 += qf;
    prod5 += qf;
    prod7 += qf;
    kf *= std::exp(-gibbs_rt[6] - gibbs_rt[4] + gibbs_rt[5] + gibbs_rt[7]);
    qf = kf * c[5] * c[7];
    q2[18] = qf;
    dest5 += qf;
    dest7 += qf;
    prod6 += qf;
    prod4 += qf;
  }

  // Compute net production rates
  // Convert from mol/(cm3*s) to kg/(m3*s)
  omega_d[0] = dest0 * 2016;
  omega_d[1] = dest1 * 1008;
  omega_d[2] = dest2 * 31998;
  omega_d[3] = dest3 * 15999;
  omega_d[4] = dest4 * 17007;
  omega_d[5] = dest5 * 33006;
  omega_d[6] = dest6 * 34014;
  omega_d[7] = dest7 * 18015;
  omega_d[8] = dest8 * 28014;

  omega[0] = (prod0 - dest0) * 2016;
  omega[1] = (prod1 - dest1) * 1008;
  omega[2] = (prod2 - dest2) * 31998;
  omega[3] = (prod3 - dest3) * 15999;
  omega[4] = (prod4 - dest4) * 17007;
  omega[5] = (prod5 - dest5) * 33006;
  omega[6] = (prod6 - dest6) * 34014;
  omega[7] = (prod7 - dest7) * 18015;
  omega[8] = (prod8 - dest8) * 28014;
}

void compute_gibbs_div_rt(real t, const Species &species, std::vector<real> &gibbs_rt) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t_inv{1 / t}, log_t{std::log(t)};
  const int ns = species.n_spec;
  for (int i = 0; i < ns; ++i) {
    if (t < species.t_low[i]) {
      const real tt = species.t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt_inv = 1 / tt, log_tt = std::log(tt);
      const auto &coeff = species.low_temp_coeff;
      gibbs_rt[i] = coeff(i, 0) * (1.0 - log_tt) - 0.5 * coeff(i, 1) * tt - coeff(i, 2) * tt2 / 6.0 -
                    coeff(i, 3) * tt3 / 12.0 - coeff(i, 4) * tt4 * 0.05 + coeff(i, 5) * tt_inv - coeff(i, 6);
    } else {
      const auto &coeff = t < species.t_mid[i] ? species.low_temp_coeff : species.high_temp_coeff;
      gibbs_rt[i] =
          coeff(i, 0) * (1.0 - log_t) - 0.5 * coeff(i, 1) * t - coeff(i, 2) * t2 / 6.0 - coeff(i, 3) * t3 / 12.0 -
          coeff(i, 4) * t4 * 0.05 + coeff(i, 5) * t_inv - coeff(i, 6);
    }
  }
}

real update_t(real T0, real imw, const Species &species, const std::vector<real> &yk, real E) {
  // Use Newton-Raphson method to update temperature
  real t = T0;
  const int ns = species.n_spec;
  constexpr int max_iter = 100;
  constexpr real eps = 1e-8;

  const real R = R_u * imw;
  real err = 1e+6;
  int iter = 0;
  while (err > eps && iter++ < max_iter) {
    std::vector<real> hk(ns, 0), cpk(ns, 0);
    species.compute_enthalpy_and_cp(t, hk.data(), cpk.data());
    real cp_tot{0}, h{0};
    for (int l = 0; l < ns; ++l) {
      cp_tot += cpk[l] * yk[l];
      h += hk[l] * yk[l];
    }
    const real et = h - R * t;
    const real cv = cp_tot - R;
    const real t1 = t - (et - E) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  return t;
}

real update_t_with_h(real T0, real imw, const Species &species, const std::vector<real> &yk, real h) {
  real t = T0;
  const int ns = species.n_spec;
  constexpr int max_iter = 100;
  constexpr real eps = 1e-8;

  const real R = R_u * imw;
  real err = 1e+6;
  int iter = 0;
  while (err > eps && iter++ < max_iter) {
    std::vector<real> hk(ns, 0), cpk(ns, 0);
    species.compute_enthalpy_and_cp(t, hk.data(), cpk.data());
    real cp_tot{0}, h_iter = 0;
    for (int l = 0; l < ns; ++l) {
      cp_tot += cpk[l] * yk[l];
      h_iter += hk[l] * yk[l];
    }
    const real t1 = t - (h_iter - h) / cp_tot;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  return t;
}

std::vector<real> forward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &c) {
  const int ns = species.n_spec, nr = reaction.n_reac;
  std::vector<real> kf(nr, 0);
  const auto &A = reaction.A, &b = reaction.b, &Ea = reaction.Ea;
  const auto &type = reaction.label;
  const auto &A2 = reaction.A2, &b2 = reaction.b2, &Ea2 = reaction.Ea2;
  const auto &third_body_coeff = reaction.third_body_coeff;
  const auto &alpha = reaction.troe_alpha, &t3 = reaction.troe_t3, &t1 = reaction.troe_t1, &t2 = reaction.troe_t2;
  for (int i = 0; i < nr; ++i) {
    kf[i] = arrhenius(t, A[i], b[i], Ea[i]);
    if (type[i] == 3) {
      // Duplicate reaction
      kf[i] += arrhenius(t, A2[i], b2[i], Ea2[i]);
    } else if (type[i] > 3) {
      real cc{0};
      for (int l = 0; l < ns; ++l) {
        cc += c[l] * third_body_coeff(i, l);
      }
      if (type[i] == 4) {
        // Third body reaction
        kf[i] *= cc;
      } else {
        const real kf_low = arrhenius(t, A2[i], b2[i], Ea2[i]);
        const real kf_high = kf[i];
        if (kf_high < 1e-25 && kf_low < 1e-25) {
          // If both kf_high and kf_low are too small, set kf to zero
          kf[i] = 0;
          continue;
        }
        const real reduced_pressure = kf_low * cc / kf_high;
        real F = 1.0;
        if (type[i] > 5) {
          // Troe form
          real f_cent = (1 - alpha[i]) * std::exp(-t / t3[i]) + alpha[i] * std::exp(-t / t1[i]);
          if (type[i] == 7) {
            f_cent += std::exp(-t2[i] / t);
          }
          const real logFc = std::log10(f_cent);
          const real cn = -0.4 - 0.67 * logFc;
          const real n = 0.75 - 1.27 * logFc;
          const real logPr = std::log10(reduced_pressure);
          const real tempo = (logPr + cn) / (n - 0.14 * (logPr + cn));
          const real p = logFc / (1.0 + tempo * tempo);
          F = std::pow(10, p);
        }
        kf[i] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * F;
      }
    }
  }
  return std::move(kf);
}

std::vector<real> backward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &kf, const std::vector<real> &c) {
  int n_gibbs{reaction.n_reac};
  const int nr = reaction.n_reac;
  const auto &type = reaction.label;
  std::vector<real> kb(nr, 0);
  for (int i = 0; i < nr; ++i) {
    if (type[i] == 0) {
      // Irreversible reaction
      kb[i] = 0;
      --n_gibbs;
    } else if (reaction.rev_type[i] == 1) {
      // REV reaction
      kb[i] = arrhenius(t, reaction.A2[i], reaction.b2[i], reaction.Ea2[i]);
      if (type[i] == 4) {
        // Third body required
        real cc{0};
        for (int l = 0; l < species.n_spec; ++l) {
          cc += c[l] * reaction.third_body_coeff(i, l);
        }
        kb[i] *= cc;
      }
      --n_gibbs;
    }
  }
  if (n_gibbs < 1)
    return std::move(kb);

  std::vector<real> gibbs_rt(species.n_spec, 0);
  compute_gibbs_div_rt(t, species, gibbs_rt);
  constexpr real temp_p = p_atm / R_u * 1e-3; // Convert the unit to mol*K/cm3
  const real temp_t = temp_p / t;             // Unit is mol/cm3
  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  const auto order = reaction.order;
  for (int i = 0; i < nr; ++i) {
    if (type[i] != 2 && type[i] != 0) {
      real d_gibbs{0};
      for (int l = 0; l < species.n_spec; ++l) {
        d_gibbs += gibbs_rt[l] * (stoi_b(i, l) - stoi_f(i, l));
      }
      const real kc{std::pow(temp_t, order[i]) * std::exp(-d_gibbs)};
      kb[i] = kf[i] / kc;
    }
  }
  return std::move(kb);
}

void rate_of_progress(const std::vector<real> &kf, const std::vector<real> &kb, const std::vector<real> &c,
  std::vector<real> &q, std::vector<real> &q1, std::vector<real> &q2, const Species &species,
  const Reaction &reaction) {
  const int ns{species.n_spec};
  const auto &stoi_f{reaction.stoi_f}, &stoi_b{reaction.stoi_b};
  for (int i = 0; i < reaction.n_reac; ++i) {
    if (reaction.label[i] != 0) {
      q1[i] = 1.0;
      q2[i] = 1.0;
      for (int j = 0; j < ns; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
        q2[i] *= std::pow(c[j], stoi_b(i, j));
      }
      q1[i] *= kf[i];
      q2[i] *= kb[i];
      q[i] = q1[i] - q2[i];
    } else {
      q1[i] = 1.0;
      q2[i] = 0.0;
      for (int j = 0; j < ns; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
      }
      q1[i] *= kf[i];
      q[i] = q1[i];
    }
  }
}

void chemical_source(const std::vector<real> &q1, const std::vector<real> &q2, std::vector<real> &omega_d,
  std::vector<real> &omega, const Species &species, const Reaction &reaction) {
  const int ns{species.n_spec};
  const int nr{reaction.n_reac};
  const auto &stoi_f = reaction.stoi_f, &stoi_b{reaction.stoi_b};
  const auto mw = species.mw;
  for (int i = 0; i < ns; ++i) {
    real creation = 0;
    omega_d[i] = 0;
    for (int j = 0; j < nr; ++j) {
      creation += q2[j] * stoi_f(j, i) + q1[j] * stoi_b(j, i);
      omega_d[i] += q1[j] * stoi_f(j, i) + q2[j] * stoi_b(j, i);
    }
    creation *= 1e+3 * mw[i];         // Unit is kg/(m3*s)
    omega_d[i] *= 1e+3 * mw[i];       // Unit is kg/(m3*s)
    omega[i] = creation - omega_d[i]; // Unit is kg/(m3*s)
  }
}

std::vector<real> compute_chem_src_jacobian(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2) {
  const int ns{species.n_spec}, nr{reaction.n_reac};
  std::vector<real> jac(ns * ns, 0);
  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  for (int m = 0; m < ns; ++m) {
    for (int n = 0; n < ns; ++n) {
      real zz{0};
      if (rhoY[n] > 1e-30) {
        for (int r = 0; r < nr; ++r) {
          // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
          zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, n) * q1[r] - stoi_b(r, n) * q2[r]);
        }
        zz /= rhoY[n];
      }
      jac[m * ns + n] = zz * 1e+3 * species.mw[m]; // //1e+3=1e-3(MW)*1e+6(cm->m)
    }
  }
  return std::move(jac);
}

std::vector<real> compute_chem_src_jacobian_diagonal(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2) {
  const int ns{species.n_spec}, nr{reaction.n_reac};
  std::vector<real> jac_diag(ns, 0);
  const auto &stoi_f = reaction.stoi_f, &stoi_b = reaction.stoi_b;
  for (int m = 0; m < ns; ++m) {
    real zz{0};
    if (rhoY[m] > 1e-30) {
      for (int r = 0; r < nr; ++r) {
        // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
        zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, m) * q1[r] - stoi_b(r, m) * q2[r]);
      }
      zz /= rhoY[m];
    }
    jac_diag[m] = zz * 1e+3 * species.mw[m]; // //1e+3=1e-3(MW)*1e+6(cm->m)
  }
  return std::move(jac_diag);
}

void EPI(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega) {
  const int ns = species.n_spec;
  std::vector<real> lhs(ns * ns, 0);
  for (int m = 0; m < ns; ++m) {
    for (int n = 0; n < ns; ++n) {
      if (m == n) {
        lhs[m * ns + n] = 1.0 - dt * jac[m * ns + n];
      } else {
        lhs[m * ns + n] = -dt * jac[m * ns + n];
      }
    }
  }
  auto ipiv = lu_decomp(lhs.data(), ns);
  lu_to_solution(lhs.data(), omega.data(), ns, ipiv);
}

void DA(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega) {
  const int ns = species.n_spec;
  for (int l = 0; l < ns; ++l) {
    omega[l] /= 1 - dt * jac[l];
  }
}

std::vector<int> lu_decomp(real *lhs, int dim) {
  std::vector iPiv(dim, 0);
  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    iPiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }
  return std::move(iPiv);
}

void lu_to_solution(real *lhs, real *rhs, int dim, const std::vector<int> &ipiv) {
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = rhs[t];
      rhs[t] = rhs[m];
      rhs[m] = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
  }
  rhs[dim - 1] /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
    rhs[m] /= lhs[m * dim + m];
  }
}

int buildArnoldi(const Species &species, const std::vector<real> &rhoY, const real *f0, real *Q, real *H,
  int krylovMaxDim, int mechanism, real T, const Reaction &reaction, real rho, real E, int krylovMinDim) {
  const int ns = species.n_spec;
  const real norm_y = gxl::vec_norm(rhoY.data(), ns);
  const real eps = 1e-6 * (1 + norm_y);

  // v0 = f0 / ||f0||
  if (real norm_f0 = gxl::vec_norm<real>(f0, ns); norm_f0 < 1e-30) {
    // use unit vector to avoid zero basis
    for (int l = 0; l < ns; ++l) {
      Q[l] = 0;
    }
    Q[0] = 1.0;
    norm_f0 = 1.0;
  } else {
    for (int l = 0; l < ns; ++l) {
      Q[l] = f0[l] / norm_f0;
    }
  }
  int krylov_dim = 1;
  std::vector<real> q1(reaction.n_reac, 0), q2(reaction.n_reac, 0), omega_d(ns, 0), f_eps(ns, 0);
  for (int col = 0; col < krylovMaxDim; ++col) {
    // w = J * v_col
    // real y_eps[MAX_SPEC_NUMBER];
    std::vector<real> y_eps(ns, 0);
    std::vector<real> yk(ns, 0);
    real imw = 0;
    for (int l = 0; l < ns; ++l) {
      y_eps[l] = rhoY[l] + eps * Q[col * ns + l];
      yk[l] = y_eps[l] / rho;
      imw += yk[l] / species.mw[l];
    }
    real T_stage = update_t(T, imw, species, yk, E);
    // real f_eps[MAX_SPEC_NUMBER];
    compute_src(mechanism, T_stage, species, reaction, y_eps, q1, q2, omega_d, f_eps);
    real w[MAX_SPEC_NUMBER];
    for (int l = 0; l < ns; ++l) {
      w[l] = (f_eps[l] - f0[l]) / eps;
    }

    // Modified Gram-Schmidt
    for (int row = 0; row <= col; ++row) {
      const real hij = gxl::vec_dot(&Q[row * ns], w, ns);
      H[row * krylovMaxDim + col] = hij;
      for (int l = 0; l < ns; ++l) {
        w[l] -= hij * Q[row * ns + l];
      }
    }
    real h_next = gxl::vec_norm(w, ns);
    if (col + 1 < krylovMaxDim) {
      H[(col + 1) * krylovMaxDim + col] = h_next;
    }
    if (col + 1 < krylovMinDim) {
      if (h_next < 1e-30) h_next = 1e-30;
      for (int l = 0; l < ns; ++l) {
        Q[(col + 1) * ns + l] = w[l] / h_next;
      }
      krylov_dim = col + 2;
      continue;
    }
    if (h_next > 1e-30 && col + 1 < krylovMaxDim) {
      for (int l = 0; l < ns; ++l) {
        Q[(col + 1) * ns + l] = w[l] / h_next;
      }
      krylov_dim = col + 2;
    } else {
      krylov_dim = col + 1;
      break;
    }
  }
  return krylov_dim;
}
#endif
}

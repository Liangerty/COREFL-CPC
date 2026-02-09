#pragma once

#include "Parameter.h"
#include "ChemData.h"

namespace cfd {
void const_volume_reactor(Parameter &parameter);

#ifdef Combustion2Part

void compute_src(int mechanism, real t, const Species &species, const Reaction &reaction, const std::vector<real> &rhoY,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega);

void compute_gibbs_div_rt(real t, const Species &species, std::vector<real> &gibbs_rt);

real update_t(real T0, real imw, const Species &species, const std::vector<real> &yk, real E);

real update_t_with_h(real T0, real imw, const Species &species, const std::vector<real> &yk, real h);

void chemical_source_hardCoded1(real t, const Species &species, const std::vector<real> &c,
  std::vector<real> &q1, std::vector<real> &q2, std::vector<real> &omega_d, std::vector<real> &omega);

real arrhenius(real t, real A, real b, real Ea);

std::vector<real> forward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &c);

std::vector<real> backward_reaction_rate(real t, const Species &species, const Reaction &reaction,
  const std::vector<real> &kf, const std::vector<real> &c);

void rate_of_progress(const std::vector<real> &kf, const std::vector<real> &kb, const std::vector<real> &c,
  std::vector<real> &q, std::vector<real> &q1, std::vector<real> &q2, const Species &species,
  const Reaction &reaction);

void chemical_source(const std::vector<real> &q1, const std::vector<real> &q2, std::vector<real> &omega_d,
  std::vector<real> &omega, const Species &species, const Reaction &reaction);

std::vector<real> compute_chem_src_jacobian(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2);

std::vector<real> compute_chem_src_jacobian_diagonal(const std::vector<real> &rhoY, const Species &species,
  const Reaction &reaction, const std::vector<real> &q1, const std::vector<real> &q2);

void EPI(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega);

void DA(const std::vector<real> &jac, const Species &species, real dt, std::vector<real> &omega);

void solve_chem_system(real *lhs, real *rhs, int ns);

std::vector<int> lu_decomp(real *lhs, int dim);

void lu_to_solution(real *lhs, real *rhs, int dim, const std::vector<int> &piv);

int buildArnoldi(const Species &species, const std::vector<real> &rhoY, const real *f0, real *Q, real *H,
  int krylovMaxDim, int mechanism, real T, const Reaction &reaction, real rho, real E, int krylovMinDim);
#endif
}

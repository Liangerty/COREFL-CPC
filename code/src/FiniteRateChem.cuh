#pragma once

#include "DParameter.cuh"

namespace cfd {
struct DZone;

__global__ void finite_rate_chemistry(DZone *zone, const DParameter *param);

__global__ void finite_rate_chemistry_1(DZone *zone, const DParameter *param);

__device__ void finite_rate_chemistry(DZone *zone, int i, int j, int k, const DParameter *param);

__device__ void forward_reaction_rate(real t, real *kf, const real *concentration, const DParameter *param);

__device__ void forward_reaction_rate_1(real t, real *kf, const real *concentration);

__device__ real arrhenius(real t, real A, real b, real Ea);

__device__ void backward_reaction_rate(real t, const real *kf, const real *concentration, const DParameter *param,
  real *kb);

__device__ void backward_reaction_rate_1(real t, const real *kf, real *kb, const DParameter *param);

__device__ void rate_of_progress(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2,
  const DParameter *param);

__device__ void rate_of_progress_1(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2);

__device__ void chemical_source(const real *q1, const real *q2, real *omega_d, real *omega, const DParameter *param);

__device__ void chemical_source_1(const real *q1, const real *q2, real *omega_d, real *omega);

__device__ void
compute_chem_src_jacobian(DZone *zone, int i, int j, int k, const DParameter *param, const real *q1, const real *q2);

__device__ void
compute_chem_src_jacobian_diagonal(DZone *zone, int i, int j, int k, const DParameter *param, const real *omega_d);

__global__ void EPI(DZone *zone, int n_spec);

__device__ void EPI_for_dq0(DZone *zone, real diag, int i, int j, int k, int n_spec);

__device__ void EPI_for_dqk(DZone *zone, real diag, int i, int j, int k, const real *dq_total, int n_spec);

__device__ void solve_chem_system(real *lhs, DZone *zone, int i, int j, int k, int n_spec);

__device__ void solve_chem_system(real *lhs, real *rhs, int dim);

__global__ void DA(DZone *zone, int n_spec);
} // cfd

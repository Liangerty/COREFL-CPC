#pragma once

#include "Define.h"

namespace cfd {
class Block;

struct DZone;
struct DParameter;

class Parameter;

template<MixtureModel mix_model> void compute_convective_term_pv(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template<MixtureModel mix_model> __global__ void compute_convective_term_pv_1D(DZone *zone, int direction,
  int max_extent, DParameter *param);

template<MixtureModel mix_model> __device__ void reconstruction(real *pv, real *pv_l, real *pv_r, int idx_shared,
  DParameter *param);

template<MixtureModel mix_model> void compute_convective_term_weno(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template<MixtureModel mix_model> void compute_convective_term_weno_new(const Block &block, DZone *zone,
  DParameter *param, int n_var, const Parameter &parameter);

template<MixtureModel mix_model> void Roe_compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template<MixtureModel mix_model> __global__ void compute_entropy_fix_delta(DZone *zone, DParameter *param);

template<MixtureModel mix_model> __global__ void Roe_compute_inviscid_flux_1D(DZone *zone, int direction,
  int max_extent, DParameter *param);

template<MixtureModel mix_model> __global__ void compute_convective_term_weno_x(DZone *zone, DParameter *param);

template<MixtureModel mix_model> __global__ void compute_convective_term_weno_y(DZone *zone, DParameter *param);

template<MixtureModel mix_model> __global__ void compute_convective_term_weno_z(DZone *zone, DParameter *param);

__device__ void compute_flux(const real *Q, const DParameter *param, const real *metric, real jac, real *Fp, real *Fm,
  real p, real cc);

template<MixtureModel mix_model> __device__ void compute_weno_flux_ch(const real *cv, const real *p, DParameter *param,
  int tid, const real *metric, const real *jac, real *fc, int i_shared, real *Fp, real *Fm, bool if_shock);

template<MixtureModel mix_model>
__device__ void compute_weno_flux_ch(const real *cv, const real *p, DParameter *param, const real *metric,
  const real *jac, real *fc, int i_shared, const real *Fp, const real *Fm, bool if_shock);

__device__ void compute_flux(const real *Q, real p, real cc, const DParameter *param, const real *metric, real jac,
  real *Fp, real *Fm);

__device__ void compute_weno_flux_cp(const real *cv, DParameter *param, int tid, const real *metric, const real *jac,
  real *fc, int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st, bool if_shock);

__device__ void compute_weno_flux_cp(DParameter *param, const real *metric, const real *jac, real *fci, int i_shared,
  const real *Fp, const real *Fm, bool if_shock);

__device__ void positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param,
  int i_shared, real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac);

__device__ void positive_preserving_limiter_1(int dim, int n_var, const real *cv, int i_shared, const real *jac,
  real dt, real *fci, const real *metric, const real *cc, const real *Fp);

__device__ real WENO5(const real *vp, const real *vm, real eps, bool if_shock);

__device__ real WENO7(const real *vp, const real *vm, real eps, bool if_shock);

__device__ real WENO(const real *vp, const real *vm, real eps, bool if_shock, int weno_scheme_i);

__device__ real WENO5_new(const real *vp, const real *vm, real eps);

__device__ real WENO7_new(const real *vp, const real *vm, real eps);

template<MixtureModel mix_model> void compute_convective_term_hybrid_ud_weno(const Block &block, DZone *zone,
  DParameter *param, int n_var, const Parameter &parameter);

__device__ void hybrid_weno_part_cp(const real *pv, const real *rhoE, int i_shared, const DParameter *param,
  const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci);

template<MixtureModel mix_model> __device__ void hybrid_weno_part(const real *pv, const real *rhoE, int i_shared,
  const DParameter *param, const real *metric, const real *jac, const real *uk, const real *cGradK, real *fci);
} // cfd

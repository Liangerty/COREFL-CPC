#include "Reconstruction.cuh"
#include "Limiter.cuh"

namespace cfd {
__device__ void first_order_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var) {
  // The variables that can be reconstructed directly are density, u, v, w, p, Y_k, the number of which is
  // equal to the number of conservative variables(n_var).
  for (int l = 0; l < n_var; ++l) {
    pv_l[l] = pv[idx_shared * n_var + l];
    pv_r[l] = pv[(idx_shared + 1) * n_var + l];
  }
}

__device__ void
MUSCL_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var, int limiter) {
  static constexpr real kappa{1.0 / 3.0};
  // The variables that can be reconstructed directly are density, u, v, w, p, Y_k, the number of which is
  // equal to the number of conservative variables(n_var).
  for (int l = 0; l < n_var; ++l) {
    // \Delta_i = u_i - u_{i-1}; \Delta_{i+1} = u_{i+1} - u_i
    const real delta_i{pv[idx_shared * n_var + l] - pv[(idx_shared - 1) * n_var + l]};
    const real delta_i1{pv[(idx_shared + 1) * n_var + l] - pv[idx_shared * n_var + l]};
    const real delta_i2{pv[(idx_shared + 2) * n_var + l] - pv[(idx_shared + 1) * n_var + l]};

    const real delta_neg_l = apply_limiter<0, 1>(limiter, delta_i, delta_i1);
    const real delta_pos_l = apply_limiter<0, 1>(limiter, delta_i1, delta_i);
    const real delta_neg_r = apply_limiter<0, 1>(limiter, delta_i1, delta_i2);
    const real delta_pos_r = apply_limiter<0, 1>(limiter, delta_i2, delta_i1);

    pv_l[l] = pv[idx_shared * n_var + l] + 0.25 * ((1 - kappa) * delta_neg_l + (1 + kappa) * delta_pos_l);
    pv_r[l] = pv[(idx_shared + 1) * n_var + l] - 0.25 * ((1 - kappa) * delta_pos_r + (1 + kappa) * delta_neg_r);
  }
}

__device__ void
NND2_reconstruct(const real *pv, real *pv_l, real *pv_r, int idx_shared, int n_var, int limiter) {
  for (int l = 0; l < n_var; ++l) {
    // \Delta_i = u_i - u_{i-1}; \Delta_{i+1} = u_{i+1} - u_i
    const real delta_i{pv[idx_shared * n_var + l] - pv[(idx_shared - 1) * n_var + l]};
    const real delta_i1{pv[(idx_shared + 1) * n_var + l] - pv[idx_shared * n_var + l]};
    const real delta_i2{pv[(idx_shared + 2) * n_var + l] - pv[(idx_shared + 1) * n_var + l]};

    pv_l[l] = pv[idx_shared * n_var + l] + 0.5 * apply_limiter<0, 1>(limiter, delta_i, delta_i1);
    pv_r[l] = pv[(idx_shared + 1) * n_var + l] - 0.5 * apply_limiter<0, 1>(limiter, delta_i1, delta_i2);
  }
}
} // cfd
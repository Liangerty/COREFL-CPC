#pragma once

#include "Define.h"
#include "Thermo.cuh"
#include "Constants.h"
#include "DParameter.cuh"
#include "Field.h"

namespace cfd {
struct DParameter;

template<MixtureModel mix_model>
void compute_viscous_flux(const Mesh &mesh, std::vector<Field> &field, DParameter *param, const Parameter &parameter);

__global__ void compute_dFv_dx(DZone *zone, const DParameter *param);

__global__ void compute_dGv_dy(DZone *zone, const DParameter *param);

__global__ void compute_dHv_dz(DZone *zone, const DParameter *param);

template<int ORDER = 8>
__device__ real d_dXi(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nx,
  int phyBoundLeft, int phyBoundRight);

template<int ORDER = 8>
__device__ real d_dEta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int ny,
  int phyBoundLeft, int phyBoundRight);

template<int ORDER = 8>
__device__ real d_dZeta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nz,
  int phyBoundLeft, int phyBoundRight);

template<MixtureModel mix_model, int ORDER = 8>
__global__ void compute_viscous_flux_collocated(DZone *zone, const DParameter *param);

template<int ORDER = 8>
__global__ void compute_viscous_flux_collocated_scalar(DZone *zone, const DParameter *param);

template<int ORDER = 8> __global__ void compute_viscous_flux_derivative(DZone *zone, const DParameter *param);

template<MixtureModel mix_model>
__global__ void compute_fv_2nd_order(DZone *zone, DParameter *param) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const auto &metric = zone->metric;

  const real xi_x = 0.5 * (metric(i, j, k, 0) + metric(i + 1, j, k, 0));
  const real xi_y = 0.5 * (metric(i, j, k, 1) + metric(i + 1, j, k, 1));
  const real xi_z = 0.5 * (metric(i, j, k, 2) + metric(i + 1, j, k, 2));
  const real eta_x = 0.5 * (metric(i, j, k, 3) + metric(i + 1, j, k, 3));
  const real eta_y = 0.5 * (metric(i, j, k, 4) + metric(i + 1, j, k, 4));
  const real eta_z = 0.5 * (metric(i, j, k, 5) + metric(i + 1, j, k, 5));
  const real zeta_x = 0.5 * (metric(i, j, k, 6) + metric(i + 1, j, k, 6));
  const real zeta_y = 0.5 * (metric(i, j, k, 7) + metric(i + 1, j, k, 7));
  const real zeta_z = 0.5 * (metric(i, j, k, 8) + metric(i + 1, j, k, 8));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = pv(i + 1, j, k, 1) - pv(i, j, k, 1);
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i + 1, j - 1, k, 1));
  const real u_zeta =
      0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i + 1, j, k + 1, 1) - pv(i + 1, j, k - 1, 1));
  const real v_xi = pv(i + 1, j, k, 2) - pv(i, j, k, 2);
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i + 1, j - 1, k, 2));
  const real v_zeta =
      0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i + 1, j, k + 1, 2) - pv(i + 1, j, k - 1, 2));
  const real w_xi = pv(i + 1, j, k, 3) - pv(i, j, k, 3);
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i + 1, j - 1, k, 3));
  const real w_zeta =
      0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i + 1, j, k + 1, 3) - pv(i + 1, j, k - 1, 3));
  const real t_xi = pv(i + 1, j, k, 5) - pv(i, j, k, 5);
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i + 1, j - 1, k, 5));
  const real t_zeta =
      0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i + 1, j, k + 1, 5) - pv(i + 1, j, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i + 1, j, k));

  // Compute the viscous stress
  real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  const real xi_x_div_jac =
      0.5 * (metric(i, j, k, 0) * zone->jac(i, j, k) + metric(i + 1, j, k, 0) * zone->jac(i + 1, j, k));
  const real xi_y_div_jac =
      0.5 * (metric(i, j, k, 1) * zone->jac(i, j, k) + metric(i + 1, j, k, 1) * zone->jac(i + 1, j, k));
  const real xi_z_div_jac =
      0.5 * (metric(i, j, k, 2) * zone->jac(i, j, k) + metric(i + 1, j, k, 2) * zone->jac(i + 1, j, k));

  auto &fv = zone->vis_flux;
  fv(i, j, k, 0) = xi_x_div_jac * tau_xx + xi_y_div_jac * tau_xy + xi_z_div_jac * tau_xz;
  fv(i, j, k, 1) = xi_x_div_jac * tau_xy + xi_y_div_jac * tau_yy + xi_z_div_jac * tau_yz;
  fv(i, j, k, 2) = xi_x_div_jac * tau_xz + xi_y_div_jac * tau_yz + xi_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i + 1, j, k, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i + 1, j, k, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i + 1, j, k, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity{0};
  if constexpr (mix_model != MixtureModel::Air) {
    conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i + 1, j, k));
  } else {
    constexpr real cp{gamma_air * R_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp;
  }

  fv(i, j, k, 3) = um * fv(i, j, k, 0) + vm * fv(i, j, k, 1) + wm * fv(i, j, k, 2) +
                   conductivity * (xi_x_div_jac * t_x + xi_y_div_jac * t_y + xi_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate,
    // this part will compute the viscous term of species equations and energy eqn.
    // If we are solving the flamelet model, this part only contributes to the energy eqn.
    const int n_spec{param->n_spec};
    const auto &y = zone->sv;

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradXi_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i + 1, j, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i + 1, j, k, l));

      const real y_xi = y(i + 1, j, k, l) - y(i, j, k, l);
      const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i + 1, j + 1, k, l) - y(i + 1, j - 1, k, l));
      const real y_zeta =
          0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i + 1, j, k + 1, l) - y(i + 1, j, k - 1, l));

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradXi_cdot_GradY = xi_x_div_jac * y_x + xi_y_div_jac * y_y + xi_z_div_jac * y_z;
      diffusion_driven_force[l] = GradXi_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradXi_cdot_GradY;

      // Term 2, the gradient of molecular weights,
      // which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradXi_cdot_GradY_over_wl += GradXi_cdot_GradY * param->imw[l];
      mw_tot += yk[l] * param->imw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradXi_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk,
    // which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi{pv(i + 1, j, k, 4) - pv(i, j, k, 4)};
      const real p_eta{
        0.25 * (pv(i, j + 1, k, 4) - pv(i, j - 1, k, 4) + pv(i + 1, j + 1, k, 4) - pv(i + 1, j - 1, k, 4))
      };
      const real p_zeta{
        0.25 * (pv(i, j, k + 1, 4) - pv(i, j, k - 1, 4) + pv(i + 1, j, k + 1, 4) - pv(i + 1, j, k - 1, 4))
      };

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradXi_cdot_gradP_over_p{
        (xi_x_div_jac * p_x + xi_y_div_jac * p_y + xi_z_div_jac * p_z) / (0.5 * (pv(i + 1, j, k, 4) + pv(i, j, k, 4)))
      };

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot * param->imw[l] - 1) * yk[l] * gradXi_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot * param->imw[l] - 1) * yk[l] * gradXi_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i + 1, j, k, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
        diffusivity[l] * (diffusion_driven_force[l] - mw_tot * yk[l] * sum_GradXi_cdot_GradY_over_wl)
        - yk[l] * CorrectionVelocityTerm
      };
      fv(i, j, k, 4 + l) = diffusion_flux;
      // Add the influence of species diffusion on total energy
      fv(i, j, k, 3) += h[l] * diffusion_flux;
    }
  }

  if (param->n_ps > 0) {
    const auto &sv = zone->sv;

    for (int l = 0; l < param->n_ps; ++l) {
      const int ls = param->i_ps + l, lc = param->i_ps_cv + l;
      // First, compute the passive scalar gradient
      const real ps_xi = sv(i + 1, j, k, ls) - sv(i, j, k, ls);
      const real ps_eta =
          0.25 * (sv(i, j + 1, k, ls) - sv(i, j - 1, k, ls) + sv(i + 1, j + 1, k, ls) - sv(i + 1, j - 1, k, ls));
      const real ps_zeta =
          0.25 * (sv(i, j, k + 1, ls) - sv(i, j, k - 1, ls) + sv(i + 1, j, k + 1, ls) - sv(i + 1, j, k - 1, ls));

      const real ps_x = ps_xi * xi_x + ps_eta * eta_x + ps_zeta * zeta_x;
      const real ps_y = ps_xi * xi_y + ps_eta * eta_y + ps_zeta * zeta_y;
      const real ps_z = ps_xi * xi_z + ps_eta * eta_z + ps_zeta * zeta_z;

      const real rhoD{mul / param->sc_ps[l]};
      fv(i, j, k, lc - 1) = rhoD * (xi_x_div_jac * ps_x + xi_y_div_jac * ps_y + xi_z_div_jac * ps_z);
    }
  }
}

template<MixtureModel mix_model>
__global__ void compute_gv_2nd_order(DZone *zone, DParameter *param) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const auto &metric = zone->metric;

  const real xi_x = 0.5 * (metric(i, j, k, 0) + metric(i, j + 1, k, 0));
  const real xi_y = 0.5 * (metric(i, j, k, 1) + metric(i, j + 1, k, 1));
  const real xi_z = 0.5 * (metric(i, j, k, 2) + metric(i, j + 1, k, 2));
  const real eta_x = 0.5 * (metric(i, j, k, 3) + metric(i, j + 1, k, 3));
  const real eta_y = 0.5 * (metric(i, j, k, 4) + metric(i, j + 1, k, 4));
  const real eta_z = 0.5 * (metric(i, j, k, 5) + metric(i, j + 1, k, 5));
  const real zeta_x = 0.5 * (metric(i, j, k, 6) + metric(i, j + 1, k, 6));
  const real zeta_y = 0.5 * (metric(i, j, k, 7) + metric(i, j + 1, k, 7));
  const real zeta_z = 0.5 * (metric(i, j, k, 8) + metric(i, j + 1, k, 8));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i - 1, j + 1, k, 1));
  const real u_eta = pv(i, j + 1, k, 1) - pv(i, j, k, 1);
  const real u_zeta =
      0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j + 1, k - 1, 1));
  const real v_xi = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i - 1, j + 1, k, 2));
  const real v_eta = pv(i, j + 1, k, 2) - pv(i, j, k, 2);
  const real v_zeta =
      0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j + 1, k - 1, 2));
  const real w_xi = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i - 1, j + 1, k, 3));
  const real w_eta = pv(i, j + 1, k, 3) - pv(i, j, k, 3);
  const real w_zeta =
      0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j + 1, k - 1, 3));
  const real t_xi = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i - 1, j + 1, k, 5));
  const real t_eta = pv(i, j + 1, k, 5) - pv(i, j, k, 5);
  const real t_zeta =
      0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j + 1, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i, j + 1, k));

  // Compute the viscous stress
  real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  const real eta_x_div_jac =
      0.5 * (metric(i, j, k, 3) * zone->jac(i, j, k) + metric(i, j + 1, k, 3) * zone->jac(i, j + 1, k));
  const real eta_y_div_jac =
      0.5 * (metric(i, j, k, 4) * zone->jac(i, j, k) + metric(i, j + 1, k, 4) * zone->jac(i, j + 1, k));
  const real eta_z_div_jac =
      0.5 * (metric(i, j, k, 5) * zone->jac(i, j, k) + metric(i, j + 1, k, 5) * zone->jac(i, j + 1, k));

  auto &gv = zone->vis_flux;
  gv(i, j, k, 0) = eta_x_div_jac * tau_xx + eta_y_div_jac * tau_xy + eta_z_div_jac * tau_xz;
  gv(i, j, k, 1) = eta_x_div_jac * tau_xy + eta_y_div_jac * tau_yy + eta_z_div_jac * tau_yz;
  gv(i, j, k, 2) = eta_x_div_jac * tau_xz + eta_y_div_jac * tau_yz + eta_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i, j + 1, k, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i, j + 1, k, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i, j + 1, k, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity{0};
  if constexpr (mix_model != MixtureModel::Air) {
    conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i, j + 1, k));
  } else {
    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp;
  }

  gv(i, j, k, 3) = um * gv(i, j, k, 0) + vm * gv(i, j, k, 1) + wm * gv(i, j, k, 2) +
                   conductivity * (eta_x_div_jac * t_x + eta_y_div_jac * t_y + eta_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate,
    // this part will compute the viscous term of species eqns and energy eqn.
    // If we are solving the flamelet model, this part only contributes to the energy eqn.
    const int n_spec{param->n_spec};
    const auto &y = zone->sv;

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradEta_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i, j + 1, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j + 1, k, l));

      const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j + 1, k, l) - y(i - 1, j + 1, k, l));
      const real y_eta = y(i, j + 1, k, l) - y(i, j, k, l);
      const real y_zeta =
          0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i, j + 1, k + 1, l) - y(i, j + 1, k - 1, l));

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradEta_cdot_GradY = eta_x_div_jac * y_x + eta_y_div_jac * y_y + eta_z_div_jac * y_z;
      diffusion_driven_force[l] = GradEta_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradEta_cdot_GradY;

      // Term 2, the gradient of molecular weights,
      // which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradEta_cdot_GradY_over_wl += GradEta_cdot_GradY * param->imw[l];
      mw_tot += yk[l] * param->imw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradEta_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk,
    // which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi =
          0.25 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4) + pv(i + 1, j + 1, k, 4) - pv(i - 1, j + 1, k, 4));
      const real p_eta = pv(i, j + 1, k, 4) - pv(i, j, k, 4);
      const real p_zeta =
          0.25 * (pv(i, j, k + 1, 4) - pv(i, j, k - 1, 4) + pv(i, j + 1, k + 1, 4) - pv(i, j + 1, k - 1, 4));

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradEta_cdot_gradP_over_p{
        (eta_x_div_jac * p_x + eta_y_div_jac * p_y + eta_z_div_jac * p_z) / (
          0.5 * (pv(i, j, k, 4) + pv(i, j + 1, k, 4)))
      };

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot * param->imw[l] - 1) * yk[l] * gradEta_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot * param->imw[l] - 1) * yk[l] * gradEta_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j + 1, k, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
        diffusivity[l] * (diffusion_driven_force[l] - mw_tot * yk[l] * sum_GradEta_cdot_GradY_over_wl)
        - yk[l] * CorrectionVelocityTerm
      };
      gv(i, j, k, 4 + l) = diffusion_flux;
      // Add the influence of species diffusion on total energy
      gv(i, j, k, 3) += h[l] * diffusion_flux;
    }
  }

  if (param->n_ps > 0) {
    const auto &sv = zone->sv;

    for (int l = 0; l < param->n_ps; ++l) {
      const int ls = param->i_ps + l, lc = param->i_ps_cv + l;
      // First, compute the passive scalar gradient
      const real ps_xi = 0.25 * (sv(i + 1, j, k, ls) - sv(i - 1, j, k, ls) + sv(i + 1, j + 1, k, ls) -
                                 sv(i - 1, j + 1, k, ls));
      const real ps_eta = sv(i, j + 1, k, ls) - sv(i, j, k, ls);
      const real ps_zeta =
          0.25 * (sv(i, j, k + 1, ls) - sv(i, j, k - 1, ls) + sv(i, j + 1, k + 1, ls) - sv(i, j + 1, k - 1, ls));

      const real ps_x = ps_xi * xi_x + ps_eta * eta_x + ps_zeta * zeta_x;
      const real ps_y = ps_xi * xi_y + ps_eta * eta_y + ps_zeta * zeta_y;
      const real ps_z = ps_xi * xi_z + ps_eta * eta_z + ps_zeta * zeta_z;

      const real rhoD{mul / param->sc_ps[l]};
      gv(i, j, k, lc - 1) = rhoD * (eta_x_div_jac * ps_x + eta_y_div_jac * ps_y + eta_z_div_jac * ps_z);
    }
  }
}

template<MixtureModel mix_model>
__global__ void compute_hv_2nd_order(DZone *zone, DParameter *param) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const auto &metric = zone->metric;

  const real xi_x = 0.5 * (metric(i, j, k, 0) + metric(i, j, k + 1, 0));
  const real xi_y = 0.5 * (metric(i, j, k, 1) + metric(i, j, k + 1, 1));
  const real xi_z = 0.5 * (metric(i, j, k, 2) + metric(i, j, k + 1, 2));
  const real eta_x = 0.5 * (metric(i, j, k, 3) + metric(i, j, k + 1, 3));
  const real eta_y = 0.5 * (metric(i, j, k, 4) + metric(i, j, k + 1, 4));
  const real eta_z = 0.5 * (metric(i, j, k, 5) + metric(i, j, k + 1, 5));
  const real zeta_x = 0.5 * (metric(i, j, k, 6) + metric(i, j, k + 1, 6));
  const real zeta_y = 0.5 * (metric(i, j, k, 7) + metric(i, j, k + 1, 7));
  const real zeta_z = 0.5 * (metric(i, j, k, 8) + metric(i, j, k + 1, 8));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j, k + 1, 1) - pv(i - 1, j, k + 1, 1));
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j - 1, k + 1, 1));
  const real u_zeta = pv(i, j, k + 1, 1) - pv(i, j, k, 1);
  const real v_xi = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j, k + 1, 2) - pv(i - 1, j, k + 1, 2));
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j - 1, k + 1, 2));
  const real v_zeta = pv(i, j, k + 1, 2) - pv(i, j, k, 2);
  const real w_xi = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j, k + 1, 3) - pv(i - 1, j, k + 1, 3));
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j - 1, k + 1, 3));
  const real w_zeta = pv(i, j, k + 1, 3) - pv(i, j, k, 3);
  const real t_xi = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j, k + 1, 5) - pv(i - 1, j, k + 1, 5));
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j - 1, k + 1, 5));
  const real t_zeta = pv(i, j, k + 1, 5) - pv(i, j, k, 5);

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i, j, k + 1));

  // Compute the viscous stress
  real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  const real zeta_x_div_jac =
      0.5 * (metric(i, j, k, 6) * zone->jac(i, j, k) + metric(i, j, k + 1, 6) * zone->jac(i, j, k + 1));
  const real zeta_y_div_jac =
      0.5 * (metric(i, j, k, 7) * zone->jac(i, j, k) + metric(i, j, k + 1, 7) * zone->jac(i, j, k + 1));
  const real zeta_z_div_jac =
      0.5 * (metric(i, j, k, 8) * zone->jac(i, j, k) + metric(i, j, k + 1, 8) * zone->jac(i, j, k + 1));

  auto &hv = zone->vis_flux;
  hv(i, j, k, 0) = zeta_x_div_jac * tau_xx + zeta_y_div_jac * tau_xy + zeta_z_div_jac * tau_xz;
  hv(i, j, k, 1) = zeta_x_div_jac * tau_xy + zeta_y_div_jac * tau_yy + zeta_z_div_jac * tau_yz;
  hv(i, j, k, 2) = zeta_x_div_jac * tau_xz + zeta_y_div_jac * tau_yz + zeta_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i, j, k + 1, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i, j, k + 1, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i, j, k + 1, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity{0};
  if constexpr (mix_model != MixtureModel::Air) {
    conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i, j, k + 1));
  } else {
    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp;
  }

  hv(i, j, k, 3) = um * hv(i, j, k, 0) + vm * hv(i, j, k, 1) + wm * hv(i, j, k, 2) +
                   conductivity * (zeta_x_div_jac * t_x + zeta_y_div_jac * t_y + zeta_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate,
    // this part will compute the viscous term of species eqns and energy eqn.
    // If we are solving the flamelet model, this part only contributes to the energy eqn.
    const int n_spec{param->n_spec};
    const auto &y = zone->sv;

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradZeta_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i, j, k + 1, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j, k + 1, l));

      const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j, k + 1, l) - y(i - 1, j, k + 1, l));
      const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i, j + 1, k + 1, l) - y(i, j - 1, k + 1, l));
      const real y_zeta = y(i, j, k + 1, l) - y(i, j, k, l);

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradZeta_cdot_GradY = zeta_x_div_jac * y_x + zeta_y_div_jac * y_y + zeta_z_div_jac * y_z;
      diffusion_driven_force[l] = GradZeta_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradZeta_cdot_GradY;

      // Term 2, the gradient of molecular weights,
      // which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradZeta_cdot_GradY_over_wl += GradZeta_cdot_GradY * param->imw[l];
      mw_tot += yk[l] * param->imw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradZeta_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk,
    // which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi =
          0.25 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4) + pv(i + 1, j, k + 1, 4) - pv(i - 1, j, k + 1, 4));
      const real p_eta =
          0.25 * (pv(i, j + 1, k, 4) - pv(i, j - 1, k, 4) + pv(i, j + 1, k + 1, 4) - pv(i, j - 1, k + 1, 4));
      const real p_zeta = pv(i, j, k + 1, 4) - pv(i, j, k, 4);

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradZeta_cdot_gradP_over_p{
        (zeta_x_div_jac * p_x + zeta_y_div_jac * p_y + zeta_z_div_jac * p_z) /
        (0.5 * (pv(i, j, k, 4) + pv(i, j, k + 1, 4)))
      };

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot * param->imw[l] - 1) * yk[l] * gradZeta_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot * param->imw[l] - 1) * yk[l] * gradZeta_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j, k + 1, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
        diffusivity[l] * (diffusion_driven_force[l] - mw_tot * yk[l] * sum_GradZeta_cdot_GradY_over_wl)
        - yk[l] * CorrectionVelocityTerm
      };
      hv(i, j, k, 4 + l) = diffusion_flux;
      // Add the influence of species diffusion on total energy
      hv(i, j, k, 3) += h[l] * diffusion_flux;
    }
  }

  if (param->n_ps > 0) {
    const auto &sv = zone->sv;

    for (int l = 0; l < param->n_ps; ++l) {
      const int ls = param->i_ps + l, lc = param->i_ps_cv + l;
      // First, compute the passive scalar gradient
      const real ps_xi =
          0.25 * (sv(i + 1, j, k, ls) - sv(i - 1, j, k, ls) + sv(i + 1, j, k + 1, ls) - sv(i - 1, j, k + 1, ls));
      const real ps_eta =
          0.25 * (sv(i, j + 1, k, ls) - sv(i, j - 1, k, ls) + sv(i, j + 1, k + 1, ls) - sv(i, j - 1, k + 1, ls));
      const real ps_zeta = sv(i, j, k + 1, ls) - sv(i, j, k, ls);

      const real ps_x = ps_xi * xi_x + ps_eta * eta_x + ps_zeta * zeta_x;
      const real ps_y = ps_xi * xi_y + ps_eta * eta_y + ps_zeta * zeta_y;
      const real ps_z = ps_xi * xi_z + ps_eta * eta_z + ps_zeta * zeta_z;

      const real rhoD{mul / param->sc_ps[l]};
      hv(i, j, k, lc - 1) = rhoD * (zeta_x_div_jac * ps_x + zeta_y_div_jac * ps_y + zeta_z_div_jac * ps_z);
    }
  }
}
} // cfd

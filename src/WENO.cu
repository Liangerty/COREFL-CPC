#include "InviscidScheme.cuh"
#include "Constants.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Field.h"

namespace cfd {
static constexpr real eps_weno = 1e-40;

__device__ void weno_ch_air_x(const DParameter *param, const real **shared_mem, int i_shared, real kx, real ky, real kz,
  real eps_ref, bool if_shock, real *fc, int idx3d, int sz3d) {
  constexpr int nx_block = 64;
  const int n_var = param->n_var;

  auto Q = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(shared_mem);
  // rho, rhoU, rhoV, rhoW, rhoE, rhoY1, rhoY2, ...
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(shared_mem + n_var * (nx_block + 2 * 4 - 1));

  // Compute the Roe average state at the interface
  real rho_l = Q[0][i_shared], rho_r = Q[0][i_shared + 1];
  real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (rho_l + temp1)};
  const real rrc{1 / (temp1 + rho_r)};
  const real um{rlc * Q[1][i_shared] + rrc * Q[1][i_shared + 1]};
  const real vm{rlc * Q[2][i_shared] + rrc * Q[2][i_shared + 1]};
  const real wm{rlc * Q[3][i_shared] + rrc * Q[3][i_shared + 1]};
  real temp3 = (rlc * F[4][i_shared] + rrc * F[4][i_shared + 1]) / R_air; // temp3 = T
  const real cm = sqrt(gamma_air * R_air * temp3);
  constexpr real gm1{gamma_air - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  // real max_lambda = abs(kx * um + ky * vm + kz * wm);
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  // max_lambda += cm / temp1;
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
  if (abs(kx) > 0.8) {
    nx = -kz;
    nz = kx;
  } else {
    ny = kz;
    nz = -ky;
  }
  temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
  nx *= temp1;
  ny *= temp1;
  nz *= temp1;
  qx = ky * nz - kz * ny;
  qy = kz * nx - kx * nz;
  qz = kx * ny - ky * nx;
  temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
  qx *= temp1;
  qy *= temp1;
  qz *= temp1;
  const real Un = nx * um + ny * vm + nz * wm;
  const real Uq = qx * um + qy * vm + qz * wm;

  real temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
  // Local Lax-Friedrichs flux splitting
  temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  real fChar[5];
  if (param->inviscid_scheme == 72) {
    const int baseP = i_shared - 3;

    // Find the max spectral radius
    real max_lambda = 0;
    for (int m = 0; m < 8; ++m) {
      if (real lam = abs(F[0][baseP + m]) + F[5][baseP + m]; lam > max_lambda) {
        // if (real lam = F[5][baseP + m]; lam > max_lambda) {
        max_lambda = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[7] = {}, vMinus[7] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 3;
      real LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                 + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      real LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
      // vPlus[0] = 0.5 * F[6][iP] * (LFm + abs(F[0][iP] + F[5][iP]) * LQm);
      for (int m = 1; m < 7; ++m) {
        iP = i_shared - 3 + m;
        LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
              + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
        LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
        // vPlus[m] = 0.5 * F[6][iP] * (LFm + abs(F[0][iP] + F[5][iP]) * LQm);
        // vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - abs(F[0][iP] + F[5][iP]) * LQm);
        vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      }
      iP = i_shared - 3 + 7;
      LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
            + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      // vMinus[6] = 0.5 * F[6][iP] * (LFm - abs(F[0][iP] + F[5][iP]) * LQm);
      vMinus[6] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
    }
  } else if (param->inviscid_scheme == 52) {
    const int baseP = i_shared - 2;

    // Find the max spectral radius
    real max_lambda = 0;
    for (int m = 0; m < 6; ++m) {
      if (real lam = abs(F[0][baseP + m]) + F[5][baseP + m]; lam > max_lambda) {
      // if (real lam = F[5][baseP + m]; lam > max_lambda) {
        max_lambda = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[5] = {}, vMinus[5] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 2;
      real LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                 + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      real LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
      for (int m = 1; m < 5; ++m) {
        iP = i_shared - 2 + m;
        LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
              + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
        LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
        vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      }
      iP = i_shared + 3;
      LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
            + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      vMinus[4] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      fChar[l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
    }
  }
  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fc[idx3d] = temp1;
  fc[idx3d + 1 * sz3d] = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
  fc[idx3d + 2 * sz3d] = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
  fc[idx3d + 3 * sz3d] = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

  // temp2 is Roe averaged enthalpy
  temp2 = rlc * (Q[4][i_shared] + F[4][i_shared]) + rrc * (Q[4][i_shared + 1] + F[4][i_shared + 1]);
  fc[idx3d + 4 * sz3d] = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];
}

__device__ void weno_ch_air_x_piro(const DParameter *param, const real **shared_mem, int i_shared, real kx, real ky,
  real kz,
  real eps_ref, bool if_shock, real *fc, int idx3d, int sz3d) {
  constexpr int nx_block = 64;
  const int n_var = param->n_var;

  auto Q = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(shared_mem);
  // rho, rhoU, rhoV, rhoW, rhoE, rhoY1, rhoY2, ...
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(shared_mem + n_var * (nx_block + 2 * 4 - 1));

  // Compute the Roe average state at the interface
  real rho_l = Q[0][i_shared], rho_r = Q[0][i_shared + 1];
  real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (rho_l + temp1)};
  const real rrc{1 / (temp1 + rho_r)};
  const real um{rlc * Q[1][i_shared] + rrc * Q[1][i_shared + 1]};
  const real vm{rlc * Q[2][i_shared] + rrc * Q[2][i_shared + 1]};
  const real wm{rlc * Q[3][i_shared] + rrc * Q[3][i_shared + 1]};
  real temp3 = (rlc * F[4][i_shared] + rrc * F[4][i_shared + 1]) / R_air; // temp3 = T
  const real cm = sqrt(gamma_air * R_air * temp3);
  constexpr real gm1{gamma_air - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  // real max_lambda = abs(kx * um + ky * vm + kz * wm);
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  // max_lambda += cm / temp1;
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
  if (abs(kx) > 0.8) {
    nx = -kz;
    nz = kx;
  } else {
    ny = kz;
    nz = -ky;
  }
  temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
  nx *= temp1;
  ny *= temp1;
  nz *= temp1;
  qx = ky * nz - kz * ny;
  qy = kz * nx - kx * nz;
  qz = kx * ny - ky * nx;
  temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
  qx *= temp1;
  qy *= temp1;
  qz *= temp1;
  const real Un = nx * um + ny * vm + nz * wm;
  const real Uq = qx * um + qy * vm + qz * wm;

  real temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
  // Local Lax-Friedrichs flux splitting
  temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  real fChar[5];
  if (param->inviscid_scheme == 72) {
    const int baseP = i_shared - 3;

    // Find the max spectral radius
    real max_lambda2 = 0;
    for (int m = 0; m < 8; ++m) {
      if (real lam = abs(F[0][baseP + m]); lam > max_lambda2) {
        max_lambda2 = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }
      real max_lambda = max_lambda2;
      if (l == 0) {
        max_lambda = 0;
        for (int m = 0; m < 8; ++m) {
          if (real lam = abs(F[0][baseP + m] - F[5][baseP + m]); lam > max_lambda) {
            max_lambda = lam;
          }
        }
      } else if (l == 4) {
        max_lambda = 0;
        for (int m = 0; m < 8; ++m) {
          if (real lam = abs(F[0][baseP + m] + F[5][baseP + m]); lam > max_lambda) {
            max_lambda = lam;
          }
        }
      }

      real vPlus[7] = {}, vMinus[7] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 3;
      real LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                 + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      real LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
      for (int m = 1; m < 7; ++m) {
        iP = i_shared - 3 + m;
        LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
              + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
        LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
        vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      }
      iP = i_shared - 3 + 7;
      LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
            + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]);
      LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP];
      vMinus[6] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
      fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
    }
  }
  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fc[idx3d] = temp1;
  fc[idx3d + 1 * sz3d] = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
  fc[idx3d + 2 * sz3d] = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
  fc[idx3d + 3 * sz3d] = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

  // temp2 is Roe averaged enthalpy
  temp2 = rlc * (Q[4][i_shared] + F[4][i_shared]) + rrc * (Q[4][i_shared + 1] + F[4][i_shared + 1]);
  fc[idx3d + 4 * sz3d] = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];
}

__device__ void weno_ch_air_yz(const DParameter *param, const real **shared_mem, int i_shared, real kx, real ky,
  real kz, real eps_ref, bool if_shock, real *fc, int idx3d, int sz3d) {
  constexpr int nyy = 32;
  const int n_var = param->n_var;
  const auto tx = static_cast<int>(threadIdx.x);

  auto Q = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(shared_mem);
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(shared_mem + n_var * (nyy + 2 * 4 - 1) * 4);

  // Compute the Roe average state at the interface
  real rho_l = Q[0][i_shared][tx], rho_r = Q[0][i_shared + 1][tx];
  real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (rho_l + temp1)};
  const real rrc{1 / (temp1 + rho_r)};
  const real um{rlc * Q[1][i_shared][tx] + rrc * Q[1][i_shared + 1][tx]};
  const real vm{rlc * Q[2][i_shared][tx] + rrc * Q[2][i_shared + 1][tx]};
  const real wm{rlc * Q[3][i_shared][tx] + rrc * Q[3][i_shared + 1][tx]};
  real temp3 = (rlc * F[4][i_shared][tx] + rrc * F[4][i_shared + 1][tx]) / R_air; // temp3 = T
  const real cm = sqrt(gamma_air * R_air * temp3);
  const real gm1{gamma_air - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  // real max_lambda = abs(kx * um + ky * vm + kz * wm);
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  // max_lambda += cm / temp1;
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
  if (abs(kx) > 0.8) {
    nx = -kz;
    nz = kx;
  } else {
    ny = kz;
    nz = -ky;
  }
  temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
  nx *= temp1;
  ny *= temp1;
  nz *= temp1;
  qx = ky * nz - kz * ny;
  qy = kz * nx - kx * nz;
  qz = kx * ny - ky * nx;
  temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
  qx *= temp1;
  qy *= temp1;
  qz *= temp1;
  const real Un = nx * um + ny * vm + nz * wm;
  const real Uq = qx * um + qy * vm + qz * wm;

  real temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
  // Local Lax-Friedrichs flux splitting
  temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  real fChar[5 + MAX_SPEC_NUMBER];
  if (param->inviscid_scheme == 72) {
    const int baseP = i_shared - 3;

    // Find the max spectral radius
    real max_lambda = 0;
    for (int m = 0; m < 8; ++m) {
      if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
        // if (real lam = F[5][baseP + m][tx]; lam > max_lambda) {
        max_lambda = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[7] = {}, vMinus[7] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 3;
      real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx]
                 + L[3] * F[3][iP][tx] + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
                 + L[4] * Q[4][iP][tx];
      vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
      // vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
      for (int m = 1; m < 7; ++m) {
        iP = i_shared - 3 + m;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
              + L[4] * Q[4][iP][tx];
        // vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
        // vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
        vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      }
      iP = i_shared - 3 + 7;
      LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
            + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
            + L[4] * Q[4][iP][tx];
      // vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
      vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
    }
  } else if (param->inviscid_scheme == 52) {
    const int baseP = i_shared - 2;

    // Find the max spectral radius
    real max_lambda = 0;
    for (int m = 0; m < 6; ++m) {
      if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
      // if (real lam = F[5][baseP + m][tx]; lam > max_lambda) {
        max_lambda = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }

      real vPlus[5] = {}, vMinus[5] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 2;
      real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx]
                 + L[3] * F[3][iP][tx] + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
                 + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx];
      vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
      for (int m = 1; m < 5; ++m) {
        iP = i_shared - 2 + m;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
              + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx];
        vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      }
      iP = i_shared + 3;
      LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
            + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
            + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx];
      vMinus[4] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      fChar[l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
    }
  }
  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fc[idx3d] = temp1;
  fc[idx3d + 1 * sz3d] = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
  fc[idx3d + 2 * sz3d] = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
  fc[idx3d + 3 * sz3d] = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

  // temp2 is Roe averaged enthalpy
  temp2 = rlc * (Q[4][i_shared][tx] + F[4][i_shared][tx]) + rrc * (Q[4][i_shared + 1][tx] + F[4][i_shared + 1][tx]);
  fc[idx3d + 4 * sz3d] = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];
}

__device__ void weno_ch_air_yz_piro(const DParameter *param, const real **shared_mem, int i_shared, real kx, real ky,
  real kz, real eps_ref, bool if_shock, real *fc, int idx3d, int sz3d) {
  constexpr int nyy = 32;
  const int n_var = param->n_var;
  const auto tx = static_cast<int>(threadIdx.x);

  auto Q = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(shared_mem);
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(shared_mem + n_var * (nyy + 2 * 4 - 1) * 4);

  // Compute the Roe average state at the interface
  real rho_l = Q[0][i_shared][tx], rho_r = Q[0][i_shared + 1][tx];
  real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
  const real rlc{1 / (rho_l + temp1)};
  const real rrc{1 / (temp1 + rho_r)};
  const real um{rlc * Q[1][i_shared][tx] + rrc * Q[1][i_shared + 1][tx]};
  const real vm{rlc * Q[2][i_shared][tx] + rrc * Q[2][i_shared + 1][tx]};
  const real wm{rlc * Q[3][i_shared][tx] + rrc * Q[3][i_shared + 1][tx]};
  real temp3 = (rlc * F[4][i_shared][tx] + rrc * F[4][i_shared + 1][tx]) / R_air; // temp3 = T
  const real cm = sqrt(gamma_air * R_air * temp3);
  const real gm1{gamma_air - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  // real max_lambda = abs(kx * um + ky * vm + kz * wm);
  temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
  // max_lambda += cm / temp1;
  kx *= temp1;
  ky *= temp1;
  kz *= temp1;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
  if (abs(kx) > 0.8) {
    nx = -kz;
    nz = kx;
  } else {
    ny = kz;
    nz = -ky;
  }
  temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
  nx *= temp1;
  ny *= temp1;
  nz *= temp1;
  qx = ky * nz - kz * ny;
  qy = kz * nx - kx * nz;
  qz = kx * ny - ky * nx;
  temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
  qx *= temp1;
  qy *= temp1;
  qz *= temp1;
  const real Un = nx * um + ny * vm + nz * wm;
  const real Uq = qx * um + qy * vm + qz * wm;

  real temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
  // Local Lax-Friedrichs flux splitting
  temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
  real fChar[5 + MAX_SPEC_NUMBER];
  if (param->inviscid_scheme == 72) {
    const int baseP = i_shared - 3;

    // Find the max spectral radius
    real max_lambda2 = 0;
    for (int m = 0; m < 8; ++m) {
      if (real lam = abs(F[0][baseP + m][tx]); lam > max_lambda2) {
        max_lambda2 = lam;
      }
    }

    for (int l = 0; l < 5; ++l) {
      temp1 = 0.5;
      real L[5];
      switch (l) {
        case 0:
          L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        case 1:
          temp1 = 0;
          L[0] = -Un;
          L[1] = nx;
          L[2] = ny;
          L[3] = nz;
          L[4] = 0;
          break;
        case 2:
          temp1 = 0;
          L[0] = -Uq;
          L[1] = qx;
          L[2] = qy;
          L[3] = qz;
          L[4] = 0;
          break;
        case 3:
          temp1 = -1;
          L[0] = 1 - temp3 * temp2;
          L[1] = gm1 * um * temp2;
          L[2] = gm1 * vm * temp2;
          L[3] = gm1 * wm * temp2;
          L[4] = -gm1 * temp2;
          break;
        case 4:
          L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
          L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
          L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
          L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
          L[4] = gm1 * temp2 * 0.5;
          break;
        default:
          break;
      }
      real max_lambda = max_lambda2;
      if (l == 0) {
        max_lambda = 0;
        for (int m = 0; m < 8; ++m) {
          if (real lam = abs(F[0][baseP + m][tx] - F[5][baseP + m][tx]); lam > max_lambda) {
            max_lambda = lam;
          }
        }
      } else if (l == 4) {
        max_lambda = 0;
        for (int m = 0; m < 8; ++m) {
          if (real lam = abs(F[0][baseP + m][tx] + F[5][baseP + m][tx]); lam > max_lambda) {
            max_lambda = lam;
          }
        }
      }

      real vPlus[7] = {}, vMinus[7] = {};
      // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
      int iP = i_shared - 3;
      real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx]
                 + L[3] * F[3][iP][tx] + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
                 + L[4] * Q[4][iP][tx];
      vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
      // vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
      for (int m = 1; m < 7; ++m) {
        iP = i_shared - 3 + m;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
              + L[4] * Q[4][iP][tx];
        // vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
        // vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
        vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      }
      iP = i_shared - 3 + 7;
      LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
            + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]);
      LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx]
            + L[4] * Q[4][iP][tx];
      // vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - abs(F[0][iP][tx] + F[5][iP][tx]) * LQm);
      vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
      fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
    }
  }
  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  temp1 = fChar[0] + fChar[3] + fChar[4];
  temp3 = fChar[0] - fChar[4];
  fc[idx3d] = temp1;
  fc[idx3d + 1 * sz3d] = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
  fc[idx3d + 2 * sz3d] = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
  fc[idx3d + 3 * sz3d] = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

  // temp2 is Roe averaged enthalpy
  temp2 = rlc * (Q[4][i_shared][tx] + F[4][i_shared][tx]) + rrc * (Q[4][i_shared + 1][tx] + F[4][i_shared + 1][tx]);
  fc[idx3d + 4 * sz3d] = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];
}

// This implements the characteristic WENO with max lambda on the stencil
template<MixtureModel mix_model> __global__ void compute_convective_term_weno_x(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx) return;

  const auto ngg{zone->ngg};

  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i + ii, j, k) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  extern __shared__ real s[];

  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  const real *__restrict__ cv_data = cv.data();
  const real *__restrict__ bv_data = bv.data();
  const real *__restrict__ metric_data = zone->metric.data();
  const real *__restrict__ acoustic_speed_data = zone->acoustic_speed.data();
  const real *__restrict__ jac_data = zone->jac.data();
  auto &fc = zone->fFlux;
  // All arrays here, cv, bv, acoustic_speed, metric, have the same indexing method and the same size, i.e., including 2ngg ghost cells.
  const int sz_3d = cv.size();

  const auto tx = static_cast<int>(threadIdx.x);
  const auto n_var{param->n_var};
  constexpr int nx_block = 64;
  const auto n_active = min(nx_block, zone->mx - static_cast<int>(blockDim.x * blockIdx.x) + 1);
  const int n_point = n_active + 2 * ngg - 1;

  const int il0 = static_cast<int>(blockDim.x * blockIdx.x) - ngg;
  const int i_shared = tx - 1 + ngg;

  const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i + 1, j, k)};
  const real kxJ = zone->metric(i, j, k, 0) * jac_l + zone->metric(i + 1, j, k, 0) * jac_r;
  const real kyJ = zone->metric(i, j, k, 1) * jac_l + zone->metric(i + 1, j, k, 1) * jac_r;
  const real kzJ = zone->metric(i, j, k, 2) * jac_l + zone->metric(i + 1, j, k, 2) * jac_r;
  real eps_ref{0};
  if (if_shock) {
    eps_ref = eps_weno * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);
  }
  real kx = 0.5 * (zone->metric(i, j, k, 0) + zone->metric(i + 1, j, k, 0));
  real ky = 0.5 * (zone->metric(i, j, k, 1) + zone->metric(i + 1, j, k, 1));
  real kz = 0.5 * (zone->metric(i, j, k, 2) + zone->metric(i + 1, j, k, 2));

  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    auto fp = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(s);
    auto fm = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(s + n_var * (nx_block + 2 * 4 - 1));

    for (int il = il0 + tx; il <= il0 + n_point - 1; il += n_active) {
      int iSh = il - il0; // iSh is the shared index

      const int idx3d = cv.idx3d(il, j, k);

      const real q0 = cv_data[idx3d + 0 * sz_3d];
      const real q1 = cv_data[idx3d + 1 * sz_3d];
      const real q2 = cv_data[idx3d + 2 * sz_3d];
      const real q3 = cv_data[idx3d + 3 * sz_3d];
      const real q4 = cv_data[idx3d + 4 * sz_3d];
      real pk = bv_data[idx3d + 4 * sz_3d];
      const real metric0 = metric_data[idx3d + 0 * sz_3d];
      const real metric1 = metric_data[idx3d + 1 * sz_3d];
      const real metric2 = metric_data[idx3d + 2 * sz_3d];

      const real Uk{(q1 * metric0 + q2 * metric1 + q3 * metric2) / q0};
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      const real lambda0 = abs(Uk) + cGradK;
      const real jac = jac_data[idx3d];
      const real sPlus = 0.5 * jac * (Uk + lambda0);
      const real sMinus = 0.5 * jac * (Uk - lambda0);
      pk *= 0.5 * jac;

      fp[0][iSh] = sPlus * q0;
      fp[1][iSh] = fma(sPlus, q1, pk * metric0);
      fp[2][iSh] = fma(sPlus, q2, pk * metric1);
      fp[3][iSh] = fma(sPlus, q3, pk * metric2);
      fp[4][iSh] = fma(sPlus, q4, pk * Uk);

      fm[0][iSh] = sMinus * q0;
      fm[1][iSh] = fma(sMinus, q1, pk * metric0);
      fm[2][iSh] = fma(sMinus, q2, pk * metric1);
      fm[3][iSh] = fma(sMinus, q3, pk * metric2);
      fm[4][iSh] = fma(sMinus, q4, pk * Uk);

      for (int l = 5; l < n_var; ++l) {
        const real ql = cv_data[idx3d + l * sz_3d];
        fp[l][iSh] = sPlus * ql;
        fm[l][iSh] = sMinus * ql;
      }
    }
    __syncthreads();

    // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
    real eps_scaled[3];
    eps_scaled[0] = eps_ref;
    eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
    eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

    for (int l = 0; l < n_var; ++l) {
      real eps_here{eps_scaled[0]};
      if (l == 1 || l == 2 || l == 3) {
        eps_here = eps_scaled[1];
      } else if (l == 4) {
        eps_here = eps_scaled[2];
      }

      if (param->inviscid_scheme == 71) {
        real vp[7], vm[7];
        vp[0] = fp[l][i_shared - 3];
        vp[1] = fp[l][i_shared - 2];
        vp[2] = fp[l][i_shared - 1];
        vp[3] = fp[l][i_shared];
        vp[4] = fp[l][i_shared + 1];
        vp[5] = fp[l][i_shared + 2];
        vp[6] = fp[l][i_shared + 3];
        vm[0] = fm[l][i_shared - 2];
        vm[1] = fm[l][i_shared - 1];
        vm[2] = fm[l][i_shared];
        vm[3] = fm[l][i_shared + 1];
        vm[4] = fm[l][i_shared + 2];
        vm[5] = fm[l][i_shared + 3];
        vm[6] = fm[l][i_shared + 4];

        fc(i, j, k, l) = WENO7(vp, vm, eps_here, if_shock);
      } else if (param->inviscid_scheme == 51) {
        real vp[5], vm[5];
        vp[0] = fp[l][i_shared - 2];
        vp[1] = fp[l][i_shared - 1];
        vp[2] = fp[l][i_shared];
        vp[3] = fp[l][i_shared + 1];
        vp[4] = fp[l][i_shared + 2];
        vm[0] = fm[l][i_shared - 1];
        vm[1] = fm[l][i_shared];
        vm[2] = fm[l][i_shared + 1];
        vm[3] = fm[l][i_shared + 2];
        vm[4] = fm[l][i_shared + 3];

        fc(i, j, k, l) = WENO5(vp, vm, eps_here, if_shock);
      }
    }

    if (param->positive_preserving) {
      real dt{0};
      if (param->dt > 0)
        dt = param->dt;
      else
        dt = zone->dt_local(i, j, k);
      const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

      const real alphaL = 0.5 * alpha * jac_l, alphaR = 0.5 * alpha * jac_r;

      bool need_compute{false};
      int start_l = -1;
      for (int l = 0; l < n_var - 5; ++l) {
        const real up = alphaL * cv(i, j, k, l + 5) - dt * fc(i, j, k, l + 5);
        const real um = alphaR * cv(i + 1, j, k, l + 5) + dt * fc(i, j, k, l + 5);
        if (up < 0 || um < 0) {
          need_compute = true;
          start_l = l;
          break;
        }
      }

      if (!need_compute) {
        return;
      }

      int idx3d = cv.idx3d(i, j, k);
      real q0 = cv_data[idx3d + 0 * sz_3d];
      real q1 = cv_data[idx3d + 1 * sz_3d];
      real q2 = cv_data[idx3d + 2 * sz_3d];
      real q3 = cv_data[idx3d + 3 * sz_3d];
      real metric0 = metric_data[idx3d + 0 * sz_3d];
      real metric1 = metric_data[idx3d + 1 * sz_3d];
      real metric2 = metric_data[idx3d + 2 * sz_3d];
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp1 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_l;

      // Load the right state
      idx3d = cv.idx3d(i + 1, j, k);
      q0 = cv_data[idx3d + 0 * sz_3d];
      q1 = cv_data[idx3d + 1 * sz_3d];
      q2 = cv_data[idx3d + 2 * sz_3d];
      q3 = cv_data[idx3d + 3 * sz_3d];
      metric0 = metric_data[idx3d + 0 * sz_3d];
      metric1 = metric_data[idx3d + 1 * sz_3d];
      metric2 = metric_data[idx3d + 2 * sz_3d];
      cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp2 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_r;

      for (int l = start_l; l < n_var - 5; ++l) {
        real f1{0.0};
        bool f1_computed{false};
        real theta_p = 1.0, theta_m = 1.0;
        const real rhoYL = cv(i, j, k, l + 5);
        const real rhoYR = cv(i + 1, j, k, l + 5);
        const real up = alphaL * rhoYL - dt * fc(i, j, k, l + 5);
        if (up < 0) {
          f1 = 0.5 * (fp[5 + l][i_shared] + temp1 * rhoYL) +
               0.5 * (fm[5 + l][i_shared + 1] - temp2 * rhoYR);
          f1_computed = true;
          const real up_lf = alphaL * rhoYL - dt * f1;
          if (abs(up - up_lf) > 1e-20 * param->rho_ref) {
            theta_p = (0 - up_lf) / (up - up_lf);
            if (theta_p > 1)
              theta_p = 1.0;
            else if (theta_p < 0)
              theta_p = 0;
          }
        }

        const real um = alphaR * rhoYR + dt * fc(i, j, k, l + 5);
        if (um < 0) {
          if (!f1_computed) {
            f1 = 0.5 * (fp[5 + l][i_shared] + temp1 * rhoYL) +
                 0.5 * (fm[5 + l][i_shared + 1] - temp2 * rhoYR);
          }
          const real um_lf = alphaR * rhoYR + dt * f1;
          if (abs(um - um_lf) > 1e-20 * param->rho_ref) {
            theta_m = (0 - um_lf) / (um - um_lf);
            if (theta_m > 1)
              theta_m = 1.0;
            else if (theta_m < 0)
              theta_m = 0;
          }
        }

        fc(i, j, k, l + 5) = min(theta_p, theta_m) * (fc(i, j, k, l + 5) - f1) + f1;
      }
    }
    return;
  }
  // characteristic method
  auto Q = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(s); // rho, rhoU, rhoV, rhoW, rhoE, rhoY1, rhoY2, ...
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nx_block + 2 * 4 - 1]>(s + n_var * (nx_block + 2 * 4 - 1));

  for (int il = il0 + tx; il <= il0 + n_point - 1; il += n_active) {
    int iSh = il - il0; // iSh is the shared index

    const int idx3d = cv.idx3d(il, j, k);

    const real q0 = cv_data[idx3d + 0 * sz_3d];
    const real q1 = cv_data[idx3d + 1 * sz_3d];
    const real q2 = cv_data[idx3d + 2 * sz_3d];
    const real q3 = cv_data[idx3d + 3 * sz_3d];
    const real q4 = cv_data[idx3d + 4 * sz_3d];

    Q[0][iSh] = q0;
    Q[1][iSh] = q1;
    Q[2][iSh] = q2;
    Q[3][iSh] = q3;
    Q[4][iSh] = q4;

    const real metric0 = metric_data[idx3d + 0 * sz_3d];
    const real metric1 = metric_data[idx3d + 1 * sz_3d];
    const real metric2 = metric_data[idx3d + 2 * sz_3d];
    const real Uk{(q1 * metric0 + q2 * metric1 + q3 * metric2) / q0};
    real pk = bv_data[idx3d + 4 * sz_3d];

    F[0][iSh] = Uk;
    F[1][iSh] = fma(q1, Uk, pk * metric0);
    F[2][iSh] = fma(q2, Uk, pk * metric1);
    F[3][iSh] = fma(q3, Uk, pk * metric2);
    F[4][iSh] = pk;

    for (int l = 5; l < n_var; ++l) {
      Q[l][iSh] = cv_data[idx3d + l * sz_3d];
    }

    real cGradK = norm3d(metric0, metric1, metric2);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK *= acoustic_speed_data[idx3d];
    else
      cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
    // const real lambda0 = abs(Uk) + cGradK;
    // F[5][iSh] = lambda0;
    F[5][iSh] = cGradK;
    F[6][iSh] = jac_data[idx3d];
  }
  __syncthreads();
  if constexpr (mix_model == MixtureModel::Air) {
    // weno_ch_air_x_piro(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, fc.data(),
    //               fc.idx3d(i, j, k), fc.size());
    weno_ch_air_x(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, fc.data(),
                  fc.idx3d(i, j, k), fc.size());
  } else {
    // Compute the Roe average state at the interface
    real rho_l = Q[0][i_shared], rho_r = Q[0][i_shared + 1];
    real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
    const real rlc{1 / (rho_l + temp1)};
    const real rrc{1 / (temp1 + rho_r)};
    const real um{rlc * Q[1][i_shared] + rrc * Q[1][i_shared + 1]};
    const real vm{rlc * Q[2][i_shared] + rrc * Q[2][i_shared + 1]};
    const real wm{rlc * Q[3][i_shared] + rrc * Q[3][i_shared + 1]};

    real svm[MAX_SPEC_NUMBER] = {};
    for (int l = 0; l < n_var - 5; ++l) {
      svm[l] = rlc * Q[l + 5][i_shared] + rrc * Q[l + 5][i_shared + 1];
    }

    const int ns{param->n_spec};
    temp1 = 0; // temp1 = gas_constant (R)
    for (int l = 0; l < ns; ++l) {
      temp1 += svm[l] * param->gas_const[l];
    }
    // temp1 = R, temp3 = T
    real temp3 = (rlc * F[4][i_shared] + rrc * F[4][i_shared + 1]) / temp1;

    // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
    real fChar[5 + MAX_SPEC_NUMBER];
    real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
    compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
    real temp2{0};                                         // temp2 = cp
    for (int l = 0; l < ns; ++l) {
      temp2 += svm[l] * fChar[l];
    }
    const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
    const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
    const real gm1{gamma - 1};

    // Next, we compute the left characteristic matrix at i+1/2.
    temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    const real Uk_bar{kx * um + ky * vm + kz * wm};
    real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
    if (abs(kx) > 0.8) {
      nx = -kz;
      nz = kx;
    } else {
      ny = kz;
      nz = -ky;
    }
    temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
    nx *= temp1;
    ny *= temp1;
    nz *= temp1;
    qx = ky * nz - kz * ny;
    qy = kz * nx - kx * nz;
    qz = kx * ny - ky * nx;
    temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
    qx *= temp1;
    qy *= temp1;
    qz *= temp1;
    const real Un = nx * um + ny * vm + nz * wm;
    const real Uq = qx * um + qy * vm + qz * wm;

    // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
    //  const real cm2_inv{1.0 / (cm * cm)};
    temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
    // Compute the characteristic flux with L.
    // compute the partial derivative of pressure to species density
    for (int l = 0; l < ns; ++l) {
      hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
      // The computations including this alpha_l are all combined with a division by cm2.
      hI_alpI[l] *= temp2;
    }

    // Local Lax-Friedrichs flux splitting
    temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
    if (param->inviscid_scheme == 72) {
      const int baseP = i_shared - 3;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 8; ++m) {
        if (real lam = abs(F[0][baseP + m]) + F[5][baseP + m]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[8], sumBetaF[8];
      for (int m = 0; m < 8; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP] * F[0][iP], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 3;
        real LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                   + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
                   + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
                + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
              + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[7];
        LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
              + temp1 * sumBetaQ[7];
        vMinus[6] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[7], vMinus[7];
        int iP = i_shared - 3;
        real LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
        real LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
        vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
          LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
          vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
        LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
        vMinus[6] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
    } else if (param->inviscid_scheme == 52) {
      const int baseP = i_shared - 2;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 6; ++m) {
        if (real lam = abs(F[0][baseP + m]) + F[5][baseP + m]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[6], sumBetaF[6];
      for (int m = 0; m < 6; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP] * F[0][iP], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[5] = {}, vMinus[5] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 2;
        real LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                   + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
                   + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
                + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
                + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = L[0] * (Q[0][iP] * F[0][iP]) + L[1] * F[1][iP] + L[2] * F[2][iP] + L[3] * F[3][iP]
              + L[4] * ((Q[4][iP] + F[4][iP]) * F[0][iP]) + temp1 * sumBetaF[5];
        LQm = L[0] * Q[0][iP] + L[1] * Q[1][iP] + L[2] * Q[2][iP] + L[3] * Q[3][iP] + L[4] * Q[4][iP]
              + temp1 * sumBetaQ[5];
        vMinus[4] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        fChar[l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[5], vMinus[5];
        int iP = i_shared - 2;
        real LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
        real LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
        vPlus[0] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
          LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
          vPlus[m] = 0.5 * F[6][iP] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = F[0][iP] * (Q[5 + l][iP] - svm[l] * Q[0][iP]);
        LQm = Q[5 + l][iP] - svm[l] * Q[0][iP];
        vMinus[4] = 0.5 * F[6][iP] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
    }

    // Project the flux back to physical space
    // We do not compute the right characteristic matrix here, because we explicitly write the components below.
    temp1 = fChar[0] + fChar[3] + fChar[4];
    temp3 = fChar[0] - fChar[4];
    fc(i, j, k, 0) = temp1;
    fc(i, j, k, 1) = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
    fc(i, j, k, 2) = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
    fc(i, j, k, 3) = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

    // temp2 is Roe averaged enthalpy
    temp2 = rlc * (Q[4][i_shared] + F[4][i_shared]) + rrc * (Q[4][i_shared + 1] + F[4][i_shared + 1]);
    fc(i, j, k, 4) = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];

    temp2 = 0;
    for (int l = 0; l < ns; ++l) {
      fc(i, j, k, 5 + l) = svm[l] * temp1 + fChar[l + 5];
      temp2 += hI_alpI[l] * fChar[l + 5];
    }
    fc(i, j, k, 4) -= temp2 * cm * cm / gm1;

    if (param->positive_preserving) {
      real dt{0};
      if (param->dt > 0)
        dt = param->dt;
      else
        dt = zone->dt_local(i, j, k);
      const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

      const real alphaL = 0.5 * alpha * jac_l, alphaR = 0.5 * alpha * jac_r;

      bool need_compute{false};
      int start_l = -1;
      for (int l = 0; l < n_var - 5; ++l) {
        const real up = alphaL * Q[l + 5][i_shared] - dt * fc(i, j, k, l + 5);
        const real uMinus = alphaR * Q[l + 5][i_shared + 1] + dt * fc(i, j, k, l + 5);
        if (up < 0 || uMinus < 0) {
          need_compute = true;
          start_l = l;
          break;
        }
      }

      if (!need_compute) {
        return;
      }

      real U1 = F[0][i_shared] * jac_l;
      real U2 = F[0][i_shared + 1] * jac_r;

      for (int l = start_l; l < n_var - 5; ++l) {
        real f1{0.0};
        bool f1_computed{false};
        real theta_p = 1.0, theta_m = 1.0;
        const real rhoYL = Q[l + 5][i_shared];
        const real rhoYR = Q[l + 5][i_shared + 1];
        real fp = 0.5 * F[6][i_shared] * (F[0][i_shared] + F[5][i_shared]) * rhoYL;
        real fm = 0.5 * F[6][i_shared + 1] * (F[0][i_shared + 1] - F[5][i_shared + 1]) * rhoYR;

        const real up = alphaL * rhoYL - dt * fc(i, j, k, l + 5);
        if (up < 0) {
          f1 = 0.5 * (fp + U1 * rhoYL) + 0.5 * (fm - U2 * rhoYR);
          f1_computed = true;
          const real up_lf = alphaL * rhoYL - dt * f1;
          if (abs(up - up_lf) > 1e-20 * param->rho_ref) {
            theta_p = (0 - up_lf) / (up - up_lf);
            if (theta_p > 1)
              theta_p = 1.0;
            else if (theta_p < 0)
              theta_p = 0;
          }
        }

        const real uMinus = alphaR * rhoYR + dt * fc(i, j, k, l + 5);
        if (uMinus < 0) {
          if (!f1_computed) {
            f1 = 0.5 * (fp + U1 * rhoYL) + 0.5 * (fm - U2 * rhoYR);
          }
          const real um_lf = alphaR * rhoYR + dt * f1;
          if (abs(uMinus - um_lf) > 1e-20 * param->rho_ref) {
            theta_m = (0 - um_lf) / (uMinus - um_lf);
            if (theta_m > 1)
              theta_m = 1.0;
            else if (theta_m < 0)
              theta_m = 0;
          }
        }

        fc(i, j, k, l + 5) = min(theta_p, theta_m) * (fc(i, j, k, l + 5) - f1) + f1;
      }
    }
  }
}

__global__ void compute_derivative_x(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dqA = zone->dq;
  auto &fcA = zone->fFlux;

  const int sz_dq = dqA.size();
  const int sz_fc = fcA.size();
  real *__restrict__ dq = dqA.data();
  const real *__restrict__ fc = fcA.data();

  const int idx_dq = dqA.idx3d(i, j, k);
  const int idx_m = fcA.idx3d(i - 1, j, k);
  const int idx_c = fcA.idx3d(i, j, k);

  dq[idx_dq + 0 * sz_dq] -= fc[idx_c + 0 * sz_fc] - fc[idx_m + 0 * sz_fc];
  dq[idx_dq + 1 * sz_dq] -= fc[idx_c + 1 * sz_fc] - fc[idx_m + 1 * sz_fc];
  dq[idx_dq + 2 * sz_dq] -= fc[idx_c + 2 * sz_fc] - fc[idx_m + 2 * sz_fc];
  dq[idx_dq + 3 * sz_dq] -= fc[idx_c + 3 * sz_fc] - fc[idx_m + 3 * sz_fc];
  dq[idx_dq + 4 * sz_dq] -= fc[idx_c + 4 * sz_fc] - fc[idx_m + 4 * sz_fc];
  for (int l = 5; l < nv; ++l) {
    dq[idx_dq + l * sz_dq] -= fc[idx_c + l * sz_fc] - fc[idx_m + l * sz_fc];
  }
}

template<MixtureModel mix_model> __global__ void compute_convective_term_weno_y(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (j >= zone->my || i >= zone->mx) return;

  const auto ngg{zone->ngg};
  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j + ii, k) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  extern __shared__ real s[];

  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  const real *__restrict__ cv_data = cv.data();
  const real *__restrict__ bv_data = bv.data();
  const real *__restrict__ metric_data = zone->metric.data();
  const real *__restrict__ acoustic_speed_data = zone->acoustic_speed.data();
  const real *__restrict__ jac_data = zone->jac.data();
  // All arrays here, cv, bv, acoustic_speed, metric, have the same indexing method and the same size, i.e., including 2ngg ghost cells.
  const int sz_3d = cv.size();

  const auto ty = static_cast<int>(threadIdx.y), tx = static_cast<int>(threadIdx.x);
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.y), zone->my - static_cast<int>(blockDim.y * blockIdx.y) + 1);
  const int n_point = n_active + 2 * ngg - 1;
  constexpr int nyy = 32;

  const int jl0 = static_cast<int>(blockDim.y * blockIdx.y) - ngg;
  const int i_shared = ty - 1 + ngg;

  auto &gc = zone->gFlux;
  const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i, j + 1, k)};
  const real kxJ = zone->metric(i, j, k, 3) * jac_l + zone->metric(i, j + 1, k, 3) * jac_r;
  const real kyJ = zone->metric(i, j, k, 4) * jac_l + zone->metric(i, j + 1, k, 4) * jac_r;
  const real kzJ = zone->metric(i, j, k, 5) * jac_l + zone->metric(i, j + 1, k, 5) * jac_r;
  real eps_ref{0};
  if (if_shock) {
    eps_ref = eps_weno * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);
  }
  real kx = 0.5 * (zone->metric(i, j, k, 3) + zone->metric(i + 1, j, k, 3));
  real ky = 0.5 * (zone->metric(i, j, k, 4) + zone->metric(i + 1, j, k, 4));
  real kz = 0.5 * (zone->metric(i, j, k, 5) + zone->metric(i + 1, j, k, 5));

  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    auto fp = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
    auto fm = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

    for (int jl = jl0 + ty; jl <= jl0 + n_point - 1; jl += n_active) {
      int iSh = jl - jl0; // iSh is the shared index

      const int idx3d = cv.idx3d(i, jl, k);

      const real q0 = cv_data[idx3d + 0 * sz_3d];
      const real q1 = cv_data[idx3d + 1 * sz_3d];
      const real q2 = cv_data[idx3d + 2 * sz_3d];
      const real q3 = cv_data[idx3d + 3 * sz_3d];
      const real q4 = cv_data[idx3d + 4 * sz_3d];
      real pk = bv_data[idx3d + 4 * sz_3d];
      const real metric0 = metric_data[idx3d + 3 * sz_3d];
      const real metric1 = metric_data[idx3d + 4 * sz_3d];
      const real metric2 = metric_data[idx3d + 5 * sz_3d];

      const real Uk{(q1 * metric0 + q2 * metric1 + q3 * metric2) / q0};
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      const real lambda0 = abs(Uk) + cGradK;
      const real halfJac = 0.5 * jac_data[idx3d];
      const real sPlus = halfJac * (Uk + lambda0);
      const real sMinus = halfJac * (Uk - lambda0);
      pk *= halfJac;

      fp[0][iSh][tx] = sPlus * q0;
      fp[1][iSh][tx] = fma(sPlus, q1, pk * metric0);
      fp[2][iSh][tx] = fma(sPlus, q2, pk * metric1);
      fp[3][iSh][tx] = fma(sPlus, q3, pk * metric2);
      fp[4][iSh][tx] = fma(sPlus, q4, pk * Uk);

      fm[0][iSh][tx] = sMinus * q0;
      fm[1][iSh][tx] = fma(sMinus, q1, pk * metric0);
      fm[2][iSh][tx] = fma(sMinus, q2, pk * metric1);
      fm[3][iSh][tx] = fma(sMinus, q3, pk * metric2);
      fm[4][iSh][tx] = fma(sMinus, q4, pk * Uk);

      for (int l = 5; l < n_var; ++l) {
        const real ql = cv_data[idx3d + l * sz_3d];
        fp[l][iSh][tx] = sPlus * ql;
        fm[l][iSh][tx] = sMinus * ql;
      }
    }
    __syncthreads();

    real eps_scaled[3];
    eps_scaled[0] = eps_ref;
    eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
    eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

    for (int l = 0; l < n_var; ++l) {
      real eps_here{eps_scaled[0]};
      if (l == 1 || l == 2 || l == 3) {
        eps_here = eps_scaled[1];
      } else if (l == 4) {
        eps_here = eps_scaled[2];
      }

      if (param->inviscid_scheme == 71) {
        real vp[7], vm[7];
        vp[0] = fp[l][i_shared - 3][tx];
        vp[1] = fp[l][i_shared - 2][tx];
        vp[2] = fp[l][i_shared - 1][tx];
        vp[3] = fp[l][i_shared][tx];
        vp[4] = fp[l][i_shared + 1][tx];
        vp[5] = fp[l][i_shared + 2][tx];
        vp[6] = fp[l][i_shared + 3][tx];
        vm[0] = fm[l][i_shared - 2][tx];
        vm[1] = fm[l][i_shared - 1][tx];
        vm[2] = fm[l][i_shared][tx];
        vm[3] = fm[l][i_shared + 1][tx];
        vm[4] = fm[l][i_shared + 2][tx];
        vm[5] = fm[l][i_shared + 3][tx];
        vm[6] = fm[l][i_shared + 4][tx];

        gc(i, j, k, l) = WENO7(vp, vm, eps_here, if_shock);
      } else if (param->inviscid_scheme == 51) {
        real vp[5], vm[5];
        vp[0] = fp[l][i_shared - 2][tx];
        vp[1] = fp[l][i_shared - 1][tx];
        vp[2] = fp[l][i_shared][tx];
        vp[3] = fp[l][i_shared + 1][tx];
        vp[4] = fp[l][i_shared + 2][tx];
        vm[0] = fm[l][i_shared - 1][tx];
        vm[1] = fm[l][i_shared][tx];
        vm[2] = fm[l][i_shared + 1][tx];
        vm[3] = fm[l][i_shared + 2][tx];
        vm[4] = fm[l][i_shared + 3][tx];

        gc(i, j, k, l) = WENO5(vp, vm, eps_here, if_shock);
      }
    }

    if (param->positive_preserving) {
      real dt{0};
      if (param->dt > 0)
        dt = param->dt;
      else
        dt = zone->dt_local(i, j, k);
      const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

      const real alphaL = 0.5 * alpha * jac_l, alphaR = 0.5 * alpha * jac_r;

      bool need_compute{false};
      int start_l = -1;
      for (int l = 0; l < n_var - 5; ++l) {
        const real up = alphaL * cv(i, j, k, l + 5) - dt * gc(i, j, k, l + 5);
        const real um = alphaR * cv(i, j + 1, k, l + 5) + dt * gc(i, j, k, l + 5);
        if (up < 0 || um < 0) {
          need_compute = true;
          start_l = l;
          break;
        }
      }

      if (!need_compute) {
        return;
      }

      int idx3d = cv.idx3d(i, j, k);
      real q0 = cv_data[idx3d + 0 * sz_3d];
      real q1 = cv_data[idx3d + 1 * sz_3d];
      real q2 = cv_data[idx3d + 2 * sz_3d];
      real q3 = cv_data[idx3d + 3 * sz_3d];
      real metric0 = metric_data[idx3d + 0 * sz_3d];
      real metric1 = metric_data[idx3d + 1 * sz_3d];
      real metric2 = metric_data[idx3d + 2 * sz_3d];
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp1 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_l;

      // Load the right state
      idx3d = cv.idx3d(i, j + 1, k);
      q0 = cv_data[idx3d + 0 * sz_3d];
      q1 = cv_data[idx3d + 1 * sz_3d];
      q2 = cv_data[idx3d + 2 * sz_3d];
      q3 = cv_data[idx3d + 3 * sz_3d];
      metric0 = metric_data[idx3d + 0 * sz_3d];
      metric1 = metric_data[idx3d + 1 * sz_3d];
      metric2 = metric_data[idx3d + 2 * sz_3d];
      cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp2 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_r;

      for (int l = start_l; l < n_var - 5; ++l) {
        real f1{0.0};
        bool f1_computed{false};
        real theta_p = 1.0, theta_m = 1.0;
        const real rhoYL = cv(i, j, k, l + 5);
        const real rhoYR = cv(i, j + 1, k, l + 5);
        const real up = alphaL * rhoYL - dt * gc(i, j, k, l + 5);
        if (up < 0) {
          f1 = 0.5 * (fp[5 + l][i_shared][tx] + temp1 * rhoYL) +
               0.5 * (fm[5 + l][i_shared + 1][tx] - temp2 * rhoYR);
          f1_computed = true;
          const real up_lf = alphaL * rhoYL - dt * f1;
          if (abs(up - up_lf) > 1e-20 * param->rho_ref) {
            theta_p = (0 - up_lf) / (up - up_lf);
            if (theta_p > 1)
              theta_p = 1.0;
            else if (theta_p < 0)
              theta_p = 0;
          }
        }

        const real um = alphaR * rhoYR + dt * gc(i, j, k, l + 5);
        if (um < 0) {
          if (!f1_computed) {
            f1 = 0.5 * (fp[5 + l][i_shared][tx] + temp1 * rhoYL) +
                 0.5 * (fm[5 + l][i_shared + 1][tx] - temp2 * rhoYR);
          }
          const real um_lf = alphaR * rhoYR + dt * f1;
          if (abs(um - um_lf) > 1e-20 * param->rho_ref) {
            theta_m = (0 - um_lf) / (um - um_lf);
            if (theta_m > 1)
              theta_m = 1.0;
            else if (theta_m < 0)
              theta_m = 0;
          }
        }

        gc(i, j, k, l + 5) = min(theta_p, theta_m) * (gc(i, j, k, l + 5) - f1) + f1;
      }
    }
    return;
  }
  // Characteristic method
  auto Q = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
  // F stores temporary values, including
  // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac
  auto F = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

  for (int jl = jl0 + ty; jl <= jl0 + n_point - 1; jl += n_active) {
    int iSh = jl - jl0; // iSh is the shared index

    const int idx3d = cv.idx3d(i, jl, k);

    const real q0 = cv_data[idx3d + 0 * sz_3d];
    const real q1 = cv_data[idx3d + 1 * sz_3d];
    const real q2 = cv_data[idx3d + 2 * sz_3d];
    const real q3 = cv_data[idx3d + 3 * sz_3d];
    const real q4 = cv_data[idx3d + 4 * sz_3d];
    real pk = bv_data[idx3d + 4 * sz_3d];
    const real metric0 = metric_data[idx3d + 3 * sz_3d];
    const real metric1 = metric_data[idx3d + 4 * sz_3d];
    const real metric2 = metric_data[idx3d + 5 * sz_3d];

    const real Uk{(q1 * metric0 + q2 * metric1 + q3 * metric2) / q0};

    Q[0][iSh][tx] = q0;
    Q[1][iSh][tx] = q1;
    Q[2][iSh][tx] = q2;
    Q[3][iSh][tx] = q3;
    Q[4][iSh][tx] = q4;

    F[0][iSh][tx] = Uk;
    F[1][iSh][tx] = fma(q1, Uk, pk * metric0);
    F[2][iSh][tx] = fma(q2, Uk, pk * metric1);
    F[3][iSh][tx] = fma(q3, Uk, pk * metric2);
    F[4][iSh][tx] = pk;

    for (int l = 5; l < n_var; ++l) {
      Q[l][iSh][tx] = cv_data[idx3d + l * sz_3d];
    }

    real cGradK = norm3d(metric0, metric1, metric2);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK *= acoustic_speed_data[idx3d];
    else
      cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
    // const real lambda0 = abs(Uk) + cGradK;
    // F[5][iSh][tx] = lambda0;
    F[5][iSh][tx] = cGradK;
    F[6][iSh][tx] = jac_data[idx3d];
  }
  __syncthreads();

  if constexpr (mix_model == MixtureModel::Air) {
    // weno_ch_air_yz_piro(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, gc.data(),
    //                gc.idx3d(i, j, k), gc.size());
    weno_ch_air_yz(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, gc.data(),
                   gc.idx3d(i, j, k), gc.size());
  } else {
    real rho_l = Q[0][i_shared][tx], rho_r = Q[0][i_shared + 1][tx];
    // First, compute the Roe average of the half-point variables.
    real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
    const real rlc{1 / (rho_l + temp1)};
    const real rrc{1 / (temp1 + rho_r)};
    const real um{rlc * Q[1][i_shared][tx] + rrc * Q[1][i_shared + 1][tx]};
    const real vm{rlc * Q[2][i_shared][tx] + rrc * Q[2][i_shared + 1][tx]};
    const real wm{rlc * Q[3][i_shared][tx] + rrc * Q[3][i_shared + 1][tx]};

    real svm[MAX_SPEC_NUMBER] = {};
    for (int l = 0; l < n_var - 5; ++l) {
      svm[l] = rlc * Q[l + 5][i_shared][tx] + rrc * Q[l + 5][i_shared + 1][tx];
    }

    const int ns{param->n_spec};
    temp1 = 0; // temp1 = gas_constant (R)
    for (int l = 0; l < ns; ++l) {
      temp1 += svm[l] * param->gas_const[l];
    }
    // temp1 = R, temp3 = T
    real temp3 = (rlc * F[4][i_shared][tx] + rrc * F[4][i_shared + 1][tx]) / temp1;

    // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
    real fChar[5 + MAX_SPEC_NUMBER];
    real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
    compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
    real temp2{0};                                         // temp2 = cp
    for (int l = 0; l < ns; ++l) {
      temp2 += svm[l] * fChar[l];
    }
    const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
    const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
    const real gm1{gamma - 1};

    // Next, we compute the left characteristic matrix at i+1/2.
    temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    const real Uk_bar{kx * um + ky * vm + kz * wm};
    real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
    if (abs(kx) > 0.8) {
      nx = -kz;
      nz = kx;
    } else {
      ny = kz;
      nz = -ky;
    }
    temp1 = rnorm3d(nx, ny, nz); // temp1 is the norm of the unit normal vector
    nx *= temp1;
    ny *= temp1;
    nz *= temp1;
    qx = ky * nz - kz * ny;
    qy = kz * nx - kx * nz;
    qz = kx * ny - ky * nx;
    temp1 = rnorm3d(qx, qy, qz); // temp1 is the norm of the unit normal vector
    qx *= temp1;
    qy *= temp1;
    qz *= temp1;
    const real Un = nx * um + ny * vm + nz * wm;
    const real Uq = qx * um + qy * vm + qz * wm;

    // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
    //  const real cm2_inv{1.0 / (cm * cm)};
    temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
    // Compute the characteristic flux with L.
    // compute the partial derivative of pressure to species density
    for (int l = 0; l < ns; ++l) {
      hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
      // The computations including this alpha_l are all combined with a division by cm2.
      hI_alpI[l] *= temp2;
    }

    // Li Xinliang's flux splitting
    //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
    temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
    if (param->inviscid_scheme == 72) {
      const int baseP = i_shared - 3;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 8; ++m) {
        if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[8], sumBetaF[8];
      for (int m = 0; m < 8; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP][tx], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP][tx] * F[0][iP][tx], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 3;
        real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][
                     tx]
                   + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][
                     iP][tx]
                   + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
                + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][
                  tx]
                + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[7];
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][
                tx]
              + temp1 * sumBetaQ[7];
        vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[7], vMinus[7];
        int iP = i_shared - 3;
        real LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        real LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
          LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
    } else if (param->inviscid_scheme == 52) {
      const int baseP = i_shared - 2;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 6; ++m) {
        if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[6], sumBetaF[6];
      for (int m = 0; m < 6; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP][tx], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP][tx] * F[0][iP][tx], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[5] = {}, vMinus[5] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 2;
        real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx]
                   + L[3] * F[3][iP][tx] + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
                   + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
                + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
                + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[5];
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
              + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[5];
        vMinus[4] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[5], vMinus[5];
        int iP = i_shared - 2;
        real LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        real LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
          LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vMinus[4] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
    } // temp2 is not 1/(c*c) anymore.

    // Project the flux back to physical space
    // We do not compute the right characteristic matrix here, because we explicitly write the components below.
    temp1 = fChar[0] + fChar[3] + fChar[4];
    temp3 = fChar[0] - fChar[4];
    gc(i, j, k, 0) = temp1;
    gc(i, j, k, 1) = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
    gc(i, j, k, 2) = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
    gc(i, j, k, 3) = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

    // temp2 is Roe averaged enthalpy
    temp2 = rlc * (Q[4][i_shared][tx] + F[4][i_shared][tx]) + rrc * (Q[4][i_shared + 1][tx] + F[4][i_shared + 1][tx]);
    gc(i, j, k, 4) = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];

    temp2 = 0;
    for (int l = 0; l < ns; ++l) {
      gc(i, j, k, 5 + l) = svm[l] * temp1 + fChar[l + 5];
      temp2 += hI_alpI[l] * fChar[l + 5];
    }
    gc(i, j, k, 4) -= temp2 * cm * cm / gm1;
  }
}

__global__ void compute_derivative_y(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dqA = zone->dq;
  auto &gcA = zone->gFlux;

  const int sz_dq = dqA.size();
  const int sz_gc = gcA.size();
  real *__restrict__ dq = dqA.data();
  const real *__restrict__ gc = gcA.data();

  const int idx_dq = dqA.idx3d(i, j, k);
  const int idx_m = gcA.idx3d(i, j - 1, k);
  const int idx_c = gcA.idx3d(i, j, k);

  dq[idx_dq + 0 * sz_dq] -= gc[idx_c + 0 * sz_gc] - gc[idx_m + 0 * sz_gc];
  dq[idx_dq + 1 * sz_dq] -= gc[idx_c + 1 * sz_gc] - gc[idx_m + 1 * sz_gc];
  dq[idx_dq + 2 * sz_dq] -= gc[idx_c + 2 * sz_gc] - gc[idx_m + 2 * sz_gc];
  dq[idx_dq + 3 * sz_dq] -= gc[idx_c + 3 * sz_gc] - gc[idx_m + 3 * sz_gc];
  dq[idx_dq + 4 * sz_dq] -= gc[idx_c + 4 * sz_gc] - gc[idx_m + 4 * sz_gc];
  for (int l = 5; l < nv; ++l) {
    dq[idx_dq + l * sz_dq] -= gc[idx_c + l * sz_gc] - gc[idx_m + l * sz_gc];
  }
}

template<MixtureModel mix_model> __global__ void compute_convective_term_weno_z(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  const int max_extent = zone->mz;
  if (k >= max_extent || i >= zone->mx) return;

  const auto ngg{zone->ngg};
  bool if_shock = false;
  if (param->sensor_threshold > 1e-10) {
    for (int ii = -ngg + 1; ii <= ngg; ++ii) {
      if (zone->shock_sensor(i, j, k + ii) > param->sensor_threshold) {
        if_shock = true;
        break;
      }
    }
  } else {
    if_shock = true;
  }

  extern __shared__ real s[];

  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  const real *__restrict__ cv_data = cv.data();
  const real *__restrict__ bv_data = bv.data();
  const real *__restrict__ metric_data = zone->metric.data();
  const real *__restrict__ acoustic_speed_data = zone->acoustic_speed.data();
  const real *__restrict__ jac_data = zone->jac.data();
  // All arrays here, cv, bv, acoustic_speed, metric, have the same indexing method and the same size, i.e., including ngg ghost cells.
  const int sz_3d = cv.size();

  const auto tz = static_cast<int>(threadIdx.z), tx = static_cast<int>(threadIdx.x);
  const auto n_var{param->n_var};
  const auto n_active = min(static_cast<int>(blockDim.z), max_extent - static_cast<int>(blockDim.z * blockIdx.z) + 1);
  const int n_point = n_active + 2 * ngg - 1;
  constexpr int nyy = 32;

  const int kl0 = static_cast<int>(blockDim.z * blockIdx.z) - ngg;
  const int i_shared = tz - 1 + ngg;

  auto &hc = zone->hFlux;
  const real jac_l{zone->jac(i, j, k)}, jac_r{zone->jac(i, j, k + 1)};
  const real kxJ = zone->metric(i, j, k, 6) * jac_l + zone->metric(i, j, k + 1, 6) * jac_r;
  const real kyJ = zone->metric(i, j, k, 7) * jac_l + zone->metric(i, j, k + 1, 7) * jac_r;
  const real kzJ = zone->metric(i, j, k, 8) * jac_l + zone->metric(i, j, k + 1, 8) * jac_r;
  real eps_ref{0};
  if (if_shock) {
    eps_ref = eps_weno * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);
  }
  real kx = 0.5 * (zone->metric(i, j, k, 6) + zone->metric(i + 1, j, k, 6));
  real ky = 0.5 * (zone->metric(i, j, k, 7) + zone->metric(i + 1, j, k, 7));
  real kz = 0.5 * (zone->metric(i, j, k, 8) + zone->metric(i + 1, j, k, 8));

  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    auto fp = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
    auto fm = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

    for (int kl = kl0 + tz; kl <= kl0 + n_point - 1; kl += n_active) {
      int iSh = kl - kl0; // iSh is the shared index

      const int idx3d = cv.idx3d(i, j, kl);

      const real q0 = cv_data[idx3d];
      const real q1 = cv_data[idx3d + 1 * sz_3d];
      const real q2 = cv_data[idx3d + 2 * sz_3d];
      const real q3 = cv_data[idx3d + 3 * sz_3d];
      const real q4 = cv_data[idx3d + 4 * sz_3d];
      real pk = bv_data[idx3d + 4 * sz_3d];
      const real metric0 = metric_data[idx3d + 6 * sz_3d];
      const real metric1 = metric_data[idx3d + 7 * sz_3d];
      const real metric2 = metric_data[idx3d + 8 * sz_3d];

      const real Uk = (q1 * metric0 + q2 * metric1 + q3 * metric2) / q0;
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      const real lambda0 = abs(Uk) + cGradK;
      const real halfJac = 0.5 * jac_data[idx3d];
      const real sPlus = halfJac * (Uk + lambda0);
      const real sMinus = halfJac * (Uk - lambda0);
      pk *= halfJac;

      fp[0][iSh][tx] = sPlus * q0;
      fp[1][iSh][tx] = fma(sPlus, q1, pk * metric0);
      fp[2][iSh][tx] = fma(sPlus, q2, pk * metric1);
      fp[3][iSh][tx] = fma(sPlus, q3, pk * metric2);
      fp[4][iSh][tx] = fma(sPlus, q4, pk * Uk);

      fm[0][iSh][tx] = sMinus * q0;
      fm[1][iSh][tx] = fma(sMinus, q1, pk * metric0);
      fm[2][iSh][tx] = fma(sMinus, q2, pk * metric1);
      fm[3][iSh][tx] = fma(sMinus, q3, pk * metric2);
      fm[4][iSh][tx] = fma(sMinus, q4, pk * Uk);

      for (int l = 5; l < n_var; ++l) {
        const real ql = cv_data[idx3d + l * sz_3d];
        fp[l][iSh][tx] = sPlus * ql;
        fm[l][iSh][tx] = sMinus * ql;
      }
    }
    __syncthreads();

    real eps_scaled[3];
    eps_scaled[0] = eps_ref;
    eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
    eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

    for (int l = 0; l < n_var; ++l) {
      real eps_here{eps_scaled[0]};
      if (l == 1 || l == 2 || l == 3) {
        eps_here = eps_scaled[1];
      } else if (l == 4) {
        eps_here = eps_scaled[2];
      }

      if (param->inviscid_scheme == 71) {
        real vp[7], vm[7];
        vp[0] = fp[l][i_shared - 3][tx];
        vp[1] = fp[l][i_shared - 2][tx];
        vp[2] = fp[l][i_shared - 1][tx];
        vp[3] = fp[l][i_shared][tx];
        vp[4] = fp[l][i_shared + 1][tx];
        vp[5] = fp[l][i_shared + 2][tx];
        vp[6] = fp[l][i_shared + 3][tx];
        vm[0] = fm[l][i_shared - 2][tx];
        vm[1] = fm[l][i_shared - 1][tx];
        vm[2] = fm[l][i_shared][tx];
        vm[3] = fm[l][i_shared + 1][tx];
        vm[4] = fm[l][i_shared + 2][tx];
        vm[5] = fm[l][i_shared + 3][tx];
        vm[6] = fm[l][i_shared + 4][tx];

        hc(i, j, k, l) = WENO7(vp, vm, eps_here, if_shock);
      } else if (param->inviscid_scheme == 51) {
        real vp[5], vm[5];
        vp[0] = fp[l][i_shared - 2][tx];
        vp[1] = fp[l][i_shared - 1][tx];
        vp[2] = fp[l][i_shared][tx];
        vp[3] = fp[l][i_shared + 1][tx];
        vp[4] = fp[l][i_shared + 2][tx];
        vm[0] = fm[l][i_shared - 1][tx];
        vm[1] = fm[l][i_shared][tx];
        vm[2] = fm[l][i_shared + 1][tx];
        vm[3] = fm[l][i_shared + 2][tx];
        vm[4] = fm[l][i_shared + 3][tx];

        hc(i, j, k, l) = WENO5(vp, vm, eps_here, if_shock);
      }
    }

    if (param->positive_preserving) {
      real dt{0};
      if (param->dt > 0)
        dt = param->dt;
      else
        dt = zone->dt_local(i, j, k);
      const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

      const real alphaL = 0.5 * alpha * jac_l, alphaR = 0.5 * alpha * jac_r;

      bool need_compute{false};
      int start_l = -1;
      for (int l = 0; l < n_var - 5; ++l) {
        const real up = alphaL * cv(i, j, k, l + 5) - dt * hc(i, j, k, l + 5);
        const real um = alphaR * cv(i, j, k + 1, l + 5) + dt * hc(i, j, k, l + 5);
        if (up < 0 || um < 0) {
          need_compute = true;
          start_l = l;
          break;
        }
      }

      if (!need_compute) {
        return;
      }

      int idx3d = cv.idx3d(i, j, k);
      real q0 = cv_data[idx3d + 0 * sz_3d];
      real q1 = cv_data[idx3d + 1 * sz_3d];
      real q2 = cv_data[idx3d + 2 * sz_3d];
      real q3 = cv_data[idx3d + 3 * sz_3d];
      real metric0 = metric_data[idx3d + 0 * sz_3d];
      real metric1 = metric_data[idx3d + 1 * sz_3d];
      real metric2 = metric_data[idx3d + 2 * sz_3d];
      real cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp1 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_l;

      // Load the right state
      idx3d = cv.idx3d(i, j, k + 1);
      q0 = cv_data[idx3d + 0 * sz_3d];
      q1 = cv_data[idx3d + 1 * sz_3d];
      q2 = cv_data[idx3d + 2 * sz_3d];
      q3 = cv_data[idx3d + 3 * sz_3d];
      metric0 = metric_data[idx3d + 0 * sz_3d];
      metric1 = metric_data[idx3d + 1 * sz_3d];
      metric2 = metric_data[idx3d + 2 * sz_3d];
      cGradK = norm3d(metric0, metric1, metric2);
      if constexpr (mix_model != MixtureModel::Air)
        cGradK *= acoustic_speed_data[idx3d];
      else
        cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
      real temp2 = (metric0 * q1 + metric1 * q2 + metric2 * q3) / q0 * jac_r;

      for (int l = start_l; l < n_var - 5; ++l) {
        real f1{0.0};
        bool f1_computed{false};
        real theta_p = 1.0, theta_m = 1.0;
        const real rhoYL = cv(i, j, k, l + 5);
        const real rhoYR = cv(i, j, k + 1, l + 5);
        const real up = alphaL * rhoYL - dt * hc(i, j, k, l + 5);
        if (up < 0) {
          f1 = 0.5 * (fp[5 + l][i_shared][tx] + temp1 * rhoYL) +
               0.5 * (fm[5 + l][i_shared + 1][tx] - temp2 * rhoYR);
          f1_computed = true;
          const real up_lf = alphaL * rhoYL - dt * f1;
          if (abs(up - up_lf) > 1e-20 * param->rho_ref) {
            theta_p = (0 - up_lf) / (up - up_lf);
            if (theta_p > 1)
              theta_p = 1.0;
            else if (theta_p < 0)
              theta_p = 0;
          }
        }

        const real um = alphaR * rhoYR + dt * hc(i, j, k, l + 5);
        if (um < 0) {
          if (!f1_computed) {
            f1 = 0.5 * (fp[5 + l][i_shared][tx] + temp1 * rhoYL) +
                 0.5 * (fm[5 + l][i_shared + 1][tx] - temp2 * rhoYR);
          }
          const real um_lf = alphaR * rhoYR + dt * f1;
          if (abs(um - um_lf) > 1e-20 * param->rho_ref) {
            theta_m = (0 - um_lf) / (um - um_lf);
            if (theta_m > 1)
              theta_m = 1.0;
            else if (theta_m < 0)
              theta_m = 0;
          }
        }

        hc(i, j, k, l + 5) = min(theta_p, theta_m) * (hc(i, j, k, l + 5) - f1) + f1;
      }
    }
    return;
  }
  // characteristic method
  auto Q = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s);
  auto F = reinterpret_cast<real (*)[nyy + 2 * 4 - 1][4]>(s + n_var * (nyy + 2 * 4 - 1) * 4);

  for (int kl = kl0 + tz; kl <= kl0 + n_point - 1; kl += n_active) {
    int iSh = kl - kl0; // iSh is the shared index

    const int idx3d = cv.idx3d(i, j, kl);

    const real q0 = cv_data[idx3d];
    const real q1 = cv_data[idx3d + 1 * sz_3d];
    const real q2 = cv_data[idx3d + 2 * sz_3d];
    const real q3 = cv_data[idx3d + 3 * sz_3d];
    const real q4 = cv_data[idx3d + 4 * sz_3d];
    real pk = bv_data[idx3d + 4 * sz_3d];
    const real metric0 = metric_data[idx3d + 6 * sz_3d];
    const real metric1 = metric_data[idx3d + 7 * sz_3d];
    const real metric2 = metric_data[idx3d + 8 * sz_3d];

    const real Uk = (q1 * metric0 + q2 * metric1 + q3 * metric2) / q0;

    Q[0][iSh][tx] = q0;
    Q[1][iSh][tx] = q1;
    Q[2][iSh][tx] = q2;
    Q[3][iSh][tx] = q3;
    Q[4][iSh][tx] = q4;

    F[0][iSh][tx] = Uk;
    F[1][iSh][tx] = fma(q1, Uk, pk * metric0);
    F[2][iSh][tx] = fma(q2, Uk, pk * metric1);
    F[3][iSh][tx] = fma(q3, Uk, pk * metric2);
    F[4][iSh][tx] = pk;

    for (int l = 5; l < n_var; ++l) {
      Q[l][iSh][tx] = cv_data[idx3d + l * sz_3d];
    }

    real cGradK = norm3d(metric0, metric1, metric2);
    if constexpr (mix_model != MixtureModel::Air)
      cGradK *= acoustic_speed_data[idx3d];
    else
      cGradK *= sqrt(gamma_air * R_air * bv_data[idx3d + 5 * sz_3d]);
    // const real lambda0 = abs(Uk) + cGradK;
    // F[5][iSh][tx] = lambda0;
    F[5][iSh][tx] = cGradK;
    F[6][iSh][tx] = jac_data[idx3d];
  }
  __syncthreads();

  if constexpr (mix_model == MixtureModel::Air) {
    // weno_ch_air_yz_piro(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, hc.data(),
    //                     hc.idx3d(i, j, k), hc.size());
    weno_ch_air_yz(param, reinterpret_cast<const real **>(s), i_shared, kx, ky, kz, eps_ref, if_shock, hc.data(),
                   hc.idx3d(i, j, k), hc.size());
  } else {
    real rho_l = Q[0][i_shared][tx], rho_r = Q[0][i_shared + 1][tx];
    // First, compute the Roe average of the half-point variables.
    real temp1 = sqrt(rho_l * rho_r); // temp1 is sqrt(rhoL*rhoR), only used in the next two lines.
    const real rlc{1 / (rho_l + temp1)};
    const real rrc{1 / (temp1 + rho_r)};
    const real um{rlc * Q[1][i_shared][tx] + rrc * Q[1][i_shared + 1][tx]};
    const real vm{rlc * Q[2][i_shared][tx] + rrc * Q[2][i_shared + 1][tx]};
    const real wm{rlc * Q[3][i_shared][tx] + rrc * Q[3][i_shared + 1][tx]};

    real svm[MAX_SPEC_NUMBER] = {};
    for (int l = 0; l < n_var - 5; ++l) {
      svm[l] = rlc * Q[l + 5][i_shared][tx] + rrc * Q[l + 5][i_shared + 1][tx];
    }

    const int ns{param->n_spec};
    temp1 = 0; // temp1 = gas_constant (R)
    for (int l = 0; l < ns; ++l) {
      temp1 += svm[l] * param->gas_const[l];
    }
    // temp1 = R, temp3 = T
    real temp3 = (rlc * F[4][i_shared][tx] + rrc * F[4][i_shared + 1][tx]) / temp1;

    // The MAX_SPEC_NUMBER part of fChar are used for cp_i computation first, and later used as the characteristic flux.
    real fChar[5 + MAX_SPEC_NUMBER];
    real hI_alpI[MAX_SPEC_NUMBER];                         // First used as h_i, later used as alpha_i.
    compute_enthalpy_and_cp(temp3, hI_alpI, fChar, param); // temp3 is T
    real temp2{0};                                         // temp2 = cp
    for (int l = 0; l < ns; ++l) {
      temp2 += svm[l] * fChar[l];
    }
    const real gamma = temp2 / (temp2 - temp1);  // temp1 = R, temp2 = cp. After here, temp2 is not cp anymore.
    const real cm = sqrt(gamma * temp1 * temp3); // temp1 is not R anymore.
    const real gm1{gamma - 1};

    // Next, we compute the left characteristic matrix at i+1/2.
    temp1 = rnorm3d(kx, ky, kz); // temp1 is the norm of the unit normal vector
    kx *= temp1;
    ky *= temp1;
    kz *= temp1;
    const real Uk_bar{kx * um + ky * vm + kz * wm};
    real nx{0}, ny{0}, nz{0}, qx{0}, qy{0}, qz{0};
    if (abs(kx) > 0.8) {
      nx = -kz;
      nz = kx;
    } else {
      ny = kz;
      nz = -ky;
    }
    qx = ky * nz - kz * ny;
    qy = kz * nx - kx * nz;
    qz = kx * ny - ky * nx;
    const real Un = nx * um + ny * vm + nz * wm;
    const real Uq = qx * um + qy * vm + qz * wm;

    // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
    //  const real cm2_inv{1.0 / (cm * cm)};
    temp2 = 1.0 / (cm * cm); // temp2 is 1/(c^2), used in the next loop.
    // Compute the characteristic flux with L.
    // compute the partial derivative of pressure to species density
    for (int l = 0; l < ns; ++l) {
      hI_alpI[l] = gamma * param->gas_const[l] * temp3 - gm1 * hI_alpI[l]; // temp3 is not T anymore.
      // The computations including this alpha_l are all combined with a division by cm2.
      hI_alpI[l] *= temp2;
    }

    // Li Xinliang's flux splitting
    //  const real alpha{gm1 * 0.5 * (um * um + vm * vm + wm * wm)};
    temp3 = 0.5 * gm1 * (um * um + vm * vm + wm * wm); // temp3 = alpha, used in the next loop.
    if (param->inviscid_scheme == 72) {
      const int baseP = i_shared - 3;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 8; ++m) {
        if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[8], sumBetaF[8];
      for (int m = 0; m < 8; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP][tx], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP][tx] * F[0][iP][tx], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 3;
        real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][
                     tx]
                   + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][
                     iP][tx]
                   + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
                + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][
                  tx]
                + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[7];
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx] + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][
                tx]
              + temp1 * sumBetaQ[7];
        vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[7], vMinus[7];
        int iP = i_shared - 3;
        real LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        real LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 7; ++m) {
          iP = i_shared - 3 + m;
          LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
          LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared - 3 + 7;
        LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vMinus[6] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_ref, if_shock);
      }
    } else if (param->inviscid_scheme == 52) {
      const int baseP = i_shared - 2;

      // Find the max spectral radius
      real max_lambda = 0;
      for (int m = 0; m < 6; ++m) {
        if (real lam = abs(F[0][baseP + m][tx]) + F[5][baseP + m][tx]; lam > max_lambda) {
          max_lambda = lam;
        }
      }

      real sumBetaQ[6], sumBetaF[6];
      for (int m = 0; m < 6; ++m) {
        const int iP = baseP + m;

        real sp{0}, sm{0};
        for (int n = 0; n < ns; ++n) {
          sp = fma(hI_alpI[n], Q[5 + n][iP][tx], sp);
          sm = fma(hI_alpI[n], Q[5 + n][iP][tx] * F[0][iP][tx], sm);
        }
        sumBetaQ[m] = sp;
        sumBetaF[m] = sm;
      }

      for (int l = 0; l < 5; ++l) {
        temp1 = 0.5;
        real L[5];
        switch (l) {
          case 0:
            L[0] = (temp3 + Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um + kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          case 1:
            temp1 = 0;
            L[0] = -Un;
            L[1] = nx;
            L[2] = ny;
            L[3] = nz;
            L[4] = 0;
            break;
          case 2:
            temp1 = 0;
            L[0] = -Uq;
            L[1] = qx;
            L[2] = qy;
            L[3] = qz;
            L[4] = 0;
            break;
          case 3:
            temp1 = -1;
            L[0] = 1 - temp3 * temp2;
            L[1] = gm1 * um * temp2;
            L[2] = gm1 * vm * temp2;
            L[3] = gm1 * wm * temp2;
            L[4] = -gm1 * temp2;
            break;
          case 4:
            L[0] = (temp3 - Uk_bar * cm) * temp2 * 0.5;
            L[1] = -(gm1 * um - kx * cm) * temp2 * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * temp2 * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * temp2 * 0.5;
            L[4] = gm1 * temp2 * 0.5;
            break;
          default:
            break;
        }

        real vPlus[5] = {}, vMinus[5] = {};
        // 0-Uk, 1-rho*u*Uk+p*kx, 2-rho*v*Uk+p*ky, 3-rho*w*Uk+p*kz, 4-pk, 5-abs(Uk)+cGradK, 6-jac, 7~9-kx,ky,kz
        int iP = i_shared - 2;
        real LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx]
                   + L[3] * F[3][iP][tx] + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[0];
        real LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
                   + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[0];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
                + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[m];
          LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
                + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[m];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = L[0] * (Q[0][iP][tx] * F[0][iP][tx]) + L[1] * F[1][iP][tx] + L[2] * F[2][iP][tx] + L[3] * F[3][iP][tx]
              + L[4] * ((Q[4][iP][tx] + F[4][iP][tx]) * F[0][iP][tx]) + temp1 * sumBetaF[5];
        LQm = L[0] * Q[0][iP][tx] + L[1] * Q[1][iP][tx] + L[2] * Q[2][iP][tx]
              + L[3] * Q[3][iP][tx] + L[4] * Q[4][iP][tx] + temp1 * sumBetaQ[5];
        vMinus[4] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
      for (int l = 0; l < ns; ++l) {
        real vPlus[5], vMinus[5];
        int iP = i_shared - 2;
        real LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        real LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vPlus[0] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
        for (int m = 1; m < 5; ++m) {
          iP = i_shared - 2 + m;
          LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
          LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
          vPlus[m] = 0.5 * F[6][iP][tx] * (LFm + max_lambda * LQm);
          vMinus[m - 1] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        }
        iP = i_shared + 3;
        LFm = F[0][iP][tx] * (Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx]);
        LQm = Q[5 + l][iP][tx] - svm[l] * Q[0][iP][tx];
        vMinus[4] = 0.5 * F[6][iP][tx] * (LFm - max_lambda * LQm);
        fChar[5 + l] = WENO5(vPlus, vMinus, eps_ref, if_shock);
      }
    } // temp2 is not 1/(c*c) anymore.

    // Project the flux back to physical space
    // We do not compute the right characteristic matrix here, because we explicitly write the components below.
    temp1 = fChar[0] + fChar[3] + fChar[4];
    temp3 = fChar[0] - fChar[4];
    hc(i, j, k, 0) = temp1;
    hc(i, j, k, 1) = um * temp1 - kx * cm * temp3 + nx * fChar[1] + qx * fChar[2];
    hc(i, j, k, 2) = vm * temp1 - ky * cm * temp3 + ny * fChar[1] + qy * fChar[2];
    hc(i, j, k, 3) = wm * temp1 - kz * cm * temp3 + nz * fChar[1] + qz * fChar[2];

    // temp2 is Roe averaged enthalpy
    temp2 = rlc * (Q[4][i_shared][tx] + F[4][i_shared][tx]) + rrc * (Q[4][i_shared + 1][tx] + F[4][i_shared + 1][tx]);
    hc(i, j, k, 4) = temp2 * temp1 - Uk_bar * cm * temp3 + Un * fChar[1] + Uq * fChar[2] - cm * cm / gm1 * fChar[3];

    temp2 = 0;
    for (int l = 0; l < ns; ++l) {
      hc(i, j, k, 5 + l) = svm[l] * temp1 + fChar[l + 5];
      temp2 += hI_alpI[l] * fChar[l + 5];
    }
    hc(i, j, k, 4) -= temp2 * cm * cm / gm1;
  }
}

__global__ void compute_derivative_z(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dqA = zone->dq;
  auto &hcA = zone->hFlux;

  const int sz_dq = dqA.size();
  const int sz_hc = hcA.size();
  real *__restrict__ dq = dqA.data();
  const real *__restrict__ hc = hcA.data();

  const int idx_dq = dqA.idx3d(i, j, k);
  const int idx_c = hcA.idx3d(i, j, k);
  const int idx_m = hcA.idx3d(i, j, k - 1);

  dq[idx_dq + 0 * sz_dq] -= hc[idx_c + 0 * sz_hc] - hc[idx_m + 0 * sz_hc];
  dq[idx_dq + 1 * sz_dq] -= hc[idx_c + 1 * sz_hc] - hc[idx_m + 1 * sz_hc];
  dq[idx_dq + 2 * sz_dq] -= hc[idx_c + 2 * sz_hc] - hc[idx_m + 2 * sz_hc];
  dq[idx_dq + 3 * sz_dq] -= hc[idx_c + 3 * sz_hc] - hc[idx_m + 3 * sz_hc];
  dq[idx_dq + 4 * sz_dq] -= hc[idx_c + 4 * sz_hc] - hc[idx_m + 4 * sz_hc];
  for (int l = 5; l < nv; ++l) {
    dq[idx_dq + l * sz_dq] -= hc[idx_c + l * sz_hc] - hc[idx_m + l * sz_hc];
  }
}

template<MixtureModel mix_model> void compute_convective_term_weno(const Block &block, DZone *zone, DParameter *param,
  int n_var) {
  // The implementation of classic WENO.
  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  auto TPB = dim3(block_dim, 1, 1);
  auto BPG = dim3((extent[0] + 1 - 1) / block_dim + 1, extent[1], extent[2]);
  n_var = max(n_var, 8);
  auto shared_mem = (block_dim + 2 * 4 - 1) * n_var * 2 * sizeof(real); // F+/F-
  compute_convective_term_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
  TPB = dim3(16, 8, 8);
  BPG = dim3((extent[0] - 1) / 16 + 1, (extent[1] - 1) / 8 + 1, (extent[2] - 1) / 8 + 1);
  compute_derivative_x<<<BPG, TPB>>>(zone, param);

  constexpr int ny = 32;
  TPB = dim3(4, ny, 1);
  BPG = dim3((extent[0] - 1) / 4 + 1, (extent[1] + 1 - 1) / ny + 1, extent[2]);
  auto shared_mem1 = (ny + 2 * 4 - 1) * 4 * n_var * 2 * sizeof(real); // F+/F-
  compute_convective_term_weno_y<mix_model><<<BPG, TPB, shared_mem1>>>(zone, param);
  TPB = dim3(16, 8, 8);
  BPG = dim3((extent[0] - 1) / 16 + 1, (extent[1] - 1) / 8 + 1, (extent[2] - 1) / 8 + 1);
  compute_derivative_y<<<BPG, TPB>>>(zone, param);

  if (extent[2] > 1) {
    TPB = dim3(4, 1, ny);
    BPG = dim3((extent[0] - 1) / 4 + 1, extent[1], (extent[2] + 1 - 1) / ny + 1);
    compute_convective_term_weno_z<mix_model><<<BPG, TPB, shared_mem1>>>(zone, param);
    TPB = dim3(16, 8, 8);
    BPG = dim3((extent[0] - 1) / 16 + 1, (extent[1] - 1) / 8 + 1, (extent[2] - 1) / 8 + 1);
    compute_derivative_z<<<BPG, TPB>>>(zone, param);
  }
}

__device__ void positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param,
  int i_shared, real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac) {
  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

  const int ns = n_var - 5;
  const int offset_yq_l = i_shared * (n_var + 2) + 5;
  const int offset_yq_r = (i_shared + 1) * (n_var + 2) + 5;
  real *fc_yq_i = &fc[tid * n_var + 5];

  for (int l = 0; l < ns; ++l) {
    real theta_p = 1.0, theta_m = 1.0;
    // if (idx_in_mesh > -1) {
    const real up = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * fc_yq_i[l];
    if (up < 0) {
      const real up_lf = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * f_1st[tid * ns + l];
      if (abs(up - up_lf) > 1e-20) {
        theta_p = (0 - up_lf) / (up - up_lf);
        if (theta_p > 1)
          theta_p = 1.0;
        else if (theta_p < 0)
          theta_p = 0;
      }
    }
    // }

    // if (idx_in_mesh < max_extent - 1) {
    const real um = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * fc_yq_i[l];
    if (um < 0) {
      const real um_lf = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * f_1st[tid * ns + l];
      if (abs(um - um_lf) > 1e-20) {
        theta_m = (0 - um_lf) / (um - um_lf);
        if (theta_m > 1)
          theta_m = 1.0;
        else if (theta_m < 0)
          theta_m = 0;
      }
    }
    // }

    fc_yq_i[l] = min(theta_p, theta_m) * (fc_yq_i[l] - f_1st[tid * ns + l]) + f_1st[tid * ns + l];
  }
}

__device__ void positive_preserving_limiter_1(int dim, int n_var, const real *cv, int i_shared, const real *jac,
  real dt, real *fci, const real *metric, const real *cc, const real *Fp) {
  const real alpha = dim == 3 ? 1.0 / 3.0 : 0.5;
  const auto *cvl = &cv[i_shared * n_var];
  const auto *cvr = &cv[(i_shared + 1) * n_var];

  for (int l = 0; l < n_var - 5; ++l) {
    real f1{0.0};
    bool f1_computed{false};
    real theta_p = 1.0, theta_m = 1.0;
    const real up = 0.5 * alpha * cvl[l + 5] * jac[i_shared] - dt * fci[l + 5];
    if (up < 0) {
      const real temp1 = abs((metric[i_shared * 3] * cvl[1] +
                              metric[i_shared * 3 + 1] * cvl[2] +
                              metric[i_shared * 3 + 2] * cvl[3]) / cvl[0] +
                             cc[i_shared] * norm3d(metric[i_shared * 3], metric[i_shared * 3 + 1],
                                                   metric[i_shared * 3 + 2])); // spectralRadThis
      const real temp2 = abs((metric[(i_shared + 1) * 3] * cvr[1] +
                              metric[(i_shared + 1) * 3 + 1] * cvr[2] +
                              metric[(i_shared + 1) * 3 + 2] * cvr[3]) / cvr[0] +
                             cc[i_shared + 1] * norm3d(metric[(i_shared + 1) * 3], metric[(i_shared + 1) * 3 + 1],
                                                       metric[(i_shared + 1) * 3 + 2])); // spectralRadNext
      f1 = 0.5 * (Fp[i_shared * n_var + l + 5] + temp1 * cvl[l + 5] * jac[i_shared]) +
           0.5 * (Fp[(i_shared + 1) * n_var + l + 5] - temp2 * cvr[l + 5] * jac[i_shared + 1]);
      f1_computed = true;
      const real up_lf = 0.5 * alpha * cvl[l + 5] * jac[i_shared] - dt * f1;
      if (abs(up - up_lf) > 1e-20) {
        theta_p = (0 - up_lf) / (up - up_lf);
        if (theta_p > 1)
          theta_p = 1.0;
        else if (theta_p < 0)
          theta_p = 0;
      }
    }

    const real um =
        0.5 * alpha * cvr[l + 5] * jac[i_shared + 1] + dt * fci[l + 5];
    if (um < 0) {
      if (!f1_computed) {
        real temp1 = abs((metric[i_shared * 3] * cvl[1] +
                          metric[i_shared * 3 + 1] * cvl[2] +
                          metric[i_shared * 3 + 2] * cvl[3]) / cvl[0] +
                         cc[i_shared] * norm3d(metric[i_shared * 3], metric[i_shared * 3 + 1],
                                               metric[i_shared * 3 + 2])); // spectralRadThis
        real temp2 = abs((metric[(i_shared + 1) * 3] * cvr[1] +
                          metric[(i_shared + 1) * 3 + 1] * cvr[2] +
                          metric[(i_shared + 1) * 3 + 2] * cvr[3]) / cvr[0] +
                         cc[i_shared + 1] * norm3d(metric[(i_shared + 1) * 3], metric[(i_shared + 1) * 3 + 1],
                                                   metric[(i_shared + 1) * 3 + 2])); // spectralRadNext
        f1 = 0.5 * (Fp[i_shared * n_var + l + 5] + temp1 * cvl[l + 5] * jac[i_shared]) +
             0.5 *
             (Fp[(i_shared + 1) * n_var + l + 5] - temp2 * cvr[l + 5] * jac[i_shared + 1]);
      }
      const real um_lf =
          0.5 * alpha * cvr[l + 5] * jac[i_shared + 1] + dt * f1;
      if (abs(um - um_lf) > 1e-20) {
        theta_m = (0 - um_lf) / (um - um_lf);
        if (theta_m > 1)
          theta_m = 1.0;
        else if (theta_m < 0)
          theta_m = 0;
      }
    }

    fci[l + 5] = min(theta_p, theta_m) * (fci[l + 5] - f1) + f1;
  }
}

__device__ real WENO5(const real *vp, const real *vm, real eps, bool if_shock) {
  if (if_shock) {
    constexpr real one6th{1.0 / 6};
    real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
    real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
    real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
    constexpr real thirteen12th{13.0 / 12};
    real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
                 0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
    real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
                 0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
    real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
                 0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
    constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
    // real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
    // real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
    // real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
    // real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
    real a0{three10th / ((eps + beta0) * (eps + beta0))};
    real a1{six10th / ((eps + beta1) * (eps + beta1))};
    real a2{one10th / ((eps + beta2) * (eps + beta2))};
    const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
    v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
    v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
    beta0 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
            0.25 * (3 * vm[2] - 4 * vm[3] + vm[4]) * (3 * vm[2] - 4 * vm[3] + vm[4]);
    beta1 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
            0.25 * (vm[1] - vm[3]) * (vm[1] - vm[3]);
    beta2 = thirteen12th * (vm[0] + vm[2] - 2 * vm[1]) * (vm[0] + vm[2] - 2 * vm[1]) +
            0.25 * (vm[0] - 4 * vm[1] + 3 * vm[2]) * (vm[0] - 4 * vm[1] + 3 * vm[2]);
    // tau5sqr = (beta0 - beta2) * (beta0 - beta2);
    // a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
    // a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
    // a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
    a0 = one10th / ((eps + beta0) * (eps + beta0));
    a1 = six10th / ((eps + beta1) * (eps + beta1));
    a2 = three10th / ((eps + beta2) * (eps + beta2));
    const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

    return fPlus + fMinus;
  }
  constexpr real one6th{1.0 / 6};
  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
  const real fPlus{0.3 * v0 + 0.6 * v1 + 0.1 * v2};

  v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
  v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
  v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
  const real fMinus{0.1 * v0 + 0.6 * v1 + 0.3 * v2};

  return fPlus + fMinus;
}

__device__ real WENO7(const real *vp, const real *vm, real eps, bool if_shock) {
  if (if_shock) {
    // Shocked, use WENO
    constexpr real one6th{1.0 / 6};
    constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};

    // Re-organize the data to improve locality
    // 1st order derivative
    real s1{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
    // 2nd order derivative
    real s2{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
    // 3rd order derivative
    real s3{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
    real beta0{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4]);
    s2 = vp[2] - 2 * vp[3] + vp[4];
    s3 = -vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4];
    real beta1{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5]);
    s3 = -vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5];
    real beta2{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    s1 = one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6]);
    s2 = 2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6];
    s3 = -vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6];
    real beta3{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

    // real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
    real tau7sqr = beta0 + 3 * beta1 - 3 * beta2 - beta3;
    tau7sqr *= tau7sqr;
    constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
    real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
    real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
    real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
    real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};
    // real a0 = c0 / ((eps + beta0) * (eps + beta0));
    // real a1 = c1 / ((eps + beta1) * (eps + beta1));
    // real a2 = c2 / ((eps + beta2) * (eps + beta2));
    // real a3 = c3 / ((eps + beta3) * (eps + beta3));

    constexpr real one12th{1.0 / 12};
    real v0{-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3]};
    real v1{vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4]};
    real v2{-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5]};
    real v3{3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6]};
    const real fPlus{one12th * (a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

    // Minus part
    s1 = one6th * (-2 * vm[6] + 9 * vm[5] - 18 * vm[4] + 11 * vm[3]);
    s2 = -vm[6] + 4 * vm[5] - 5 * vm[4] + 2 * vm[3];
    s3 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
    beta0 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (vm[5] - 6 * vm[4] + 3 * vm[3] + 2 * vm[2]);
    s2 = vm[4] - 2 * vm[3] + vm[2];
    s3 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
    beta1 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (-2 * vm[4] - 3 * vm[3] + 6 * vm[2] - vm[1]);
    s3 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
    beta2 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    s1 = one6th * (-11 * vm[3] + 18 * vm[2] - 9 * vm[1] + 2 * vm[0]);
    s2 = 2 * vm[3] - 5 * vm[2] + 4 * vm[1] - vm[0];
    s3 = -vm[3] + 3 * vm[2] - 3 * vm[1] + vm[0];
    beta3 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

    // tau7sqr = (beta0 - beta3) * (beta0 - beta3);
    tau7sqr = beta0 + 3 * beta1 - 3 * beta2 - beta3;
    tau7sqr *= tau7sqr;
    a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
    a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
    a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
    a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));
    // a0 = c0 / ((eps + beta0) * (eps + beta0));
    // a1 = c1 / ((eps + beta1) * (eps + beta1));
    // a2 = c2 / ((eps + beta2) * (eps + beta2));
    // a3 = c3 / ((eps + beta3) * (eps + beta3));

    v0 = -3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3];
    v1 = vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2];
    v2 = -vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1];
    v3 = 3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0];
    const real fMinus{one12th * (a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

    return fPlus + fMinus;
  }
  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
  constexpr real one12th{1.0 / 12};
  real v3{0}, v2{0}, v1{0}, v0{0};
  v3 = 3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6];
  v2 = -vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5];
  v1 = vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4];
  v0 = -3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3];
  const real fPlus{one12th * (c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3)};

  // Minus part
  v0 = -3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3];
  v1 = vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2];
  v2 = -vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1];
  v3 = 3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0];
  const real fMinus{one12th * (c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3)};

  return fPlus + fMinus;
}

template void compute_convective_term_weno<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var);

template void compute_convective_term_weno<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var);
}

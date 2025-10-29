#include "SpeciesStat.cuh"
#include <mpi.h>
#include "../DParameter.cuh"

__device__ void cfd::collect_species_dissipation_rate(DZone *zone, const DParameter *param, int i, int j, int k) {
  auto &collect = zone->collect_scalar_budget;
  const auto &sv = zone->sv;
  const auto &metric = zone->metric;

  const real xi_x = metric(i, j, k, 0);
  const real xi_y = metric(i, j, k, 1);
  const real xi_z = metric(i, j, k, 2);
  const real eta_x = metric(i, j, k, 3);
  const real eta_y = metric(i, j, k, 4);
  const real eta_z = metric(i, j, k, 5);
  const real zeta_x = metric(i, j, k, 6);
  const real zeta_y = metric(i, j, k, 7);
  const real zeta_z = metric(i, j, k, 8);

  for (int i_s = 0; i_s < param->n_species_stat; ++i_s) {
    const int is = param->specStatIndex[i_s];
    const int l = i_s * ScalarDissipationRate::n_collect;

    const real zx = 0.5 * (xi_x * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_x * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_x * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zy = 0.5 * (xi_y * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_y * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_y * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zz = 0.5 * (xi_z * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_z * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_z * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const auto rhoD = zone->rho_D(i, j, k, is);

    // Rho*D*GradH2*GradH2
    collect(i, j, k, l) += rhoD * (zx * zx + zy * zy + zz * zz);
    // Rho*D*H2x
    collect(i, j, k, l + 1) += rhoD * zx;
    // Rho*D*H2y
    collect(i, j, k, l + 2) += rhoD * zy;
    // Rho*D*H2z
    collect(i, j, k, l + 3) += rhoD * zz;
    // Rho*D
    collect(i, j, k, l + 4) += rhoD;
  }
  for (int i_s = 0; i_s < param->n_ps; ++i_s) {
    const int is = param->i_ps + i_s;
    const int l = (param->n_species_stat + i_s) * ScalarDissipationRate::n_collect;

    const real zx = 0.5 * (xi_x * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_x * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_x * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zy = 0.5 * (xi_y * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_y * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_y * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zz = 0.5 * (xi_z * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_z * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_z * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const auto rhoD = zone->mul(i, j, k) / param->sc_ps[i_s];

    // Rho*D*GradH2*GradH2
    collect(i, j, k, l) += rhoD * (zx * zx + zy * zy + zz * zz);
    // Rho*D*H2x
    collect(i, j, k, l + 1) += rhoD * zx;
    // Rho*D*H2y
    collect(i, j, k, l + 2) += rhoD * zy;
    // Rho*D*H2z
    collect(i, j, k, l + 3) += rhoD * zz;
    // Rho*D
    collect(i, j, k, l + 4) += rhoD;
  }
}

__device__ void
cfd::collect_species_velocity_correlation(DZone *zone, const DParameter *param, int i, int j, int k) {
  auto &collect = zone->collect_scalar_vel_correlation;
  const auto &sv = zone->sv;
  const auto &bv = zone->bv;

  const auto rho = bv(i, j, k, 0), u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3);
  for (int i_s = 0; i_s < param->n_species_stat; ++i_s) {
    const int is = param->specStatIndex[i_s];
    const int l = i_s * ScalarVelocityCorrelation::n_collect;

    const real z = sv(i, j, k, is);
    collect(i, j, k, l) += rho * u * z;
    collect(i, j, k, l + 1) += rho * v * z;
    collect(i, j, k, l + 2) += rho * w * z;
  }
  for (int i_s = 0; i_s < param->n_ps; ++i_s) {
    const int is = param->i_ps + i_s;
    const int l = (param->n_species_stat + i_s) * ScalarVelocityCorrelation::n_collect;

    const real z = sv(i, j, k, is);
    collect(i, j, k, l) += rho * u * z;
    collect(i, j, k, l + 1) += rho * v * z;
    collect(i, j, k, l + 2) += rho * w * z;
  }
}

void cfd::ScalarFlucBudget::read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count,
  MPI_Datatype ty, MPI_Status *status) {
  MPI_File_read_at(fp, offset_read, zone.collect_scalar_budget[index], count, ty, status);
}

void cfd::ScalarFlucBudget::copy_to_device(Field &zone, int nv, long long sz) {
  cudaMemcpy(zone.h_ptr->collect_scalar_budget.data(), zone.collect_scalar_budget.data(), sz * nv,
             cudaMemcpyHostToDevice);
}

void cfd::ScalarFlucBudget::copy_to_host(Field &zone, int nv, long long sz) {
  cudaMemcpy(zone.collect_scalar_budget.data(), zone.h_ptr->collect_scalar_budget.data(), sz * nv,
             cudaMemcpyDeviceToHost);
}

void cfd::ScalarFlucBudget::write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty,
  MPI_Status *status) {
  MPI_File_write_at(fp, offset, zone.collect_scalar_budget.data(), count, ty, status);
}

__device__ void cfd::collect_scalar_fluc_budget(DZone *zone, const DParameter *param, int i, int j, int k) {
  auto &collect = zone->collect_scalar_budget;
  const auto &sv = zone->sv;
  const auto &bv = zone->bv;
  const auto &tau = zone->collect_tke_budget;
  const auto &metric = zone->metric;

  const real xi_x = metric(i, j, k, 0);
  const real xi_y = metric(i, j, k, 1);
  const real xi_z = metric(i, j, k, 2);
  const real eta_x = metric(i, j, k, 3);
  const real eta_y = metric(i, j, k, 4);
  const real eta_z = metric(i, j, k, 5);
  const real zeta_x = metric(i, j, k, 6);
  const real zeta_y = metric(i, j, k, 7);
  const real zeta_z = metric(i, j, k, 8);

  const auto rho = bv(i, j, k, 0), u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3);
  const int ns_stat = param->n_species_stat;
  // pressure gradient
  real dXi = bv(i + 1, j, k, 4) - bv(i - 1, j, k, 4);
  real dEta = bv(i, j + 1, k, 4) - bv(i, j - 1, k, 4);
  real dZeta = bv(i, j, k + 1, 4) - bv(i, j, k - 1, 4);
  real dp_dx = 0.5 * (xi_x * dXi + eta_x * dEta + zeta_x * dZeta);
  real dp_dy = 0.5 * (xi_y * dXi + eta_y * dEta + zeta_y * dZeta);
  real dp_dz = 0.5 * (xi_z * dXi + eta_z * dEta + zeta_z * dZeta);
  // dTau_{1j}/dx_j
  real d_dx = 0.5 * (xi_x * (tau(i + 1, j, k, 0) - tau(i - 1, j, k, 0)) +
                     eta_x * (tau(i, j + 1, k, 0) - tau(i, j - 1, k, 0)) +
                     zeta_x * (tau(i, j, k + 1, 0) - tau(i, j, k - 1, 0)));
  real d_dy = 0.5 * (xi_y * (tau(i + 1, j, k, 1) - tau(i - 1, j, k, 1)) +
                     eta_y * (tau(i, j + 1, k, 1) - tau(i, j - 1, k, 1)) +
                     zeta_y * (tau(i, j, k + 1, 1) - tau(i, j, k - 1, 1)));
  real d_dz = 0.5 * (xi_z * (tau(i + 1, j, k, 2) - tau(i - 1, j, k, 2)) +
                     eta_z * (tau(i, j + 1, k, 2) - tau(i, j - 1, k, 2)) +
                     zeta_z * (tau(i, j, k + 1, 2) - tau(i, j, k - 1, 2)));
  const real dTau1j_dxj = d_dx + d_dy + d_dz;
  // dTau_{2j}/dx_j
  d_dx = 0.5 * (xi_x * (tau(i + 1, j, k, 2) - tau(i - 1, j, k, 2)) +
                eta_x * (tau(i, j + 1, k, 2) - tau(i, j - 1, k, 2)) +
                zeta_x * (tau(i, j, k + 1, 2) - tau(i, j, k - 1, 2)));
  d_dy = 0.5 * (xi_y * (tau(i + 1, j, k, 4) - tau(i - 1, j, k, 4)) +
                eta_y * (tau(i, j + 1, k, 4) - tau(i, j - 1, k, 4)) +
                zeta_y * (tau(i, j, k + 1, 4) - tau(i, j, k - 1, 4)));
  d_dz = 0.5 * (xi_z * (tau(i + 1, j, k, 5) - tau(i - 1, j, k, 5)) +
                eta_z * (tau(i, j + 1, k, 5) - tau(i, j - 1, k, 5)) +
                zeta_z * (tau(i, j, k + 1, 5) - tau(i, j, k - 1, 5)));
  const real dTau2j_dxj = d_dx + d_dy + d_dz;
  // dTau_{3j}/dx_j
  d_dx = 0.5 * (xi_x * (tau(i + 1, j, k, 2) - tau(i - 1, j, k, 2)) +
                eta_x * (tau(i, j + 1, k, 2) - tau(i, j - 1, k, 2)) +
                zeta_x * (tau(i, j, k + 1, 2) - tau(i, j, k - 1, 2)));
  d_dy = 0.5 * (xi_y * (tau(i + 1, j, k, 4) - tau(i - 1, j, k, 4)) +
                eta_y * (tau(i, j + 1, k, 4) - tau(i, j - 1, k, 4)) +
                zeta_y * (tau(i, j, k + 1, 4) - tau(i, j, k - 1, 4)));
  d_dz = 0.5 * (xi_z * (tau(i + 1, j, k, 5) - tau(i - 1, j, k, 5)) +
                eta_z * (tau(i, j + 1, k, 5) - tau(i, j - 1, k, 5)) +
                zeta_z * (tau(i, j, k + 1, 5) - tau(i, j, k - 1, 5)));
  const real dTau3j_dxj = d_dx + d_dy + d_dz;

  for (int i_s = 0; i_s < ns_stat + param->n_ps; ++i_s) {
    const bool is_species = i_s < ns_stat;
    const int is = is_species ? param->specStatIndex[i_s] : param->i_ps + i_s - ns_stat;
    const int l = i_s * ScalarFlucBudget::n_collect;

    const real z = sv(i, j, k, is);
    collect(i, j, k, l) += rho * u * z;
    collect(i, j, k, l + 1) += rho * v * z;
    collect(i, j, k, l + 2) += rho * w * z;
    collect(i, j, k, l + 3) += rho * u * z * z;
    collect(i, j, k, l + 4) += rho * v * z * z;
    collect(i, j, k, l + 5) += rho * w * z * z;

    const real zx = 0.5 * (xi_x * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_x * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_x * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zy = 0.5 * (xi_y * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_y * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_y * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const real zz = 0.5 * (xi_z * (sv(i + 1, j, k, is) - sv(i - 1, j, k, is)) +
                           eta_z * (sv(i, j + 1, k, is) - sv(i, j - 1, k, is)) +
                           zeta_z * (sv(i, j, k + 1, is) - sv(i, j, k - 1, is)));
    const auto rhoD = is_species ? zone->rho_D(i, j, k, is) : zone->mul(i, j, k) / param->sc_ps[is];

    // Rho*D*GradZ*GradZ
    collect(i, j, k, l + 6) += rhoD * (zx * zx + zy * zy + zz * zz);
    // Rho*D*Zx
    collect(i, j, k, l + 7) += rhoD * zx;
    // Rho*D*Zy
    collect(i, j, k, l + 8) += rhoD * zy;
    // Rho*D*Zz
    collect(i, j, k, l + 9) += rhoD * zz;
    // Rho*D
    collect(i, j, k, l + 10) += rhoD;
    // Rho*D*Zx
    collect(i, j, k, l + 11) += rhoD * z * zx;
    // Rho*D*Zy
    collect(i, j, k, l + 12) += rhoD * z * zy;
    // Rho*D*Zz
    collect(i, j, k, l + 13) += rhoD * z * zz;
    // Rho*u*u*z
    collect(i, j, k, l + 14) += rho * u * u * z;
    // Rho*u*v*z
    collect(i, j, k, l + 15) += rho * u * v * z;
    // Rho*u*w*z
    collect(i, j, k, l + 16) += rho * u * w * z;
    // Rho*v*v*z
    collect(i, j, k, l + 17) += rho * v * v * z;
    // Rho*v*w*z
    collect(i, j, k, l + 18) += rho * v * w * z;
    // Rho*w*w*z
    collect(i, j, k, l + 19) += rho * w * w * z;

    // pressure gradient
    // z*dp/dx
    collect(i, j, k, l + 20) += z * dp_dx;
    // z*dp/dy
    collect(i, j, k, l + 21) += z * dp_dy;
    // z*dp/dz
    collect(i, j, k, l + 22) += z * dp_dz;

    // z*gradTau1j
    collect(i, j, k, l + 23) += z * dTau1j_dxj;
    // z*gradTau2j
    collect(i, j, k, l + 24) += z * dTau2j_dxj;
    // z*gradTau3j
    collect(i, j, k, l + 25) += z * dTau3j_dxj;

    // Diffusion flux = d(rho*D*dZ/dx_j)/dx_j = d(rho*D)/dx_j*dZ/dx_j + rho*D*d^2Z/dx_j^2
    // For species, the rhoD has gradients, while for passive scalar, rhoD is constant.
    // gradient of z has been computed.
    real diffuseFlux = 0.5 * rhoD * ((sv(i + 1, j, k, is) - 2 * sv(i, j, k, is) + sv(i - 1, j, k, is)) *
                                     (xi_x * xi_x + xi_y * xi_y + xi_z * xi_z) +
                                     (sv(i, j + 1, k, is) - 2 * sv(i, j, k, is) + sv(i, j - 1, k, is)) *
                                     (eta_x * eta_x + eta_y * eta_y + eta_z * eta_z) +
                                     (sv(i, j, k + 1, is) - 2 * sv(i, j, k, is) + sv(i, j, k - 1, is)) *
                                     (zeta_x * zeta_x + zeta_y * zeta_y + zeta_z * zeta_z) +
                                     (sv(i + 1, j + 1, k, is) - sv(i - 1, j + 1, k, is) - sv(i + 1, j - 1, k, is) + sv(
                                        i - 1, j - 1, k, is)) * (xi_x * eta_x + xi_y * eta_y + xi_z * eta_z) +
                                     (sv(i + 1, j, k + 1, is) - sv(i - 1, j, k + 1, is) - sv(i + 1, j, k - 1, is) + sv(
                                        i - 1, j, k - 1, is)) * (xi_x * zeta_x + xi_y * zeta_y + xi_z * zeta_z) +
                                     (sv(i, j + 1, k + 1, is) - sv(i, j - 1, k + 1, is) - sv(i, j + 1, k - 1, is) + sv(
                                        i, j - 1, k - 1, is)) * (eta_x * zeta_x + eta_y * zeta_y + eta_z * zeta_z));
    if (is_species) {
      dXi = zone->rho_D(i + 1, j, k, is) - zone->rho_D(i - 1, j, k, is);
      dEta = zone->rho_D(i, j + 1, k, is) - zone->rho_D(i, j - 1, k, is);
      dZeta = zone->rho_D(i, j, k + 1, is) - zone->rho_D(i, j, k - 1, is);
      d_dx = 0.5 * (xi_x * dXi + eta_x * dEta + zeta_x * dZeta);
      d_dy = 0.5 * (xi_y * dXi + eta_y * dEta + zeta_y * dZeta);
      d_dz = 0.5 * (xi_z * dXi + eta_z * dEta + zeta_z * dZeta);
      diffuseFlux += d_dx * zx + d_dy * zy + d_dz * zz;
    }
    collect(i, j, k, l + 26) += u * diffuseFlux;
    collect(i, j, k, l + 27) += v * diffuseFlux;
    collect(i, j, k, l + 28) += w * diffuseFlux;
  }
}

void cfd::ScalarDissipationRate::read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index, int count,
  MPI_Datatype ty, MPI_Status *status) {
  MPI_File_read_at(fp, offset_read, zone.collect_scalar_budget[index], count, ty, status);
}

void cfd::ScalarDissipationRate::copy_to_device(Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.h_ptr->collect_scalar_budget.data(), zone.collect_scalar_budget.data(), sz * nv,
             cudaMemcpyHostToDevice);
}

void cfd::ScalarDissipationRate::copy_to_host(Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.collect_scalar_budget.data(), zone.h_ptr->collect_scalar_budget.data(), sz * nv,
             cudaMemcpyDeviceToHost);
}

void cfd::ScalarDissipationRate::write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty,
  MPI_Status *status) {
  MPI_File_write_at(fp, offset, zone.collect_scalar_budget.data(), count, ty, status);
}

void cfd::ScalarVelocityCorrelation::read(const MPI_File &fp, MPI_Offset offset_read, Field &zone, int index,
  int count, MPI_Datatype ty, MPI_Status *status) {
  MPI_File_read_at(fp, offset_read, zone.collect_scalar_vel_correlation[index], count, ty, status);
}

void cfd::ScalarVelocityCorrelation::copy_to_device(Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.h_ptr->collect_scalar_vel_correlation.data(), zone.collect_scalar_vel_correlation.data(), sz * nv,
             cudaMemcpyHostToDevice);
}

void cfd::ScalarVelocityCorrelation::copy_to_host(Field &zone, int nv, long long int sz) {
  cudaMemcpy(zone.collect_scalar_vel_correlation.data(), zone.h_ptr->collect_scalar_vel_correlation.data(), sz * nv,
             cudaMemcpyDeviceToHost);
}

void
cfd::ScalarVelocityCorrelation::write(const MPI_File &fp, MPI_Offset offset, Field &zone, int count, MPI_Datatype ty,
  MPI_Status *status) {
  MPI_File_write_at(fp, offset, zone.collect_scalar_vel_correlation.data(), count, ty, status);
}

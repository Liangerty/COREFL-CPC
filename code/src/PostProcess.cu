#include "PostProcess.h"
#include "Field.h"
#include <filesystem>
#include <fstream>

void
cfd::wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<Field> &field, const Parameter &parameter) {
  const std::filesystem::path out_dir("output/wall");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  const auto path_name = out_dir.string();

  int size{mesh[0].mx};
  for (int blk = 1; blk < mesh.n_block; ++blk) {
    if (mesh[blk].mx > size) {
      size = mesh[blk].mx;
    }
  }
  std::vector<double> friction(size, 0);
  std::vector<double> heat_flux(size, 0);
  real *cf = nullptr, *qw = nullptr;
  cudaMalloc(&cf, size * sizeof(real));
  cudaMalloc(&qw, sizeof(real) * size);

  const double rho_inf = parameter.get_real("rho_inf");
  const double v_inf = parameter.get_real("v_inf");
  const double dyn_pressure = 0.5 * rho_inf * v_inf * v_inf;
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    auto &block = mesh[blk];
    const int mx{block.mx};

    dim3 bpg((mx - 1) / 128 + 1, 1, 1);
    wall_friction_heatFlux_2d<<<bpg, 128>>>(field[blk].d_ptr, cf, qw, dyn_pressure);
    cudaMemcpy(friction.data(), cf, size * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(heat_flux.data(), qw, size * sizeof(real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::ofstream f(path_name + "/friction_heatflux.dat");
    f << "variables = \"x\", \"cf\", \"qw\"\n";
    for (int i = 0; i < mx; ++i) {
      f << block.x(i, 0, 0) << '\t' << friction[i] << '\t' << heat_flux[i] << '\n';
    }
    f.close();
  }
  cudaFree(cf);
  cudaFree(qw);
}

__global__ void cfd::wall_friction_heatFlux_2d(DZone *zone, real *friction, real *heat_flux, real dyn_pressure) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= zone->mx) return;

  auto &pv = zone->bv;

  auto &metric = zone->metric;
  const real xi_x = metric(i, 0, 0, 0), xi_y = metric(i, 0, 0, 1);
  const real eta_x = metric(i, 0, 0, 3), eta_y = metric(i, 0, 0, 4);

  const real viscosity = zone->mul(i, 0, 0);
  const double u_parallel_wall = (xi_x * pv(i, 1, 0, 1) + xi_y * pv(i, 1, 0, 2)) / sqrt(xi_x * xi_x + xi_y * xi_y);
  const double grad_eta = sqrt(eta_x * eta_x + eta_y * eta_y);
  const double du_normal_wall = u_parallel_wall * grad_eta;
  // dimensionless fiction coefficient, cf
  friction[i] = viscosity * du_normal_wall / dyn_pressure;

  constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
  const double conductivity = zone->mul(i, 0, 0) * cp / 0.72;
  // dimensional heat flux
  heat_flux[i] = conductivity * (pv(i, 1, 0, 5) - pv(i, 0, 0, 5)) * grad_eta;

}

void cfd::wall_friction_heatFlux_3d(const Mesh &mesh, const std::vector<Field> &field,
                                    const Parameter &parameter, DParameter *param) {
  const std::filesystem::path out_dir("output");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  const auto path_name = out_dir.string();

  bool stat_on{parameter.get_bool("if_collect_statistics")};
  bool spanwise_ave{parameter.get_bool("perform_spanwise_average")};
  for (int b = 0; b < mesh.n_block; ++b) {
    int mx{mesh[b].mx}, mz{mesh[b].mz};
    if (spanwise_ave) {
      mz = 1;
    }

    ggxl::VectorField2DHost<real> cfQw_host;
    printf("mx=%d,mz=%d\n", mx, mz);
    cfQw_host.allocate_memory(mx, mz, 2, 0);
    ggxl::VectorField2D<real> cfQw_device_hPtr;
    ggxl::VectorField2D<real> *cfQw_device = nullptr;
    cfQw_device_hPtr.allocate_memory(mx, mz, 2, 0);
    cudaMalloc(&cfQw_device, sizeof(ggxl::VectorField2D<real>));
    cudaMemcpy(cfQw_device, &cfQw_device_hPtr, sizeof(ggxl::VectorField2D<real>), cudaMemcpyHostToDevice);

    dim3 tpb(32, 1, 32);
    if (spanwise_ave) {
      tpb = dim3{128, 1, 1};
    }
    dim3 bpg((mx - 1) / tpb.x + 1, 1, (mz - 1) / tpb.z + 1);


    wall_friction_heatFlux_3d<<<bpg, tpb>>>(field[b].d_ptr, cfQw_device, param, stat_on, spanwise_ave);
    cudaMemcpy(cfQw_host.data(), cfQw_device_hPtr.data(), mx * mz * 2 * sizeof(real), cudaMemcpyDeviceToHost);
    if (!spanwise_ave) {
      std::ofstream f(path_name + "/friction_heatFlux-block-" + std::to_string(b) + ".dat");
      f << "variables = \"x\", \"z\", \"cf\", \"y_plus\"\n";
      f << "zone,i=" << mx << ",j=" << mz << ",f=point\n";
      for (int kk = 0; kk < mz; ++kk) {
        for (int ii = 0; ii < mx; ++ii) {
          f << mesh[b].x(ii, 0, kk) << '\t' << mesh[b].z(ii, 0, kk) << '\t' << cfQw_host(ii, kk, 0) << '\t'
            << cfQw_host(ii, kk, 1) << '\n';
        }
      }
      f.close();
    } else {
      std::ofstream f(path_name + "/spanaveraged_friction_heatFlux-block-" + std::to_string(b) + ".dat");
      f << "variables = \"x\", \"cf\", \"y_plus\"\n";
      for (int ii = 0; ii < mx; ++ii) {
        f << mesh[b].x(ii, 0, 0) << '\t' << cfQw_host(ii, 0, 0) << '\t' << cfQw_host(ii, 0, 1) << '\n';
      }
      f.close();
    }


    cfQw_host.deallocate_memory();
    cfQw_device_hPtr.deallocate_memory();
  }
}

__global__ void
cfd::wall_friction_heatFlux_3d(DZone *zone, ggxl::VectorField2D<real> *cfQw, const DParameter *param, bool stat_on,
                               bool spanwise_ave) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= zone->mx || k >= zone->mz) return;

  constexpr int j = 1;
  auto &metric = zone->metric;
  const real d_wini = rnorm3d(metric(i, 0, k, 3), metric(i, 0, k, 4), metric(i, 0, k, 5));

  real u, v, w;
  real rho_w;
  const real dy = zone->y(i, j, k) - zone->y(i, j - 1, k);
  if (!stat_on) {
    auto &pv = zone->bv;
    u = pv(i, j, k, 1), v = pv(i, j, k, 2), w = pv(i, j, k, 3);
    rho_w = pv(i, 0, k, 0);
  } else {
//    auto &pv = zone->mean_value;
    auto &pv = zone->stat_favre_1st;
    u = pv(i, j, k, 0), v = pv(i, j, k, 1), w = pv(i, j, k, 2);
    rho_w = zone->stat_reynolds_1st(i, 0, k, 0);
  }
  const real rho_ref = param->rho_ref, v_ref = param->v_ref;
  gxl::Matrix<real, 3, 3, 1> bdjin;
  real d1 = metric(i, 0, k, 3);
  real d2 = metric(i, 0, k, 4);
  real d3 = metric(i, 0, k, 5);
  real kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(1, 1) = d1 / kk;
  bdjin(1, 2) = d2 / kk;
  bdjin(1, 3) = d3 / kk;

  d1 = bdjin(1, 2) - bdjin(1, 3);
  d2 = bdjin(1, 3) - bdjin(1, 1);
  d3 = bdjin(1, 1) - bdjin(1, 2);
  kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(2, 1) = d1 / kk;
  bdjin(2, 2) = d2 / kk;
  bdjin(2, 3) = d3 / kk;

  d1 = bdjin(1, 2) * bdjin(2, 3) - bdjin(1, 3) * bdjin(2, 2);
  d2 = bdjin(1, 3) * bdjin(2, 1) - bdjin(1, 1) * bdjin(2, 3);
  d3 = bdjin(1, 1) * bdjin(2, 2) - bdjin(1, 2) * bdjin(2, 1);
  kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  bdjin(3, 1) = d1 / kk;
  bdjin(3, 2) = d2 / kk;
  bdjin(3, 3) = d3 / kk;

  const real vt = bdjin(2, 1) * u + bdjin(2, 2) * v + bdjin(2, 3) * w;
  const real vs = bdjin(3, 1) * u + bdjin(3, 2) * v + bdjin(3, 3) * w;
  const real velocity_tau = sqrt(vt * vt + vs * vs);

  const real tau = velocity_tau / d_wini * zone->mul(i, 0, k);
  const real cf = tau / (0.5 * (rho_ref * v_ref * v_ref));
  const real u_tau = sqrt(tau / rho_w);
  const real y_plus = rho_w * u_tau * dy / zone->mul(i, 0, k);

  (*cfQw)(i, k, 0) = cf;
  (*cfQw)(i, k, 1) = y_plus;
}

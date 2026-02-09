#include "ViscousScheme.cuh"
#include "DataCommunication.cuh"
#include "Constants.h"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model> void compute_viscous_flux(const Mesh &mesh, std::vector<Field> &field,
  DParameter *param, const Parameter &parameter) {
  const int viscous_order = parameter.get_int("viscous_order");

  if (viscous_order == 0)
    return;

  if (viscous_order == 2) {
    for (int b = 0; b < mesh.n_block; ++b) {
      const auto &block = mesh[b];
      const auto mx = block.mx, my = block.my, mz = block.mz;
      const int dim{mz == 1 ? 2 : 3};

      dim3 tpb = {32, 8, 2};
      if (dim == 2)
        tpb = {32, 16, 1};
      dim3 BPG{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};

      auto bpg = dim3(mx/*+1-1*/ / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
      auto zone = field[b].d_ptr;
      compute_fv_2nd_order<mix_model><<<bpg, tpb>>>(zone, param);
      compute_dFv_dx<<<BPG, tpb>>>(zone, param);

      bpg = dim3((mx - 1) / tpb.x + 1, my/*+1-1*/ / tpb.y + 1, (mz - 1) / tpb.z + 1);
      compute_gv_2nd_order<mix_model><<<bpg, tpb>>>(zone, param);
      compute_dGv_dy<<<BPG, tpb>>>(zone, param);

      if (dim == 3) {
        dim3 TPB = {32, 8, 2};
        bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
        compute_hv_2nd_order<mix_model><<<bpg, TPB>>>(zone, param);

        compute_dHv_dz<<<BPG, tpb>>>(zone, param);
      }
    }
  } else if (viscous_order == 8) {
    for (int b = 0; b < mesh.n_block; ++b) {
      const auto &block = mesh[b];
      const auto mx = block.mx, my = block.my, mz = block.mz;
      const int dim{mz == 1 ? 2 : 3};

      dim3 tpb = {32, 8, 2};
      if (dim == 2)
        tpb = {32, 16, 1};
      auto bpg = dim3((mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
      compute_viscous_flux_collocated<mix_model, 8><<<bpg, tpb>>>(field[b].d_ptr, param);

      if constexpr (mix_model != MixtureModel::Air)
        compute_viscous_flux_collocated_scalar<8><<<bpg, tpb>>>(field[b].d_ptr, param);
    }
    cudaDeviceSynchronize();
    // After computing the values on the nodes, we exchange the fv, gv, hv to acquire values on ghost grids.
    exchange_value(mesh, field, parameter, param, 1);

    for (int b = 0; b < mesh.n_block; ++b) {
      const auto &block = mesh[b];
      const auto mx = block.mx, my = block.my, mz = block.mz;
      const int dim{mz == 1 ? 2 : 3};

      dim3 tpb = {32, 8, 2};
      if (dim == 2)
        tpb = {32, 16, 1};
      auto bpg = dim3((mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
      compute_viscous_flux_derivative<8><<<bpg,tpb>>>(field[b].d_ptr, param);
    }
  }
}

__global__ void compute_dFv_dx(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &fv = zone->vis_flux;
  dq(i, j, k, 1) += fv(i, j, k, 0) - fv(i - 1, j, k, 0);
  dq(i, j, k, 2) += fv(i, j, k, 1) - fv(i - 1, j, k, 1);
  dq(i, j, k, 3) += fv(i, j, k, 2) - fv(i - 1, j, k, 2);
  dq(i, j, k, 4) += fv(i, j, k, 3) - fv(i - 1, j, k, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += fv(i, j, k, l - 1) - fv(i - 1, j, k, l - 1);
  }
}

__global__ void compute_dGv_dy(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &gv = zone->vis_flux;
  dq(i, j, k, 1) += gv(i, j, k, 0) - gv(i, j - 1, k, 0);
  dq(i, j, k, 2) += gv(i, j, k, 1) - gv(i, j - 1, k, 1);
  dq(i, j, k, 3) += gv(i, j, k, 2) - gv(i, j - 1, k, 2);
  dq(i, j, k, 4) += gv(i, j, k, 3) - gv(i, j - 1, k, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += gv(i, j, k, l - 1) - gv(i, j - 1, k, l - 1);
  }
}

__global__ void compute_dHv_dz(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &hv = zone->vis_flux;
  dq(i, j, k, 1) += hv(i, j, k, 0) - hv(i, j, k - 1, 0);
  dq(i, j, k, 2) += hv(i, j, k, 1) - hv(i, j, k - 1, 1);
  dq(i, j, k, 3) += hv(i, j, k, 2) - hv(i, j, k - 1, 2);
  dq(i, j, k, 4) += hv(i, j, k, 3) - hv(i, j, k - 1, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += hv(i, j, k, l - 1) - hv(i, j, k - 1, l - 1);
  }
}

template<int ORDER> __device__ real d_dXi(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nx,
  int phyBoundLeft, int phyBoundRight) {
  real df = 0;
  if constexpr (ORDER == 8) {
    bool computed{false};
    if (phyBoundLeft) {
      if (i == 0) {
        df = -25.0 / 12 * f(i, j, k, l) + 4.0 * f(i + 1, j, k, l) - 3 * f(i + 2, j, k, l) + 4.0 / 3 * f(i + 3, j, k, l)
             - 0.25 * f(i + 4, j, k, l);
        computed = true;
      } else if (i == 1) {
        df = -0.25 * f(i - 1, j, k, l) - 5.0 / 6 * f(i, j, k, l) + 1.5 * f(i + 1, j, k, l) - 0.5 * f(i + 2, j, k, l) +
             1.0 / 12 * f(i + 3, j, k, l);
        computed = true;
      } else if (i == 2) {
        df = -1.0 / 12 * (f(i + 2, j, k, l) - f(i - 2, j, k, l)) + 2.0 / 3 * (f(i + 1, j, k, l) - f(i - 1, j, k, l));
        computed = true;
      } else if (i == 3) {
        df = 0.75 * (f(i + 1, j, k, l) - f(i - 1, j, k, l)) - 0.15 * (f(i + 2, j, k, l) - f(i - 2, j, k, l))
             + 1.0 / 60 * (f(i + 3, j, k, l) - f(i - 3, j, k, l));
        computed = true;
      }
    }
    if (phyBoundRight) {
      if (i == nx - 1) {
        df = 25.0 / 12 * f(i, j, k, l) - 4.0 * f(i - 1, j, k, l) + 3 * f(i - 2, j, k, l) - 4.0 / 3 * f(i - 3, j, k, l) +
             0.25 * f(i - 4, j, k, l);
        computed = true;
      } else if (i == nx - 2) {
        df = 0.25 * f(i + 1, j, k, l) + 5.0 / 6 * f(i, j, k, l) - 1.5 * f(i - 1, j, k, l) + 0.5 * f(i - 2, j, k, l) -
             1.0 / 12 * f(i - 3, j, k, l);
        computed = true;
      } else if (i == nx - 3) {
        df = -1.0 / 12 * (f(i + 2, j, k, l) - f(i - 2, j, k, l)) + 2.0 / 3 * (f(i + 1, j, k, l) - f(i - 1, j, k, l));
        computed = true;
      } else if (i == nx - 4) {
        df = 0.75 * (f(i + 1, j, k, l) - f(i - 1, j, k, l)) - 0.15 * (f(i + 2, j, k, l) - f(i - 2, j, k, l))
             + 1.0 / 60 * (f(i + 3, j, k, l) - f(i - 3, j, k, l));
        computed = true;
      }
    }
    if (!computed) {
      constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
      df = a1 * (f(i + 1, j, k, l) - f(i - 1, j, k, l)) + a2 * (f(i + 2, j, k, l) - f(i - 2, j, k, l))
           + a3 * (f(i + 3, j, k, l) - f(i - 3, j, k, l)) + a4 * (f(i + 4, j, k, l) - f(i - 4, j, k, l));
    }
  }
  return df;
}

template<int ORDER = 8> __device__ real d_dXi_inner(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l) {
  real df = 0;
  if constexpr (ORDER == 8) {
    constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
    df = a1 * (f(i + 1, j, k, l) - f(i - 1, j, k, l)) + a2 * (f(i + 2, j, k, l) - f(i - 2, j, k, l))
         + a3 * (f(i + 3, j, k, l) - f(i - 3, j, k, l)) + a4 * (f(i + 4, j, k, l) - f(i - 4, j, k, l));
  }
  return df;
}

template<int ORDER> __device__ real d_dEta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int ny,
  int phyBoundLeft, int phyBoundRight) {
  real df = 0;
  if constexpr (ORDER == 8) {
    bool computed{false};
    if (phyBoundLeft) {
      if (j == 0) {
        df = -25.0 / 12 * f(i, j, k, l) + 4.0 * f(i, j + 1, k, l) - 3 * f(i, j + 2, k, l) + 4.0 / 3 * f(i, j + 3, k, l)
             - 0.25 * f(i, j + 4, k, l);
        computed = true;
      } else if (j == 1) {
        df = -0.25 * f(i, j - 1, k, l) - 5.0 / 6 * f(i, j, k, l) + 1.5 * f(i, j + 1, k, l) - 0.5 * f(i, j + 2, k, l) +
             1.0 / 12 * f(i, j + 3, k, l);
        computed = true;
      } else if (j == 2) {
        df = -1.0 / 12 * (f(i, j + 2, k, l) - f(i, j - 2, k, l)) + 2.0 / 3 * (f(i, j + 1, k, l) - f(i, j - 1, k, l));
        computed = true;
      } else if (j == 3) {
        df = 0.75 * (f(i, j + 1, k, l) - f(i, j - 1, k, l)) - 0.15 * (f(i, j + 2, k, l) - f(i, j - 2, k, l))
             + 1.0 / 60 * (f(i, j + 3, k, l) - f(i, j - 3, k, l));
        computed = true;
      }
    }
    if (phyBoundRight) {
      if (j == ny - 1) {
        df = 25.0 / 12 * f(i, j, k, l) - 4.0 * f(i, j - 1, k, l) + 3 * f(i, j - 2, k, l) - 4.0 / 3 * f(i, j - 3, k, l) +
             0.25 * f(i, j - 4, k, l);
        computed = true;
      } else if (j == ny - 2) {
        df = 0.25 * f(i, j + 1, k, l) + 5.0 / 6 * f(i, j, k, l) - 1.5 * f(i, j - 1, k, l) + 0.5 * f(i, j - 2, k, l) -
             1.0 / 12 * f(i, j - 3, k, l);
        computed = true;
      } else if (j == ny - 3) {
        df = -1.0 / 12 * (f(i, j + 2, k, l) - f(i, j - 2, k, l)) + 2.0 / 3 * (f(i, j + 1, k, l) - f(i, j - 1, k, l));
        computed = true;
      } else if (j == ny - 4) {
        df = 0.75 * (f(i, j + 1, k, l) - f(i, j - 1, k, l)) - 0.15 * (f(i, j + 2, k, l) - f(i, j - 2, k, l))
             + 1.0 / 60 * (f(i, j + 3, k, l) - f(i, j - 3, k, l));
        computed = true;
      }
    }
    if (!computed) {
      constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
      df = a1 * (f(i, j + 1, k, l) - f(i, j - 1, k, l)) + a2 * (f(i, j + 2, k, l) - f(i, j - 2, k, l))
           + a3 * (f(i, j + 3, k, l) - f(i, j - 3, k, l)) + a4 * (f(i, j + 4, k, l) - f(i, j - 4, k, l));
    }
  }
  return df;
}

template<int ORDER = 8> __device__ real d_dEta_inner(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l) {
  real df = 0;
  if constexpr (ORDER == 8) {
    constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
    df = a1 * (f(i, j + 1, k, l) - f(i, j - 1, k, l)) + a2 * (f(i, j + 2, k, l) - f(i, j - 2, k, l))
         + a3 * (f(i, j + 3, k, l) - f(i, j - 3, k, l)) + a4 * (f(i, j + 4, k, l) - f(i, j - 4, k, l));
  }
  return df;
}

template<int ORDER> __device__ real d_dZeta(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l, int nz,
  int phyBoundLeft, int phyBoundRight) {
  real df = 0;
  if constexpr (ORDER == 8) {
    bool computed{false};
    if (phyBoundLeft) {
      if (k == 0) {
        df = -25.0 / 12 * f(i, j, k, l) + 4.0 * f(i, j, k + 1, l) - 3 * f(i, j, k + 2, l) + 4.0 / 3 * f(i, j, k + 3, l)
             - 0.25 * f(i, j, k + 4, l);
        computed = true;
      } else if (j == 1) {
        df = -0.25 * f(i, j, k - 1, l) - 5.0 / 6 * f(i, j, k, l) + 1.5 * f(i, j, k + 1, l) - 0.5 * f(i, j, k + 2, l) +
             1.0 / 12 * f(i, j, k + 3, l);
        computed = true;
      } else if (j == 2) {
        df = -1.0 / 12 * (f(i, j, k + 2, l) - f(i, j, k - 2, l)) + 2.0 / 3 * (f(i, j, k + 1, l) - f(i, j, k - 1, l));
        computed = true;
      } else if (j == 3) {
        df = 0.75 * (f(i, j, k + 1, l) - f(i, j, k - 1, l)) - 0.15 * (f(i, j, k + 2, l) - f(i, j, k - 2, l))
             + 1.0 / 60 * (f(i, j, k + 3, l) - f(i, j, k - 3, l));
        computed = true;
      }
    }
    if (phyBoundRight) {
      if (k == nz - 1) {
        df = 25.0 / 12 * f(i, j, k, l) - 4.0 * f(i, j, k - 1, l) + 3 * f(i, j, k - 2, l) - 4.0 / 3 * f(i, j, k - 3, l) +
             0.25 * f(i, j, k - 4, l);
        computed = true;
      } else if (k == nz - 2) {
        df = 0.25 * f(i, j, k + 1, l) + 5.0 / 6 * f(i, j, k, l) - 1.5 * f(i, j, k - 1, l) + 0.5 * f(i, j, k - 2, l) -
             1.0 / 12 * f(i, j, k - 3, l);
        computed = true;
      } else if (k == nz - 3) {
        df = -1.0 / 12 * (f(i, j, k + 2, l) - f(i, j, k - 2, l)) + 2.0 / 3 * (f(i, j, k + 1, l) - f(i, j, k - 1, l));
        computed = true;
      } else if (k == nz - 4) {
        df = 0.75 * (f(i, j, k + 1, l) - f(i, j, k - 1, l)) - 0.15 * (f(i, j, k + 2, l) - f(i, j, k - 2, l))
             + 1.0 / 60 * (f(i, j, k + 3, l) - f(i, j, k - 3, l));
        computed = true;
      }
    }
    if (!computed) {
      constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
      df = a1 * (f(i, j, k + 1, l) - f(i, j, k - 1, l)) + a2 * (f(i, j, k + 2, l) - f(i, j, k - 2, l))
           + a3 * (f(i, j, k + 3, l) - f(i, j, k - 3, l)) + a4 * (f(i, j, k + 4, l) - f(i, j, k - 4, l));
    }
  }
  return df;
}

template<int ORDER = 8> __device__ real d_dZeta_inner(const ggxl::VectorField3D<real> &f, int i, int j, int k, int l) {
  real df = 0;
  if constexpr (ORDER == 8) {
    constexpr real a1{0.8}, a2{-0.2}, a3{4.0 / 105}, a4{-1.0 / 280};
    df = a1 * (f(i, j, k + 1, l) - f(i, j, k - 1, l)) + a2 * (f(i, j, k + 2, l) - f(i, j, k - 2, l))
         + a3 * (f(i, j, k + 3, l) - f(i, j, k - 3, l)) + a4 * (f(i, j, k + 4, l) - f(i, j, k - 4, l));
  }
  return df;
}

template<MixtureModel mix_model, int ORDER> __global__ void compute_viscous_flux_collocated(DZone *zone,
  const DParameter *param) {
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const auto mx = zone->mx, my = zone->my, mz = zone->mz;
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  int compute_type[6] = {0, 0, 0, 0, 0, 0};
  if (zone->bType_il(j, k) != 0)
    compute_type[0] = 1;
  if (zone->bType_ir(j, k) != 0)
    compute_type[1] = 1;
  if (zone->bType_jl(i, k) != 0)
    compute_type[2] = 1;
  if (zone->bType_jr(i, k) != 0)
    compute_type[3] = 1;
  if (zone->bType_kl(i, j) != 0)
    compute_type[4] = 1;
  if (zone->bType_kr(i, j) != 0)
    compute_type[5] = 1;

  const auto &pv = zone->bv;
  const real u_xi = d_dXi<ORDER>(pv, i, j, k, 1, mx, compute_type[0], compute_type[1]);
  const real u_eta = d_dEta<ORDER>(pv, i, j, k, 1, my, compute_type[2], compute_type[3]);
  const real u_zeta = d_dZeta<ORDER>(pv, i, j, k, 1, mz, compute_type[4], compute_type[5]);
  const real v_xi = d_dXi<ORDER>(pv, i, j, k, 2, mx, compute_type[0], compute_type[1]);
  const real v_eta = d_dEta<ORDER>(pv, i, j, k, 2, my, compute_type[2], compute_type[3]);
  const real v_zeta = d_dZeta<ORDER>(pv, i, j, k, 2, mz, compute_type[4], compute_type[5]);
  const real w_xi = d_dXi<ORDER>(pv, i, j, k, 3, mx, compute_type[0], compute_type[1]);
  const real w_eta = d_dEta<ORDER>(pv, i, j, k, 3, my, compute_type[2], compute_type[3]);
  const real w_zeta = d_dZeta<ORDER>(pv, i, j, k, 3, mz, compute_type[4], compute_type[5]);
  const real t_xi = d_dXi<ORDER>(pv, i, j, k, 5, mx, compute_type[0], compute_type[1]);
  const real t_eta = d_dEta<ORDER>(pv, i, j, k, 5, my, compute_type[2], compute_type[3]);
  const real t_zeta = d_dZeta<ORDER>(pv, i, j, k, 5, mz, compute_type[4], compute_type[5]);

  // chain rule for derivative
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
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = zone->mul(i, j, k);

  // Compute the viscous stress
  const real tau_xx = mul * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = mul * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = mul * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = mul * (u_y + v_x);
  const real tau_xz = mul * (u_z + w_x);
  const real tau_yz = mul * (v_z + w_y);

  auto &fv = zone->fFlux, &gv = zone->gFlux, &hv = zone->hFlux;
  fv(i, j, k, 1) = xi_x * tau_xx + xi_y * tau_xy + xi_z * tau_xz;
  fv(i, j, k, 2) = xi_x * tau_xy + xi_y * tau_yy + xi_z * tau_yz;
  fv(i, j, k, 3) = xi_x * tau_xz + xi_y * tau_yz + xi_z * tau_zz;

  gv(i, j, k, 1) = eta_x * tau_xx + eta_y * tau_xy + eta_z * tau_xz;
  gv(i, j, k, 2) = eta_x * tau_xy + eta_y * tau_yy + eta_z * tau_yz;
  gv(i, j, k, 3) = eta_x * tau_xz + eta_y * tau_yz + eta_z * tau_zz;

  hv(i, j, k, 1) = zeta_x * tau_xx + zeta_y * tau_xy + zeta_z * tau_xz;
  hv(i, j, k, 2) = zeta_x * tau_xy + zeta_y * tau_yy + zeta_z * tau_yz;
  hv(i, j, k, 3) = zeta_x * tau_xz + zeta_y * tau_yz + zeta_z * tau_zz;

  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity{0};
  if constexpr (mix_model != MixtureModel::Air) {
    conductivity = zone->thermal_conductivity(i, j, k);
  } else {
    constexpr real cp{gamma_air * R_air / (gamma_air - 1)};
    conductivity = mul / param->Pr * cp;
  }
  const real u = pv(i, j, k, 1), v = pv(i, j, k, 2), w = pv(i, j, k, 3);
  const real Ex = u * tau_xx + v * tau_xy + w * tau_xz + conductivity * t_x;
  const real Ey = u * tau_xy + v * tau_yy + w * tau_yz + conductivity * t_y;
  const real Ez = u * tau_xz + v * tau_yz + w * tau_zz + conductivity * t_z;

  fv(i, j, k, 4) = xi_x * Ex + xi_y * Ey + xi_z * Ez;
  gv(i, j, k, 4) = eta_x * Ex + eta_y * Ey + eta_z * Ez;
  hv(i, j, k, 4) = zeta_x * Ex + zeta_y * Ey + zeta_z * Ez;
}

template<int ORDER> __global__ void compute_viscous_flux_collocated_scalar(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const auto mx = zone->mx, my = zone->my, mz = zone->mz;
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;


  int compute_type[6] = {0, 0, 0, 0, 0, 0};
  if (zone->bType_il(j, k) != 0)
    compute_type[0] = 1;
  if (zone->bType_ir(j, k) != 0)
    compute_type[1] = 1;
  if (zone->bType_jl(i, k) != 0)
    compute_type[2] = 1;
  if (zone->bType_jr(i, k) != 0)
    compute_type[3] = 1;
  if (zone->bType_kl(i, j) != 0)
    compute_type[4] = 1;
  if (zone->bType_kr(i, j) != 0)
    compute_type[5] = 1;

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

  // Here, we only consider the influence of species diffusion.
  // That is, if we are solving mixture or finite rate,
  // this part will compute the viscous term of species equations and energy eqn.
  // If we are solving the flamelet model, this part only contributes to the energy eqn.
  const int n_spec{param->n_spec};
  const auto &y = zone->sv;

  real diffusivity[MAX_SPEC_NUMBER];
  real sum_GradXi_cdot_GradY_over_wl{0}, sumGradEta_cdot_GradY_over_wl{0}, sumGradZeta_cdot_GradY_over_wl{0};
  real sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
  real CorrectionVelocityTerm[3]{0, 0, 0};
  real mw_tot{0};
  real driven_force_x[MAX_SPEC_NUMBER * 3];
  real *driven_force_y = &driven_force_x[n_spec];
  real *driven_force_z = &driven_force_y[n_spec];
  for (int l = 0; l < n_spec; ++l) {
    yk[l] = y(i, j, k, l);
    diffusivity[l] = zone->rho_D(i, j, k, l);

    const real y_xi = d_dXi<ORDER>(y, i, j, k, l, mx, compute_type[0], compute_type[1]);
    const real y_eta = d_dEta<ORDER>(y, i, j, k, l, my, compute_type[2], compute_type[3]);
    const real y_zeta = d_dZeta<ORDER>(y, i, j, k, l, mz, compute_type[4], compute_type[5]);

    const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
    const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
    const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
    // Term 1, the gradient of mass fraction.
    const real GradXi_cdot_GradY = xi_x * y_x + xi_y * y_y + xi_z * y_z;
    const real GradEta_cdot_GradY = eta_x * y_x + eta_y * y_y + eta_z * y_z;
    const real GradZeta_cdot_GradY = zeta_x * y_x + zeta_y * y_y + zeta_z * y_z;
    driven_force_x[l] = GradXi_cdot_GradY;
    driven_force_y[l] = GradEta_cdot_GradY;
    driven_force_z[l] = GradZeta_cdot_GradY;
    CorrectionVelocityTerm[0] += diffusivity[l] * GradXi_cdot_GradY;
    CorrectionVelocityTerm[1] += diffusivity[l] * GradEta_cdot_GradY;
    CorrectionVelocityTerm[2] += diffusivity[l] * GradZeta_cdot_GradY;

    // Term 2, the gradient of molecular weights,
    // which is represented by sum of "gradient of mass fractions divided by molecular weight".
    sum_GradXi_cdot_GradY_over_wl += GradXi_cdot_GradY * param->imw[l];
    sumGradEta_cdot_GradY_over_wl += GradEta_cdot_GradY * param->imw[l];
    sumGradZeta_cdot_GradY_over_wl += GradZeta_cdot_GradY * param->imw[l];
    mw_tot += yk[l] * param->imw[l];
    sum_rhoDkYk += diffusivity[l] * yk[l];
  }
  mw_tot = 1.0 / mw_tot;
  CorrectionVelocityTerm[0] -= mw_tot * sum_rhoDkYk * sum_GradXi_cdot_GradY_over_wl;
  CorrectionVelocityTerm[1] -= mw_tot * sum_rhoDkYk * sumGradEta_cdot_GradY_over_wl;
  CorrectionVelocityTerm[2] -= mw_tot * sum_rhoDkYk * sumGradZeta_cdot_GradY_over_wl;

  // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk,
  // which is more significant when the molecular weight is light.
  const auto &pv = zone->bv;
  if (param->gradPInDiffusionFlux) {
    const real p_xi = d_dXi<ORDER>(pv, i, j, k, 4, mx, compute_type[0], compute_type[1]);
    const real p_eta = d_dEta<ORDER>(pv, i, j, k, 4, my, compute_type[2], compute_type[3]);
    const real p_zeta = d_dZeta<ORDER>(pv, i, j, k, 4, mz, compute_type[4], compute_type[5]);

    const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
    const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
    const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

    const real gradXi_cdot_gradP_over_p = (xi_x * p_x + xi_y * p_y + xi_z * p_z) / pv(i, j, k, 4);
    const real gradEta_cdot_gradP_over_p = (eta_x * p_x + eta_y * p_y + eta_z * p_z) / pv(i, j, k, 4);
    const real gradZeta_cdot_gradP_over_p = (zeta_x * p_x + zeta_y * p_y + zeta_z * p_z) / pv(i, j, k, 4);

    // Velocity correction for the 3rd term
    for (int l = 0; l < n_spec; ++l) {
      const real coefficient = (mw_tot * param->imw[l] - 1) * yk[l];
      driven_force_x[l] += coefficient * gradXi_cdot_gradP_over_p;
      driven_force_y[l] += coefficient * gradEta_cdot_gradP_over_p;
      driven_force_z[l] += coefficient * gradZeta_cdot_gradP_over_p;
      CorrectionVelocityTerm[0] += coefficient * gradXi_cdot_gradP_over_p * diffusivity[l];
      CorrectionVelocityTerm[1] += coefficient * gradEta_cdot_gradP_over_p * diffusivity[l];
      CorrectionVelocityTerm[2] += coefficient * gradZeta_cdot_gradP_over_p * diffusivity[l];
    }
  }

  real h[MAX_SPEC_NUMBER];
  const real tm = pv(i, j, k, 5);
  compute_enthalpy(tm, h, param);

  auto &fv = zone->fFlux, &gv = zone->gFlux, &hv = zone->hFlux;
  for (int l = 0; l < n_spec; ++l) {
    real diffusion_flux = diffusivity[l] * (driven_force_x[l] - mw_tot * yk[l] * sum_GradXi_cdot_GradY_over_wl) -
                          yk[l] * CorrectionVelocityTerm[0];
    fv(i, j, k, 5 + l) = diffusion_flux;
    // Add the influence of species diffusion on total energy
    fv(i, j, k, 4) += h[l] * diffusion_flux;

    diffusion_flux = diffusivity[l] * (driven_force_y[l] - mw_tot * yk[l] * sumGradEta_cdot_GradY_over_wl) -
                     yk[l] * CorrectionVelocityTerm[1];
    gv(i, j, k, 5 + l) = diffusion_flux;
    gv(i, j, k, 4) += h[l] * diffusion_flux;

    diffusion_flux = diffusivity[l] * (driven_force_z[l] - mw_tot * yk[l] * sumGradZeta_cdot_GradY_over_wl) -
                     yk[l] * CorrectionVelocityTerm[2];
    hv(i, j, k, 5 + l) = diffusion_flux;
    hv(i, j, k, 4) += h[l] * diffusion_flux;
  }


  if (param->n_ps > 0) {
    const auto &sv = zone->sv;

    for (int l = 0; l < param->n_ps; ++l) {
      const int ls = param->i_ps + l, lc = param->i_ps_cv + l;
      // First, compute the passive scalar gradient
      const real ps_xi = d_dXi<ORDER>(sv, i, j, k, ls, mx, compute_type[0], compute_type[1]);
      const real ps_eta = d_dEta<ORDER>(sv, i, j, k, ls, my, compute_type[2], compute_type[3]);
      const real ps_zeta = d_dZeta<ORDER>(sv, i, j, k, ls, mz, compute_type[4], compute_type[5]);

      const real ps_x = ps_xi * xi_x + ps_eta * eta_x + ps_zeta * zeta_x;
      const real ps_y = ps_xi * xi_y + ps_eta * eta_y + ps_zeta * zeta_y;
      const real ps_z = ps_xi * xi_z + ps_eta * eta_z + ps_zeta * zeta_z;

      const real rhoD{zone->mul(i, j, k) / param->sc_ps[l]};
      fv(i, j, k, lc) = rhoD * (xi_x * ps_x + xi_y * ps_y + xi_z * ps_z);
      gv(i, j, k, lc) = rhoD * (eta_x * ps_x + eta_y * ps_y + eta_z * ps_z);
      hv(i, j, k, lc) = rhoD * (zeta_x * ps_x + zeta_y * ps_y + zeta_z * ps_z);
    }
  }
}

template<int ORDER> __global__ void compute_viscous_flux_derivative(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const auto mx = zone->mx, my = zone->my, mz = zone->mz;
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;


  int compute_type[6] = {0, 0, 0, 0, 0, 0};
  if (zone->bType_il(j, k) != 0)
    compute_type[0] = 1;
  if (zone->bType_ir(j, k) != 0)
    compute_type[1] = 1;
  if (zone->bType_jl(i, k) != 0)
    compute_type[2] = 1;
  if (zone->bType_jr(i, k) != 0)
    compute_type[3] = 1;
  if (zone->bType_kl(i, j) != 0)
    compute_type[4] = 1;
  if (zone->bType_kr(i, j) != 0)
    compute_type[5] = 1;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  const auto &fv = zone->fFlux, &gv = zone->gFlux, hv = zone->hFlux;
  const real jac = zone->jac(i, j, k);

  dq(i, j, k, 1) += (d_dXi<ORDER>(fv, i, j, k, 1, mx, compute_type[0], compute_type[1])
                     + d_dEta<ORDER>(gv, i, j, k, 1, my, compute_type[2], compute_type[3])
                     + d_dZeta<ORDER>(hv, i, j, k, 1, mz, compute_type[4], compute_type[5])) * jac;
  dq(i, j, k, 2) += (d_dXi<ORDER>(fv, i, j, k, 2, mx, compute_type[0], compute_type[1])
                     + d_dEta<ORDER>(gv, i, j, k, 2, my, compute_type[2], compute_type[3])
                     + d_dZeta<ORDER>(hv, i, j, k, 2, mz, compute_type[4], compute_type[5])) * jac;
  dq(i, j, k, 3) += (d_dXi<ORDER>(fv, i, j, k, 3, mx, compute_type[0], compute_type[1])
                     + d_dEta<ORDER>(gv, i, j, k, 3, my, compute_type[2], compute_type[3])
                     + d_dZeta<ORDER>(hv, i, j, k, 3, mz, compute_type[4], compute_type[5])) * jac;
  dq(i, j, k, 4) += (d_dXi<ORDER>(fv, i, j, k, 4, mx, compute_type[0], compute_type[1])
                     + d_dEta<ORDER>(gv, i, j, k, 4, my, compute_type[2], compute_type[3])
                     + d_dZeta<ORDER>(hv, i, j, k, 4, mz, compute_type[4], compute_type[5])) * jac;
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += (d_dXi<ORDER>(fv, i, j, k, l, mx, compute_type[0], compute_type[1])
                       + d_dEta<ORDER>(gv, i, j, k, l, my, compute_type[2], compute_type[3])
                       + d_dZeta<ORDER>(hv, i, j, k, l, mz, compute_type[4], compute_type[5])) * jac;
  }
}

template<MixtureModel mix_model> __global__ void compute_fv_2nd_order(DZone *zone, DParameter *param) {
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

template<MixtureModel mix_model> __global__ void compute_gv_2nd_order(DZone *zone, DParameter *param) {
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

template void compute_viscous_flux<MixtureModel::Air>(const Mesh &mesh, std::vector<Field> &field, DParameter *param,
  const Parameter &parameter);

template void compute_viscous_flux<MixtureModel::Mixture>(const Mesh &mesh, std::vector<Field> &field,
  DParameter *param, const Parameter &parameter);
} // namespace cfd

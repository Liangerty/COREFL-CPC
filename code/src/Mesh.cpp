#include "Mesh.h"
#include "gxl_lib/Math.hpp"
#include "fmt/format.h"
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include "Parameter.h"
#include <sstream>
#include "fmt/os.h"
#include "Mesh.cuh"

cfd::Boundary::Boundary(int x1, int x2, int y1, int y2, int z1, int z2, int type) :
  range_start{x1, y1, z1}, range_end{x2, y2, z2}, type_label{type} {}

void cfd::Boundary::register_boundary(const int dim, const int ngg) {
  for (int i = 0; i < dim; ++i) {
    if (range_start[i] == range_end[i]) {
      face = i;
      if (range_start[i] == 0) direction = -1;
      break;
    }
  }
  for (int i = 0; i < dim; ++i) {
    if (i == face) continue;
    if (range_end[i] < range_start[i]) std::swap(range_end[i], range_start[i]);
    range_start[i] -= ngg;
    range_end[i] += ngg;
  }
  if (dim == 2) {
    range_start[2] = -ngg;
    range_end[2] = ngg;
  }
}

cfd::InnerFace::InnerFace(int x1, int x2, int y1, int y2, int z1, int z2, int tx1, int tx2, int ty1, int ty2, int tz1,
  int tz2, int block_id) :
  range_start{x1, y1, z1}, range_end{x2, y2, z2}, target_start{tx1, ty1, tz1}, target_end{tx2, ty2, tz2},
  target_block{block_id - 1} {}

void cfd::InnerFace::register_boundary(const int ngg, const int dim) {
  for (int i = 0; i < dim; ++i) {
    // if start==end, then it is the normal direction
    if (range_start[i] == range_end[i]) {
      face = i;
      // if the same value is 1, it is the small label face, and the normal
      // points to the negative direction
      if (range_start[i] == 1) direction = -1;
      range_start[i] -= 1;
      range_end[i] -= 1;
      break;
    }
  }
  for (int i = 0; i < dim; ++i) {
    if (target_start[i] == target_end[i]) {
      target_face = i;
      if (target_start[i] == 1) target_direction = -1;
      target_start[i] -= 1;
      target_end[i] -= 1;
      break;
    }
  }
  src_tar[target_face] = -(direction * target_direction) * (face + 1);
  if (dim == 2) {
    const int next_face{1 - face}, target_next_face{1 - target_face};
    range_start[next_face] = std::abs(range_start[next_face]) - 1;
    range_end[next_face] = std::abs(range_end[next_face]) - 1;
    target_start[target_next_face] = std::abs(target_start[target_next_face]) - 1;
    target_end[target_next_face] = std::abs(target_end[target_next_face]) - 1;
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
    src_tar[2] = 3;
    range_start[2] = 0;
    range_end[2] = 0;
    target_start[2] = 0;
    target_end[2] = 0;
  } else {
    int next_face{0}, target_next_face{0};
    for (int i = 0; i < dim; ++i) {
      if (range_start[i] < 0) {
        next_face = i;
        range_start[i] = -range_start[i] - 1;
        range_end[i] = -range_end[i] - 1;
        break;
      }
    }
    for (int i = 0; i < dim; ++i) {
      if (target_start[i] < 0) {
        target_next_face = i;
        target_start[i] = -target_start[i] - 1;
        target_end[i] = -target_end[i] - 1;
        break;
      }
    }
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
    next_face = 3 - face - next_face;
    target_next_face = 3 - target_face - target_next_face;
    range_start[next_face] -= 1;
    range_end[next_face] -= 1;
    target_start[target_next_face] -= 1;
    target_end[target_next_face] -= 1;
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
  }

  // Here, the range of loop for corresponding face is changed to the ghost grid
  // range. The other two directions are also changed, which means the current
  // method also communicates the data in corners. ACANS and Wang's code
  // communicate the corners in default. Later, if we do not want to communicate
  // corner data in default, the ranges for the other 2 directions should be
  // decreased, which can be achieved by just commenting the last loop of
  // this function.
  range_start[face] = direction < 0 ? -ngg : range_end[face];
  range_end[face] = direction < 0 ? 0 : range_end[face] + ngg;
  const int sgn_tar = gxl::sgn(src_tar[target_face]);
  if (target_direction > 0) {
    if (sgn_tar > 0) {
      target_start[target_face] = target_end[target_face] - ngg;
    } else {
      target_start[target_face] = target_end[target_face];
    }
  } else {
    target_start[target_face] = sgn_tar > 0 ? 0 : ngg;
  }
  for (int i = 0; i < 3; ++i) {
    loop_dir[i] = range_start[i] <= range_end[i] ? 1 : -1;
    target_loop_dir[i] = gxl::sgn(src_tar[i]);
    src_tar[i] = std::abs(src_tar[i]) - 1;
  }
  // Include the corners in the transfer
  // Commented on 2023/6/20. The data on corners are not communicated
  /*
   * For corners such as
   *
   *                    |
   *                    |
   *                    |
   *                    |
   *                    |
   * -------------------*
   *                    :
   *                    :
   *                    :
   *                    :
   *
   * which consists of a vertical wall(|),
   *                   a horizontal wall(-)
   *               and a vertical inner boundary(:)
   *             below the corner point(*).
   * We name the left side block 0, which contains the horizontal line and the fluid below,
   * the right side block 1, which contains the vertical line and the fluid on its right.
   *
   * If the communication contains the ghost grids, then for block 1,
   * the communicated region would contain not only the vertical inner boundary,
   * but also 2 points above the corner point.
   * Those 2 points above the corner point is assigned to be the average between the 2 blocks.
   * For this scene, the value of those 2 points is 0 in block 1 because they are on the wall.
   * Those values on block 0 are the negative of the inner 2 points below the horizontal wall
   * as they are ghost grids in block 0. Thus, when we average the 2 values, we get non-zero value on these 2 points,
   * which disobeys the wall boundary condition.
   *
   * */
  //  for (int i = 0; i < 3; ++i) {
  //    if (i != face) {
  //      range_start[i] -= loop_dir[i] * ngg;
  //      range_end[i] += loop_dir[i] * ngg;
  //      target_start[i] -= target_loop_dir[i] * loop_dir[i] * ngg;
  //      target_end[i] += target_loop_dir[i] * loop_dir[i] * ngg;
  //    }
  //  }
  for (int i = 0; i < 3; ++i) {
    n_point[i] = abs(range_start[i] - range_end[i]) + 1;
  }
}

cfd::ParallelFace::ParallelFace(int x1, int x2, int y1, int y2, int z1, int z2, int proc_id, int flag_s, int flag_r):
  range_start{x1, y1, z1}, range_end{x2, y2, z2}, target_process{proc_id}, flag_send{flag_s}, flag_receive{flag_r} {}

void cfd::ParallelFace::register_boundary(const int dim) {
  for (int i = 0; i < dim; ++i) {
    // if start==end, then it is the normal direction
    if (range_start[i] == range_end[i]) {
      face = i;
      // if the same value is 1, it is the small label face, and the normal
      // points to the negative direction
      if (range_start[i] == 1) direction = -1;
      range_start[i] -= 1;
      range_end[i] -= 1;
      loop_order[0] = i;
      loop_dir[i] = 0;
      break;
    }
  }
  if (dim == 2) {
    const int next_face{1 - face};
    range_start[next_face] = std::abs(range_start[next_face]) - 1;
    range_end[next_face] = std::abs(range_end[next_face]) - 1;
    loop_order[1] = next_face;
    loop_dir[1] = range_start[next_face] < range_end[next_face] ? 1 : -1;
    range_start[2] = 0;
    range_end[2] = 0;
    loop_order[2] = 2;
    loop_dir[2] = 0;
  } else {
    int next_face{0};
    for (int i = 0; i < dim; ++i) {
      if (range_start[i] < 0) {
        next_face = i;
        range_start[i] = -range_start[i] - 1;
        range_end[i] = -range_end[i] - 1;
        loop_order[2] = i;
        loop_dir[i] = range_start[i] < range_end[i] ? 1 : -1;
        break;
      }
    }
    next_face = 3 - face - next_face;
    range_start[next_face] -= 1;
    range_end[next_face] -= 1;
    loop_order[1] = next_face;
    loop_dir[next_face] = range_start[next_face] < range_end[next_face] ? 1 : -1;
  }
  // Include the corner into the transfer
  // Commented
  //  for (int i = 0; i < 3; ++i) {
  //    if (i != face) {
  //      range_start[i] -= loop_dir[i] * ngg;
  //      range_end[i] += loop_dir[i] * ngg;
  //    }
  //  }
  for (int i = 0; i < 3; ++i) {
    n_point[i] = abs(range_start[i] - range_end[i]) + 1;
  }
}

cfd::Block::Block(int _mx, int _my, int _mz, int _ngg, int _id, const Parameter &parameter) :
  mx{_mx}, my{_my}, mz{_mz}, n_grid{mx * my * mz}, block_id{_id}, ngg{_ngg}, x{mx, my, mz, _ngg + 1},
  y{mx, my, mz, _ngg + 1}, z{mx, my, mz, _ngg + 1}, jacobian{mx, my, mz, _ngg},
  bType_il(_my, _mz, 0), bType_ir(_my, _mz, 0), bType_jl(_mx, _mz, 0), bType_jr(_mx, _mz, 0), bType_kl(_mx, _my, 0),
  bType_kr(_mx, _my, 0), metric{mx, my, mz, 9, _ngg} {
  if (parameter.get_bool("turbulence") && parameter.get_int("turbulence_method") == 2) {
    des_scale.resize(mx, my, mz, ngg);
  }
}

void cfd::Block::compute_jac_metric(int myid) {
  // First, the inner part is computed, excluding the boundaries and ghost grids
  for (int i = 1; i < mx - 1; ++i) {
    for (int j = 1; j < my - 1; ++j) {
      int k_min{1}, k_max{mz - 1};
      if (mz == 1) {
        // which means dimension = 2
        k_min = 0;
        k_max = 1;
      }
      for (int k = k_min; k < k_max; ++k) {
        const real dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) * 0.5,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) * 0.5,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) * 0.5;
        const real dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) * 0.5,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) * 0.5,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) * 0.5;
        const real dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) * 0.5,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) * 0.5,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) * 0.5;
        const real jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                         dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                         dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          fmt::print("Negative Jacobian from process {}\nBlock {}, index is ({}, {}, {}).\nStop simulation.\n", myid,
                     block_id, i, j, k);
          fmt::print("dxd1=({} - {})/2={}, dyd1=({} - {})/2={}, dzd1=({} - {})/2={}\n",
                     x(i + 1, j, k), x(i - 1, j, k), dxd1,
                     y(i + 1, j, k), y(i - 1, j, k), dyd1,
                     z(i + 1, j, k), z(i - 1, j, k), dzd1);
          fmt::print("dxd2=({} - {})/2={}, dyd2=({} - {})/2={}, dzd2=({} - {})/2={}\n",
                     x(i, j + 1, k), x(i, j - 1, k), dxd2,
                     y(i, j + 1, k), y(i, j - 1, k), dyd2,
                     z(i, j + 1, k), z(i, j - 1, k), dzd2);
          fmt::print("dxd3=({} - {})/2={}, dyd3=({} - {})/2={}, dzd3=({} - {})/2={}\n",
                     x(i, j, k + 1), x(i, j, k - 1), dxd3,
                     y(i, j, k + 1), y(i, j, k - 1), dyd3,
                     z(i, j, k + 1), z(i, j, k - 1), dzd3);
          // for (int zz = 0; zz < mz; ++zz) {
          //   fmt::print("z({},{},{}) = {}\n", i, j, zz, z(i, j, zz));
          // }
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Start to compute boundaries, edges and corners.
  // If a negative value appears here, assign the value of the inner layer to it.
  // First block:
  // k=-1,-ng No edges and corners
  for (int i = 1; i < mx - 1; ++i) {
    for (int j = 1; j < my - 1; ++j) {
      for (int k = 0; k >= -ngg; --k) {
        const double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) * 0.5,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) * 0.5,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) * 0.5;
        const double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) * 0.5,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) * 0.5,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) * 0.5;
        const double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) * 0.5,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) * 0.5,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) * 0.5;
        const double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                           dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                           dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j, k + 1);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i, j, k + 1, l);
          }
//          metric(i, j, k) = metric(i, j, k + 1);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Second block: k=mz,mz+ngg-1, No edges and corners
  for (int i = 1; i < mx - 1; ++i) {
    for (int j = 1; j < my - 1; ++j) {
      for (int k = mz - 1; k < mz + ngg; ++k) {
        const double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        const double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        const double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        const double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                           dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                           dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j, k - 1);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i, j, k - 1, l);
          }
//          metric(i, j, k) = metric(i, j, k - 1);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Third block: j=-1,-ngg, with corners
  for (int i = 1; i < mx - 1; ++i) {
    for (int j = 0; j >= -ngg; --j) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j + 1, k);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i, j + 1, k, l);
          }
//          metric(i, j, k) = metric(i, j + 1, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Fourth block: j=my,my+ngg-1, with edges and corners
  for (int i = 1; i < mx - 1; ++i) {
    for (int j = my - 1; j < my + ngg; ++j) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j - 1, k);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i, j - 1, k, l);
          }
//          metric(i, j, k) = metric(i, j - 1, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Fifth block: i=-1,-ngg, with corners
  for (int i = 0; i >= -ngg; --i) {
    for (int j = -ngg; j < my + ngg; ++j) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i + 1, j, k);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i + 1, j, k, l);
          }
//          metric(i, j, k) = metric(i + 1, j, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Sixth block: i=mx,mx+ngg-1, with corners and edges
  for (int i = mx - 1; i < mx + ngg; ++i) {
    for (int j = -ngg; j < my + ngg; ++j) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i - 1, j, k);
          for (int l = 0; l < 9; ++l) {
            metric(i, j, k, l) = metric(i - 1, j, k, l);
          }
//          metric(i, j, k) = metric(i - 1, j, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        metric(i, j, k, 0) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        metric(i, j, k, 1) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        metric(i, j, k, 2) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        metric(i, j, k, 3) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        metric(i, j, k, 4) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        metric(i, j, k, 5) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        metric(i, j, k, 6) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        metric(i, j, k, 7) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        metric(i, j, k, 8) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
}

// template<int O>
// void cfd::Block::compute_jac_metric_symmetric_conservative(int myid) {
//   gxl::Array3D<real> x1y_y1x, y2x_x2y, y1z_z1y, z2y_y2z;
//   x1y_y1x.resize(mx, my, mz, ngg);
//   y2x_x2y.resize(mx, my, mz, ngg);
//   y1z_z1y.resize(mx, my, mz, ngg);
//   z2y_y2z.resize(mx, my, mz, ngg);
//
//   gxl::Array3D<real> z1x_x1z, x2z_z2x, x3y_y3x, y1x_x1y;
//   z1x_x1z.resize(mx, my, mz, ngg);
//   x2z_z2x.resize(mx, my, mz, ngg);
//   x3y_y3x.resize(mx, my, mz, ngg);
//   y1x_x1y.resize(mx, my, mz, ngg);
//
//
//   gxl::Array3D<real> y3z_z3y, z1y_y1z, z3x_x3z, x1z_z1x;
//   y3z_z3y.resize(mx, my, mz, ngg);
//   z1y_y1z.resize(mx, my, mz, ngg);
//   z3x_x3z.resize(mx, my, mz, ngg);
//   x1z_z1x.resize(mx, my, mz, ngg);
//
//   gxl::Array3D<real> x2y_y2x, y3x_x3y, y2z_z2y, z3y_y3z;
//   x2y_y2x.resize(mx, my, mz, ngg);
//   y3x_x3y.resize(mx, my, mz, ngg);
//   y2z_z2y.resize(mx, my, mz, ngg);
//   z3y_y3z.resize(mx, my, mz, ngg);
//
//   gxl::Array3D<real> z2x_x2z, x3z_z3x;
//   z2x_x2z.resize(mx, my, mz, ngg);
//   x3z_z3x.resize(mx, my, mz, ngg);
//
//   for (int k = -ngg; k < mz + ngg; ++k) {
//     for (int j = -ngg; j < my + ngg; ++j) {
//       for (int i = -ngg; i < mx + ngg; ++i) {
//         const real xh = x(i, j, k), yh = y(i, j, k), zh = z(i, j, k);
//         const real x1 = Derivatives<O>(x, i, j, k, 1), y1 = Derivatives<O>(y, i, j, k, 1),
//             z1 = Derivatives<O>(z, i, j, k, 1);
//         const real x2 = Derivatives<O>(x, i, j, k, 2), y2 = Derivatives<O>(y, i, j, k, 2),
//             z2 = Derivatives<O>(z, i, j, k, 2);
//         const real x3 = Derivatives<O>(x, i, j, k, 3), y3 = Derivatives<O>(y, i, j, k, 3),
//             z3 = Derivatives<O>(z, i, j, k, 3);
//
//         x1y_y1x(i, j, k) = x1 * yh - y1 * xh;
//         y2x_x2y(i, j, k) = y2 * xh - x2 * yh;
//         y1z_z1y(i, j, k) = y1 * zh - z1 * yh;
//         z2y_y2z(i, j, k) = z2 * yh - y2 * zh;
//
//         z1x_x1z(i, j, k) = z1 * xh - x1 * zh;
//         x2z_z2x(i, j, k) = x2 * zh - z2 * xh;
//         x3y_y3x(i, j, k) = x3 * yh - y3 * xh;
//         y1x_x1y(i, j, k) = y1 * xh - x1 * yh;
//
//         y3z_z3y(i, j, k) = y3 * zh - z3 * yh;
//         z1y_y1z(i, j, k) = z1 * yh - y1 * zh;
//         z3x_x3z(i, j, k) = z3 * xh - x3 * zh;
//         x1z_z1x(i, j, k) = x1 * zh - z1 * xh;
//
//         x2y_y2x(i, j, k) = x2 * yh - y2 * xh;
//         y3x_x3y(i, j, k) = y3 * xh - x3 * yh;
//         y2z_z2y(i, j, k) = y2 * zh - z2 * yh;
//         z3y_y3z(i, j, k) = z3 * yh - y3 * zh;
//
//         z2x_x2z(i, j, k) = z2 * xh - x2 * zh;
//         x3z_z3x(i, j, k) = x3 * zh - z3 * xh;
//       }
//     }
//   }
//   gxl::Array3D<real> D3, D2, D1;
//   D3.resize(mx, my, mz, ngg);
//   D2.resize(mx, my, mz, ngg);
//   D1.resize(mx, my, mz, ngg);
//   for (int k = -ngg; k < mz + ngg; ++k) {
//     for (int j = -ngg; j < my + ngg; ++j) {
//       for (int i = -ngg; i < mx + ngg; ++i) {
//         D3(i, j, k) = (Derivatives<O>(x1y_y1x, i, j, k, 2) + Derivatives<O>(y2x_x2y, i, j, k, 1)) * z(i, j, k)
//                       + (Derivatives<O>(y1z_z1y, i, j, k, 2) + Derivatives<O>(z2y_y2z, i, j, k, 1)) * x(i, j, k)
//                       + (Derivatives<O>(z1x_x1z, i, j, k, 2) + Derivatives<O>(x2z_z2x, i, j, k, 1)) * y(i, j, k);
//
//         D2(i, j, k) = (Derivatives<O>(x3y_y3x, i, j, k, 1) + Derivatives<O>(y1x_x1y, i, j, k, 3)) * z(i, j, k)
//                       + (Derivatives<O>(y3z_z3y, i, j, k, 1) + Derivatives<O>(z1y_y1z, i, j, k, 3)) * x(i, j, k)
//                       + (Derivatives<O>(z3x_x3z, i, j, k, 1) + Derivatives<O>(x1z_z1x, i, j, k, 3)) * y(i, j, k);
//
//         D1(i, j, k) = (Derivatives<O>(x2y_y2x, i, j, k, 3) + Derivatives<O>(y3x_x3y, i, j, k, 2)) * z(i, j, k)
//                       + (Derivatives<O>(y2z_z2y, i, j, k, 3) + Derivatives<O>(z3y_y3z, i, j, k, 2)) * x(i, j, k)
//                       + (Derivatives<O>(z2x_x2z, i, j, k, 3) + Derivatives<O>(x3z_z3x, i, j, k, 2)) * y(i, j, k);
//       }
//     }
//   }
//
//   for (int k = -ngg; k < mz + ngg; ++k) {
//     for (int j = -ngg; j < my + ngg; ++j) {
//       for (int i = -ngg; i < mx + ngg; ++i) {
//         const real jac = (Derivatives<O>(D3, i, j, k, 3) + Derivatives<O>(D2, i, j, k, 2) +
//                           Derivatives<O>(D1, i, j, k, 1)) / 6.0;
//         if (jac <= 0) {
//           fmt::print("Negative Jacobian from process {}\nBlock {}, index is ({}, {}, {}).\n"
//                      "Stop simulation.\n", myid, block_id, i, j, k);
//           MPI_Abort(MPI_COMM_WORLD, 1);
//         }
//         jacobian(i, j, k) = jac;
//         const real jI = 0.5 / jac;
//         auto &m = metric(i, j, k);
//         m(1, 1) = (Derivatives<O>(y2z_z2y, i, j, k, 3) - Derivatives<O>(y3z_z3y, i, j, k, 2)) * jI;
//         m(1, 2) = (Derivatives<O>(z2x_x2z, i, j, k, 3) - Derivatives<O>(z3x_x3z, i, j, k, 2)) * jI;
//         m(1, 3) = (Derivatives<O>(x2y_y2x, i, j, k, 3) - Derivatives<O>(x3y_y3x, i, j, k, 2)) * jI;
//         m(2, 1) = (Derivatives<O>(y3z_z3y, i, j, k, 1) - Derivatives<O>(y1z_z1y, i, j, k, 3)) * jI;
//         m(2, 2) = (Derivatives<O>(z3x_x3z, i, j, k, 1) - Derivatives<O>(z1x_x1z, i, j, k, 3)) * jI;
//         m(2, 3) = (Derivatives<O>(x3y_y3x, i, j, k, 1) - Derivatives<O>(x1y_y1x, i, j, k, 3)) * jI;
//         m(3, 1) = (Derivatives<O>(y1z_z1y, i, j, k, 2) - Derivatives<O>(y2z_z2y, i, j, k, 1)) * jI;
//         m(3, 2) = (Derivatives<O>(z1x_x1z, i, j, k, 2) - Derivatives<O>(z2x_x2z, i, j, k, 1)) * jI;
//         m(3, 3) = (Derivatives<O>(x1y_y1x, i, j, k, 2) - Derivatives<O>(x2y_y2x, i, j, k, 1)) * jI;
//       }
//     }
//   }
// }

void cfd::Block::log_negative_jacobian(const int myid, const int i, const int j, const int k) const {
  const std::filesystem::path out_dir("error-log/negative-jacobian");
  if (!exists(out_dir)) create_directories(out_dir);
  const auto path1 = out_dir.string();
  const auto name = fmt::format("{}/Process_{}-Block_{}.log", path1, myid, block_id);
  auto err_log = fmt::output_file(name, fmt::file::WRONLY | fmt::file::CREATE | fmt::file::APPEND);
  err_log.print("Block {}, I = {}\tJ = {}\tK = {}\n", block_id, i, j, k);
  err_log.close();
}

void cfd::Block::trim_abundant_ghost_mesh() {
  gxl::Array3D<real> xx(x), yy(y), zz(z);
  x.resize(mx, my, mz, ngg);
  y.resize(mx, my, mz, ngg);
  z.resize(mx, my, mz, ngg);
  for (int i = -ngg; i < mx + ngg; i++) {
    for (int j = -ngg; j < my + ngg; ++j) {
      for (int k = -ngg; k < mz + ngg; k++) {
        x(i, j, k) = xx(i, j, k);
        y(i, j, k) = yy(i, j, k);
        z(i, j, k) = zz(i, j, k);
      }
    }
  }
}

void cfd::Block::compute_des_scale(const Parameter &parameter) {
  const auto dim = parameter.get_int("dimension");
  const auto gridDeltaMethod = parameter.get_int("des_scale_method");

  for (int k = -ngg; k < mz + ngg; ++k) {
    for (int j = -ngg; j < my + ngg; ++j) {
      for (int i = -ngg; i < mx + ngg; ++i) {
        // Compute the left and right half-point coordinate
        real left[3], right[3];
        left[0] = 0.5 * (x(i - 1, j, k) + x(i, j, k));
        left[1] = 0.5 * (y(i - 1, j, k) + y(i, j, k));
        left[2] = 0.5 * (z(i - 1, j, k) + z(i, j, k));
        right[0] = 0.5 * (x(i, j, k) + x(i + 1, j, k));
        right[1] = 0.5 * (y(i, j, k) + y(i + 1, j, k));
        right[2] = 0.5 * (z(i, j, k) + z(i + 1, j, k));
        // compute the length of i direction
        real hx = sqrt((right[0] - left[0]) * (right[0] - left[0]) +
                       (right[1] - left[1]) * (right[1] - left[1]) +
                       (right[2] - left[2]) * (right[2] - left[2]));

        // Do the same for the j direction
        left[0] = 0.5 * (x(i, j - 1, k) + x(i, j, k));
        left[1] = 0.5 * (y(i, j - 1, k) + y(i, j, k));
        left[2] = 0.5 * (z(i, j - 1, k) + z(i, j, k));
        right[0] = 0.5 * (x(i, j, k) + x(i, j + 1, k));
        right[1] = 0.5 * (y(i, j, k) + y(i, j + 1, k));
        right[2] = 0.5 * (z(i, j, k) + z(i, j + 1, k));
        real hy = sqrt((right[0] - left[0]) * (right[0] - left[0]) +
                       (right[1] - left[1]) * (right[1] - left[1]) +
                       (right[2] - left[2]) * (right[2] - left[2]));

        real h_max{std::max(hx, hy)};
        real cubic_root_vol{sqrt(hx * hy)};
        if (dim == 3) {
          // Do the same for the k direction
          left[0] = 0.5 * (x(i, j, k - 1) + x(i, j, k));
          left[1] = 0.5 * (y(i, j, k - 1) + y(i, j, k));
          left[2] = 0.5 * (z(i, j, k - 1) + z(i, j, k));
          right[0] = 0.5 * (x(i, j, k) + x(i, j, k + 1));
          right[1] = 0.5 * (y(i, j, k) + y(i, j, k + 1));
          right[2] = 0.5 * (z(i, j, k) + z(i, j, k + 1));
          real hz = sqrt((right[0] - left[0]) * (right[0] - left[0]) +
                         (right[1] - left[1]) * (right[1] - left[1]) +
                         (right[2] - left[2]) * (right[2] - left[2]));

          h_max = std::max(h_max, hz);
          cubic_root_vol = cbrt(hx * hy * hz);
        }

        switch (gridDeltaMethod) {
          case 1: // Use the max of the 3 dimensions as the grid scale.
            des_scale(i, j, k) = h_max;
            break;
          case 0:
          default: // Use the cubic root of the cell volume as the grid scale.
            des_scale(i, j, k) = cubic_root_vol;
            break;
        }
      }
    }
  }
}

cfd::Mesh::Mesh(Parameter &parameter) :
  n_proc{parameter.get_int("n_proc")}, ngg{parameter.get_int("ngg")}, nblk{new int[n_proc]} {
  const int myid = parameter.get_int("myid");
  const bool parallel = parameter.get_bool("parallel");
  if (myid == 0)
    fmt::print("\n{:*^80}\n", "Grid Information");
  //First read the grid points into memory
  read_grid(myid, parameter);
  parameter.update_parameter("n_block", n_block);
  parameter.update_parameter("dimension", dimension);
  if (dimension == 2) {
    parameter.update_parameter("perform_spanwise_average", false);
  }
  if (myid == 0) {
    fmt::print("\t->-> {}-D problem\n", dimension);
    fmt::print("\t->-> {:<20} : total grid number\n", n_grid_total);
  }

  const bool need_extend_boundary = parameter.get_int("viscous_order") == 2;

  const auto bcs = parameter.get_string_array("boundary_conditions");
  int periodic_label{-100};
  for (auto &n: bcs) {
    const auto& st = parameter.get_struct(n);
    if (auto type = std::get<std::string>(st.at("type")); type == "periodic")
      periodic_label = std::get<int>(st.at("label"));
  }
  parameter.update_parameter("periodic_label", periodic_label);

  read_boundary(myid, need_extend_boundary, periodic_label);

  read_inner_interface(myid/*, ngg*/);

  if (parallel) {
    read_parallel_interface(myid/*, ngg*/);
  }

  /*Scale the grid to physical units*/
  if (const auto grid_scale = parameter.get_real("gridScale"); std::abs(grid_scale - 1.0) > 1e-15) {
    scale(grid_scale);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0) {
      fmt::print("\t->-> {:<20} : grid scale(meter).\n", grid_scale);
    }
  }

  /*Initialize the coordinates of ghost grids*/
  init_ghost_grid(myid, parallel/*, ngg*/);

  /*Compute the metrics and jacobian values of all grid points*/
  for (auto &b: block)
    compute_jac_metric(myid, b, parameter);
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) fmt::print("\tFinish computing metrics and jacobian values.\n");

  if (parameter.get_bool("turbulence") && parameter.get_int("turbulence_method") == 2) {
    // When we use DES simulation, we need to compute the grid scale \Delta.
    for (auto &b: block) b.compute_des_scale(parameter);
    if (parameter.get_int("myid") == 0) {
      fmt::print("\tFinish getting DES scales.\n");
    }
  }
  for (auto &b: block) b.trim_abundant_ghost_mesh();
}

cfd::Block &cfd::Mesh::operator[](const size_t i) {
  return block[i];
}

const cfd::Block &cfd::Mesh::operator[](const size_t i) const {
  return block[i];
}

void cfd::Mesh::read_grid(const int myid, const Parameter &parameter) {
  if (parameter.get_int("mesh_file_type") == 1) {
    // This is a binary plot3d file(not treated by readGrid preprocessor), we need to read it and get its info on our own.
    // Not implemented.
    auto file_name = "./input/" + parameter.get_string("mesh_prefix") + ".dat";
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_close(&file);
  } else {
    // mesh_file_type == 0: the default method, which reads the file from readGrid preprocessor.
    if (parameter.get_int("gridIsBinary")) {
      // Binary grid grd with MPI_File
      MPI_File grd;
      MPI_File_open(MPI_COMM_SELF, fmt::format("./input/grid/grid{:>4}.dat", myid).c_str(), MPI_MODE_RDONLY,
                    MPI_INFO_NULL, &grd);
      MPI_File_read(grd, &n_block, 1, MPI_INT, MPI_STATUS_IGNORE); //	Read number of grid blocks
      for (int i = 0; i < n_block; ++i) {
        //	Read grid number in three directions of each block
        int mx, my, mz;
        MPI_File_read(grd, &mx, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_read(grd, &my, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_read(grd, &mz, 1, MPI_INT, MPI_STATUS_IGNORE);
        block.emplace_back(mx, my, mz, ngg, i, parameter);
      }
      if (block[0].mz == 1) {
        dimension = 2;
      }
      block.shrink_to_fit();
      n_grid = 0;
      for (auto &i: block) {
        n_grid += i.n_grid;
      } //compute the total grid number in this process
      for (int blk = 0; blk < n_block; ++blk) {
        //read grid coordinates
        MPI_Datatype ty;
        int lSize[3] = {block[blk].mx + 2 * (ngg + 1), block[blk].my + 2 * (ngg + 1), block[blk].mz + 2 * (ngg + 1)};
        int sSize[3] = {block[blk].mx, block[blk].my, block[blk].mz};
        int start_idx[3] = {ngg + 1, ngg + 1, ngg + 1};
        MPI_Type_create_subarray(3, lSize, sSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
        MPI_Type_commit(&ty);
        MPI_File_read(grd, block[blk].x.data(), 1, ty, MPI_STATUS_IGNORE);
        MPI_File_read(grd, block[blk].y.data(), 1, ty, MPI_STATUS_IGNORE);
        MPI_File_read(grd, block[blk].z.data(), 1, ty, MPI_STATUS_IGNORE);
        MPI_Type_free(&ty);
      }
      MPI_File_close(&grd);
    } else {
      // ASCII grid file
      std::ifstream grd(fmt::format("./input/grid/grid{:>4}.grd", myid), std::ios::in);
      std::string input;
      std::getline(grd, input);
      n_block = std::stoi(input); //	Read number of grid blocks
      for (int i = 0; i < n_block; ++i) {
        //	Read grid number in three directions of each block
        std::getline(grd, input);
        std::istringstream line(input);
        int mx = 0, my = 0, mz = 0;
        line >> mx >> my >> mz;
        block.emplace_back(mx, my, mz, ngg, i, parameter);
      }
      if (block[0].mz == 1) {
        dimension = 2;
      }
      block.shrink_to_fit();
      n_grid = 0;
      for (auto &i: block) {
        n_grid += i.n_grid;
      } //compute the total grid number in this process
      for (int blk = 0; blk < n_block; ++blk) {
        //read grid coordinates
        for (int k = 0; k < block[blk].mz; ++k) {
          for (int j = 0; j < block[blk].my; ++j) {
            for (int i = 0; i < block[blk].mx; ++i) {
              grd >> block[blk].x(i, j, k);
            }
          }
        }
        for (int k = 0; k < block[blk].mz; ++k) {
          for (int j = 0; j < block[blk].my; ++j) {
            for (int i = 0; i < block[blk].mx; ++i) {
              grd >> block[blk].y(i, j, k);
            }
          }
        }
        for (int k = 0; k < block[blk].mz; ++k) {
          for (int j = 0; j < block[blk].my; ++j) {
            for (int i = 0; i < block[blk].mx; ++i) {
              grd >> block[blk].z(i, j, k);
            }
          }
        }
      }
      grd.close();
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //Sum grid number in all processes to the root process
  MPI_Reduce(&n_grid, &n_grid_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_grid_total, 1, MPI_INT, 0, MPI_COMM_WORLD); //Broadcast total grid number to all processes

  // Specify how many blocks there are on each process
  MPI_Allgather(&n_block, 1, MPI_INT, nblk, 1, MPI_INT, MPI_COMM_WORLD);
  auto *disp = new int[n_proc];
  disp[0] = 0;
  n_block_total = nblk[0];
  for (int i = 1; i < n_proc; ++i) {
    disp[i] = disp[i - 1] + nblk[i - 1];
    n_block_total += nblk[i];
  }

  mx_blk = new int[n_block_total];
  my_blk = new int[n_block_total];
  mz_blk = new int[n_block_total];
  auto *m_this = new int[n_block * 3];
  int *mx_this = m_this, *my_this = &mx_this[n_block], *mz_this = &my_this[n_block];
  for (int i = 0; i < n_block; ++i) {
    mx_this[i] = block[i].mx;
    my_this[i] = block[i].my;
    mz_this[i] = block[i].mz;
  }
  MPI_Allgatherv(mx_this, n_block, MPI_INT, mx_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgatherv(my_this, n_block, MPI_INT, my_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgatherv(mz_this, n_block, MPI_INT, mz_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  delete[]m_this;
  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0) {
    fmt::print("\t[[{}]] blocks in all, the block number in each process is:\n", n_block_total);
    // Print the number of blocks in each process, with 5 processes in each line.
    printf("\t");
    for (int i = 0; i < n_proc; ++i) {
      fmt::print("\t{:>4}", nblk[i]);
      if ((i + 1) % 5 == 0 && i + 1 != n_proc) fmt::print("\n\t");
    }
    fmt::print("\n\tThe dimension of the blocks are:\n");
    for (int i = 0; i < n_block_total; ++i) {
      fmt::print("\t\tBlock {}:, mx = {}, my = {}, mz = {}\n", i, mx_blk[i], my_blk[i], mz_blk[i]);
    }
  }
}

void cfd::Mesh::read_boundary(int myid, bool need_extend_boundary, int periodic_label) {
  std::ifstream grd(fmt::format("./input/boundary_condition/boundary{:>4}.txt", myid), std::ios::in);
  std::string input{};
  for (int blk = 0; blk < n_block; ++blk) {
    std::getline(grd, input);
    const int bsz = std::stoi(input);
    int i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
    for (int ii = 0; ii < bsz; ++ii) {
      std::getline(grd, input);
      std::istringstream line(input);
      line >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7;
      block[blk].boundary.emplace_back(i1, i2, i3, i4, i5, i6, i7);
      if (need_extend_boundary)
        block[blk].boundary[ii].register_boundary(dimension, ngg);
      else
        block[blk].boundary[ii].register_boundary(dimension);

      // assign the type to bTypes
      if (i7 != periodic_label) {
        if (i1 == i2) {
          // The i=const face
          if (i1 == 0) {
            // il face
            auto &f = block[blk].bType_il;
            for (int j = i3; j <= i4; ++j) {
              for (int k = i5; k <= i6; ++k) {
                f(j, k) = i7;
              }
            }
          } else if (i1 != 0) {
            // ir face
            auto &f = block[blk].bType_ir;
            for (int j = i3; j <= i4; ++j) {
              for (int k = i5; k <= i6; ++k) {
                f(j, k) = i7;
              }
            }
          }
        } else if (i3 == i4) {
          // The j=const face
          if (i3 == 0) {
            // jl face
            auto &f = block[blk].bType_jl;
            for (int i = i1; i <= i2; ++i) {
              for (int k = i5; k <= i6; ++k) {
                f(i, k) = i7;
              }
            }
          } else if (i3 != 0) {
            // jr face
            auto &f = block[blk].bType_jr;
            for (int i = i1; i <= i2; ++i) {
              for (int k = i5; k <= i6; ++k) {
                f(i, k) = i7;
              }
            }
          }
        } else if (i5 == i6) {
          // The k=const face
          if (i5 == 0) {
            auto &f = block[blk].bType_kl;
            for (int i = i1; i <= i2; ++i) {
              for (int j = i3; j <= i4; ++j) {
                f(i, j) = i7;
              }
            }
          } else if (i5 != 0) {
            auto &f = block[blk].bType_kr;
            for (int i = i1; i <= i2; ++i) {
              for (int j = i3; j <= i4; ++j) {
                f(i, j) = i7;
              }
            }
          }
        }
      }
    }
    block[blk].boundary.shrink_to_fit();
  }
  grd.close();
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    fmt::print("\tFinish reading boundary conditions.\n");
  }
}

void cfd::Mesh::read_inner_interface(const int myid/*, int ngg*/) {
  std::ifstream grd(fmt::format("./input/boundary_condition/inner{:>4}.txt", myid), std::ios::in);
  std::string input{};
  // Read how many inner faces there are in each block and store them in @n_face.
  std::getline(grd, input);
  auto n_face = new int[n_block];
  for (int i = 0; i < n_block; ++i) {
    grd >> n_face[i];
  }
  std::getline(grd, input);
  std::getline(grd, input);
  for (int blk = 0; blk < n_block; ++blk) {
    int i1{0}, i2{0}, i3{0}, i4{0}, i5{0}, i6{0}, i7{0};
    int j1{0}, j2{0}, j3{0}, j4{0}, j5{0}, j6{0}, j7{0};
    for (int l = 0; l < n_face[blk]; ++l) {
      std::getline(grd, input);
      std::istringstream line(input);
      line >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7;
      std::getline(grd, input);
      line.clear();
      line.str(input);
      line >> j1 >> j2 >> j3 >> j4 >> j5 >> j6 >> j7;
      block[blk].inner_face.emplace_back(i1, i2, i3, i4, i5, i6,
                                         j1, j2, j3, j4, j5, j6, j7);
      block[blk].inner_face[l].register_boundary(ngg, dimension);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    fmt::print("\tFinish reading inner communication faces.\n");
  }
  grd.close();
  delete[]n_face;
}

void cfd::Mesh::read_parallel_interface(const int myid/*, int ngg*/) {
  std::ifstream grd(fmt::format("./input/boundary_condition/parallel{:>4}.txt", myid), std::ios::in);
  std::string input{};
  int blk_num1{0}, tot_face{0};
  //Read block number of the process first to see if it matches the grid file information.
  grd >> blk_num1 >> tot_face;
  if (blk_num1 != n_block) {
    fmt::print("Read parallel error! In process {}, ", myid);
    fmt::print("n_block in grid file is {}, in parallel file is {}\n", n_block, blk_num1);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Store the number of parallel faces that need to communicate among processes in @n_face.
  auto *n_face = new int[n_block];
  for (int i = 0; i < n_block; ++i) {
    grd >> n_face[i];
  } //Read the number of parallel communication faces of each block
  std::getline(grd, input);
  std::getline(grd, input);
  for (int blk = 0; blk < n_block; ++blk) {
    int i1{0}, i2{0}, i3{0}, i4{0}, i5{0}, i6{0}, i7{0}, flag_s{0}, flag_r{0};
    for (int f = 0; f < n_face[blk]; ++f) {
      grd >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> flag_s >> flag_r;
      block[blk].parallel_face.emplace_back(i1, i2, i3, i4, i5, i6,
                                            i7, flag_s, flag_r);
      block[blk].parallel_face[f].register_boundary(dimension);
    }
  }
  delete[]n_face;
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) fmt::print("\tFinish reading parallel communication faces.\n");
  grd.close();
}

void cfd::Mesh::scale(const real scale) {
  for (int b = 0; b < n_block; ++b) {
    for (auto &gg: block[b].x) {
      gg *= scale;
    }
    for (auto &gg: block[b].y) {
      gg *= scale;
    }
    for (auto &gg: block[b].z) {
      gg *= scale;
    }
  }
}

void cfd::Mesh::init_ghost_grid(const int myid, const bool parallel/*, const int ngg*/) {
  //Add ghost grid except corners
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto mx = B.mx, my = B.my, mz = B.mz;
    //add ghost grid in i direction
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = -1; i >= -ngg - 1; --i) {
          B.x(i, j, k) = 2 * B.x(0, j, k) - B.x(-i, j, k);
          B.y(i, j, k) = 2 * B.y(0, j, k) - B.y(-i, j, k);
          B.z(i, j, k) = 2 * B.z(0, j, k) - B.z(-i, j, k);
        }
        for (int i = mx; i < mx + ngg + 1; ++i) {
          B.x(i, j, k) = 2 * B.x(mx - 1, j, k) - B.x(2 * (mx - 1) - i, j, k);
          B.y(i, j, k) = 2 * B.y(mx - 1, j, k) - B.y(2 * (mx - 1) - i, j, k);
          B.z(i, j, k) = 2 * B.z(mx - 1, j, k) - B.z(2 * (mx - 1) - i, j, k);
        }
      }
    }
    //add ghost grid in j direction
    for (int k = 0; k < mz; ++k) {
      for (int i = 0; i < mx; ++i) {
        for (int j = -1; j >= -ngg - 1; --j) {
          B.x(i, j, k) = 2 * B.x(i, 0, k) - B.x(i, -j, k);
          B.y(i, j, k) = 2 * B.y(i, 0, k) - B.y(i, -j, k);
          B.z(i, j, k) = 2 * B.z(i, 0, k) - B.z(i, -j, k);
        }
        for (int j = my; j < my + ngg + 1; ++j) {
          B.x(i, j, k) = 2 * B.x(i, my - 1, k) - B.x(i, 2 * (my - 1) - j, k);
          B.y(i, j, k) = 2 * B.y(i, my - 1, k) - B.y(i, 2 * (my - 1) - j, k);
          B.z(i, j, k) = 2 * B.z(i, my - 1, k) - B.z(i, 2 * (my - 1) - j, k);
        }
      }
    }
    //add ghost grid in k direction
    if (dimension == 3) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int k = -1; k >= -ngg - 1; --k) {
            B.x(i, j, k) = 2 * B.x(i, j, 0) - B.x(i, j, -k);
            B.y(i, j, k) = 2 * B.y(i, j, 0) - B.y(i, j, -k);
            B.z(i, j, k) = 2 * B.z(i, j, 0) - B.z(i, j, -k);
          }
          for (int k = mz; k < mz + ngg + 1; ++k) {
            B.x(i, j, k) = 2 * B.x(i, j, mz - 1) - B.x(i, j, 2 * (mz - 1) - k);
            B.y(i, j, k) = 2 * B.y(i, j, mz - 1) - B.y(i, j, 2 * (mz - 1) - k);
            B.z(i, j, k) = 2 * B.z(i, j, mz - 1) - B.z(i, j, 2 * (mz - 1) - k);
          }
        }
      }
    } else {
      // dimension = 2
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int k = -1; k >= -ngg - 1; --k) {
            B.x(i, j, k) = B.x(i, j, 0);
            B.y(i, j, k) = B.y(i, j, 0);
            B.z(i, j, k) = B.z(i, j, 0) + k;
          }
          for (int k = mz; k < mz + ngg + 1; ++k) {
            B.x(i, j, k) = B.x(i, j, mz - 1);
            B.y(i, j, k) = B.y(i, j, mz - 1);
            B.z(i, j, k) = B.z(i, j, mz - 1) + k - mz + 1;
          }
        }
      }
    }
  }
  //Assign the ghost grid of inner boundary by neighbor grid
  init_inner_ghost_grid();
  //If we use parallel computation, assign the ghost grid of parallel boundaries by MPI
  if (parallel) {
    init_parallel_ghost_grid();
  }
  /*Assign ghost grids on the edges and at the corners by using parallelogram hypothesis*/
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto mx = B.mx, my = B.my, mz = B.mz;
    //The whole ghost grid range in z direction are assigned, which is  [-ngg-1,mx+ngg]*[-ngg-1,my+ngg]*[0,mz-1]
    for (int i = 1; i <= ngg + 1; ++i) {
      for (int j = 1; j <= ngg + 1; ++j) {
        for (int k = 0; k < mz; ++k) {
          const int i1{-i}, j1{-j};
          const int i2{mx - 1 + i}, j2{my - 1 + j};
          B.x(i1, j1, k) = B.x(i1, 0, k) + B.x(0, j1, k) - B.x(0, 0, k);
          B.y(i1, j1, k) = B.y(i1, 0, k) + B.y(0, j1, k) - B.y(0, 0, k);
          B.z(i1, j1, k) = B.z(i1, 0, k) + B.z(0, j1, k) - B.z(0, 0, k);

          B.x(i2, j2, k) = B.x(i2, my - 1, k) + B.x(mx - 1, j2, k) - B.x(mx - 1, my - 1, k);
          B.y(i2, j2, k) = B.y(i2, my - 1, k) + B.y(mx - 1, j2, k) - B.y(mx - 1, my - 1, k);
          B.z(i2, j2, k) = B.z(i2, my - 1, k) + B.z(mx - 1, j2, k) - B.z(mx - 1, my - 1, k);

          B.x(i1, j2, k) = B.x(i1, my - 1, k) + B.x(0, j2, k) - B.x(0, my - 1, k);
          B.y(i1, j2, k) = B.y(i1, my - 1, k) + B.y(0, j2, k) - B.y(0, my - 1, k);
          B.z(i1, j2, k) = B.z(i1, my - 1, k) + B.z(0, j2, k) - B.z(0, my - 1, k);

          B.x(i2, j1, k) = B.x(i2, 0, k) + B.x(mx - 1, j1, k) - B.x(mx - 1, 0, k);
          B.y(i2, j1, k) = B.y(i2, 0, k) + B.y(mx - 1, j1, k) - B.y(mx - 1, 0, k);
          B.z(i2, j1, k) = B.z(i2, 0, k) + B.z(mx - 1, j1, k) - B.z(mx - 1, 0, k);
        }
      }
    }
    //And in y direction. [-ngg-1,mx+ngg]*[0,my-1]*[-ngg-1,my+ngg]
    for (int i = 1; i <= ngg + 1; ++i) {
      for (int j = 0; j < my; ++j) {
        for (int k = 1; k <= ngg + 1; ++k) {
          const int i1{-i}, k1{-k};
          const int i2{mx - 1 + i}, k2{mz - 1 + k};
          B.x(i1, j, k1) = B.x(i1, j, 0) + B.x(0, j, k1) - B.x(0, j, 0);
          B.y(i1, j, k1) = B.y(i1, j, 0) + B.y(0, j, k1) - B.y(0, j, 0);
          B.z(i1, j, k1) = B.z(i1, j, 0) + B.z(0, j, k1) - B.z(0, j, 0);

          B.x(i2, j, k2) = B.x(i2, j, mz - 1) + B.x(mx - 1, j, k2) - B.x(mx - 1, j, mz - 1);
          B.y(i2, j, k2) = B.y(i2, j, mz - 1) + B.y(mx - 1, j, k2) - B.y(mx - 1, j, mz - 1);
          B.z(i2, j, k2) = B.z(i2, j, mz - 1) + B.z(mx - 1, j, k2) - B.z(mx - 1, j, mz - 1);

          B.x(i1, j, k2) = B.x(i1, j, mz - 1) + B.x(0, j, k2) - B.x(0, j, mz - 1);
          B.y(i1, j, k2) = B.y(i1, j, mz - 1) + B.y(0, j, k2) - B.y(0, j, mz - 1);
          B.z(i1, j, k2) = B.z(i1, j, mz - 1) + B.z(0, j, k2) - B.z(0, j, mz - 1);

          B.x(i2, j, k1) = B.x(i2, j, 0) + B.x(mx - 1, j, k1) - B.x(mx - 1, j, 0);
          B.y(i2, j, k1) = B.y(i2, j, 0) + B.y(mx - 1, j, k1) - B.y(mx - 1, j, 0);
          B.z(i2, j, k1) = B.z(i2, j, 0) + B.z(mx - 1, j, k1) - B.z(mx - 1, j, 0);
        }
      }
    }
    //The rest are grids in x direction and at the corners, which is the range defined by
    //[-ngg-1,mx+ngg]*([-ngg-1,-1]|[my,my+ngg])*([-ngg-1,-1]|[mz,mz+ngg])
    for (int i = -ngg - 1; i <= mx + ngg; ++i) {
      for (int j = 1; j <= ngg + 1; ++j) {
        for (int k = 1; k <= ngg + 1; ++k) {
          const int j1{-j}, k1{-k};
          const int j2{my - 1 + j}, k2{mz - 1 + k};
          B.x(i, j1, k1) = B.x(i, j1, 0) + B.x(i, 0, k1) - B.x(i, 0, 0);
          B.x(i, j2, k2) = B.x(i, j2, mz - 1) + B.x(i, my - 1, k2) - B.x(i, my - 1, mz - 1);
          B.x(i, j1, k2) = B.x(i, j1, mz - 1) + B.x(i, 0, k2) - B.x(i, 0, mz - 1);
          B.x(i, j2, k1) = B.x(i, j2, 0) + B.x(i, my - 1, k1) - B.x(i, my - 1, 0);
          B.y(i, j1, k1) = B.y(i, j1, 0) + B.y(i, 0, k1) - B.y(i, 0, 0);
          B.y(i, j2, k2) = B.y(i, j2, mz - 1) + B.y(i, my - 1, k2) - B.y(i, my - 1, mz - 1);
          B.y(i, j1, k2) = B.y(i, j1, mz - 1) + B.y(i, 0, k2) - B.y(i, 0, mz - 1);
          B.y(i, j2, k1) = B.y(i, j2, 0) + B.y(i, my - 1, k1) - B.y(i, my - 1, 0);
          B.z(i, j1, k1) = B.z(i, j1, 0) + B.z(i, 0, k1) - B.z(i, 0, 0);
          B.z(i, j2, k2) = B.z(i, j2, mz - 1) + B.z(i, my - 1, k2) - B.z(i, my - 1, mz - 1);
          B.z(i, j1, k2) = B.z(i, j1, mz - 1) + B.z(i, 0, k2) - B.z(i, 0, mz - 1);
          B.z(i, j2, k1) = B.z(i, j2, 0) + B.z(i, my - 1, k1) - B.z(i, my - 1, 0);
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    fmt::print("\tFinish creating ghost grids.\n");
  }
}

void cfd::Mesh::init_inner_ghost_grid() {
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto in_face = static_cast<int>(B.inner_face.size());
    for (int f = 0; f < in_face; ++f) {
      const auto &fc = B.inner_face[f];
      auto &nB = block[fc.target_block]; //n for neighbour
      int di[]{0, 0, 0};
      int min_[]{fc.range_start[0], fc.range_start[1], fc.range_start[2]},
          max_[]{fc.range_end[0], fc.range_end[1], fc.range_end[2]},
          mint_[]{fc.target_start[0], fc.target_start[1], fc.target_start[2]};
      // The following condition is only for this ghost grid assignment, otherwise no need of it
      if (fc.direction < 0) {
        min_[fc.face] -= 1;
        max_[fc.face] -= 1;
        if (fc.target_direction > 0) {
          mint_[fc.target_face] -= 1;
        } else {
          mint_[fc.target_face] += 1;
        }
      } else {
        max_[fc.face] += 1;
        min_[fc.face] += 1;
        if (fc.target_direction > 0) {
          mint_[fc.target_face] -= 1;
        } else {
          mint_[fc.target_face] += 1;
        }
      }
      const auto stx = fc.target_loop_dir[0], sty = fc.target_loop_dir[1], stz = fc.target_loop_dir[2];
      const auto di1 = fc.loop_dir[0], dj1 = fc.loop_dir[1], dk1 = fc.loop_dir[2];
      for (int i1 = min_[0]; di1 * (i1 - max_[0]) != 1; i1 += di1) {
        di[0] = i1 - min_[0];
        for (int j1 = min_[1]; dj1 * (j1 - max_[1]) != 1; j1 += dj1) {
          di[1] = j1 - min_[1];
          for (int k1 = min_[2]; dk1 * (k1 - max_[2]) != 1; k1 += dk1) {
            di[2] = k1 - min_[2];
            const int i2 = mint_[0] + stx * di[fc.src_tar[0]];
            const int j2 = mint_[1] + sty * di[fc.src_tar[1]];
            const int k2 = mint_[2] + stz * di[fc.src_tar[2]];
            B.x(i1, j1, k1) = nB.x(i2, j2, k2);
            B.y(i1, j1, k1) = nB.y(i2, j2, k2);
            B.z(i1, j1, k1) = nB.z(i2, j2, k2);
          }
        }
      }
    }
  }
}

void cfd::Mesh::init_parallel_ghost_grid() {
  //Add up to the total face number
  size_t total_face = 0;
  for (int m = 0; m < n_block; ++m) {
    total_face += block[m].parallel_face.size();
  }

  //A 2-D array which is the cache used when using MPI to send/recv messages.
  //The first dimension is the face index,
  //while the second dimension is the coordinate of that face, 3 consecutive numbers represent one position.
  const auto temp_s = new real *[total_face], temp_r = new real *[total_face];

  //Added with iterating through faces and will equal to the total face number when the loop ends
  int fc_num = 0;
  //Compute the array size of different faces and allocate them. Different for different faces.
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto fc = static_cast<int>(B.parallel_face.size());
    for (int f = 0; f < fc; ++f) {
      const auto &face = B.parallel_face[f];
      //The length of the array is 3*(ngg+1)*${number of grid points of the face}
      //ngg+1 is the number of layers to communicate, 3 for 3 coordinates(x,y,z)
      int extent[3]{
        std::abs(face.range_start[0] - face.range_end[0]) + 1,
        std::abs(face.range_start[1] - face.range_end[1]) + 1,
        std::abs(face.range_start[2] - face.range_end[2]) + 1
      };
      for (int i = 0; i < 3; ++i) {
        if (i == face.face) continue;
        extent[i] += 2 * ngg + 2;
      }
      const int len = 3 * (ngg + 1) * extent[0] * extent[1] * extent[2];
      temp_s[fc_num] = new real[len];
      temp_r[fc_num] = new real[len];
      ++fc_num;
    }
  }

  //Create array for MPI_ISEND/IRecv
  //MPI_REQUEST is an array representing whether the face sends/recvs successfully
  const auto s_request = new MPI_Request[total_face], r_request = new MPI_Request[total_face];
  const auto s_status = new MPI_Status[total_face], r_status = new MPI_Status[total_face];
  fc_num = 0;

  for (int m = 0; m < n_block; ++m) {
    auto &B = block[m];
    const auto f_num = static_cast<int>(B.parallel_face.size());
    for (int f = 0; f < f_num; ++f) {
      //Iterate through the faces
      const auto &Fc = B.parallel_face[f];
      int num = 0;
      int min_[]{Fc.range_start[0], Fc.range_start[1], Fc.range_start[2]},
          max_[]{Fc.range_end[0], Fc.range_end[1], Fc.range_end[2]};
      min_[0] = Fc.range_start[Fc.loop_order[0]] - Fc.direction;
      max_[0] = Fc.range_end[Fc.loop_order[0]] - Fc.direction * (ngg + 1);
      min_[1] = Fc.range_start[Fc.loop_order[1]] - Fc.loop_dir[Fc.loop_order[1]] * (ngg + 1);
      max_[1] = Fc.range_end[Fc.loop_order[1]] + Fc.loop_dir[Fc.loop_order[1]] * (ngg + 1);
      min_[2] = Fc.range_start[Fc.loop_order[2]] - Fc.loop_dir[Fc.loop_order[2]] * (ngg + 1);
      max_[2] = Fc.range_end[Fc.loop_order[2]] + Fc.loop_dir[Fc.loop_order[2]] * (ngg + 1);
      const int di1 = -Fc.direction, dj1 = Fc.loop_dir[Fc.loop_order[1]];
      int dk1 = Fc.loop_dir[Fc.loop_order[2]];
      if (dimension == 2) {
        min_[2] = -3;
        max_[2] = 3;
        dk1 = 1;
      }
      int send_ijk[]{0, 0, 0};
      for (send_ijk[Fc.loop_order[0]] = min_[0]; di1 * (send_ijk[Fc.loop_order[0]] - max_[0]) != 1;
           send_ijk[Fc.loop_order[0]] += di1) {
        for (send_ijk[Fc.loop_order[1]] = min_[1]; dj1 * (send_ijk[Fc.loop_order[1]] - max_[1]) != 1;
             send_ijk[Fc.loop_order[1]] += dj1) {
          for (send_ijk[Fc.loop_order[2]] = min_[2]; dk1 * (send_ijk[Fc.loop_order[2]] - max_[2]) != 1;
               send_ijk[Fc.loop_order[2]] += dk1) {
            const int i1{send_ijk[0]}, j1{send_ijk[1]}, k1{send_ijk[2]};
            temp_s[fc_num][num] = B.x(i1, j1, k1);
            temp_s[fc_num][num + 1] = B.y(i1, j1, k1);
            temp_s[fc_num][num + 2] = B.z(i1, j1, k1);
            num += 3;
          }
        }
      }
      //Send and receive. Take care of the first address!
      MPI_Isend(&temp_s[fc_num][0], num, MPI_DOUBLE, Fc.target_process, Fc.flag_send, MPI_COMM_WORLD,
                &s_request[fc_num]);
      MPI_Irecv(&temp_r[fc_num][0], num, MPI_DOUBLE, Fc.target_process, Fc.flag_receive, MPI_COMM_WORLD,
                &r_request[fc_num]);
      ++fc_num;
    }
  }

  //Wait for all faces finishing communication
  MPI_Waitall(static_cast<int>(total_face), s_request, s_status);
  MPI_Waitall(static_cast<int>(total_face), r_request, r_status);
  MPI_Barrier(MPI_COMM_WORLD);

  //Assign the correct value got by MPI receive
  fc_num = 0;
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const size_t f_num = B.parallel_face.size();
    for (size_t f = 0; f < f_num; ++f) {
      const auto &fc = B.parallel_face[f];
      int min_[]{0, 0, 0}, max_[]{0, 0, 0};
      min_[0] = fc.range_start[fc.loop_order[0]] + fc.direction;
      max_[0] = fc.range_end[fc.loop_order[0]] + fc.direction * (ngg + 1);
      min_[1] = fc.range_start[fc.loop_order[1]] - fc.loop_dir[fc.loop_order[1]] * (ngg + 1);
      max_[1] = fc.range_end[fc.loop_order[1]] + fc.loop_dir[fc.loop_order[1]] * (ngg + 1);
      min_[2] = fc.range_start[fc.loop_order[2]] - fc.loop_dir[fc.loop_order[2]] * (ngg + 1);
      max_[2] = fc.range_end[fc.loop_order[2]] + fc.loop_dir[fc.loop_order[2]] * (ngg + 1);
      const int di1 = fc.direction, dj1 = fc.loop_dir[fc.loop_order[1]];
      int dk1 = fc.loop_dir[fc.loop_order[2]];
      int n{0};
      if (dimension == 2) {
        min_[2] = -3;
        max_[2] = 3;
        dk1 = 1;
      }
      int recv_ijk[]{0, 0, 0};
      // Because x/y/z are stored independently, these values should be read continuously.
      for (recv_ijk[fc.loop_order[0]] = min_[0]; di1 * (recv_ijk[fc.loop_order[0]] - max_[0]) != 1;
           recv_ijk[fc.loop_order[0]] += di1) {
        for (recv_ijk[fc.loop_order[1]] = min_[1]; dj1 * (recv_ijk[fc.loop_order[1]] - max_[1]) != 1;
             recv_ijk[fc.loop_order[1]] += dj1) {
          for (recv_ijk[fc.loop_order[2]] = min_[2]; dk1 * (recv_ijk[fc.loop_order[2]] - max_[2]) != 1;
               recv_ijk[fc.loop_order[2]] += dk1) {
            const int i1{recv_ijk[0]}, j1{recv_ijk[1]}, k1{recv_ijk[2]};
            B.x(i1, j1, k1) = temp_r[fc_num][n];
            B.y(i1, j1, k1) = temp_r[fc_num][n + 1];
            B.z(i1, j1, k1) = temp_r[fc_num][n + 2];
            n += 3;
          }
        }
      }
      fc_num++;
    }
  }

  //Free dynamic allocated memory
  delete[]s_status;
  delete[]r_status;
  delete[]s_request;
  delete[]r_request;
  for (size_t i = 0; i < total_face; ++i) {
    delete[]temp_s[i];
    delete[]temp_r[i];
  }
  delete[]temp_s;
  delete[]temp_r;
}

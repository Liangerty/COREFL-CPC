#pragma once

#include "Define.h"
#include "gxl_lib/Array.hpp"
#include "gxl_lib/Matrix.hpp"
#include <vector>

namespace cfd {
struct Boundary {
  Boundary() = default;

  Boundary(int x1, int x2, int y1, int y2, int z1, int z2, int type);

  /**
   * \brief process the boundary information to identify its face and direction.
   * \param dim dimension of the problem
   * \param ngg = 0 number of ghost grids in each direction
   */
  void register_boundary(int dim, int ngg = 0);

  /** the coordinate range of 3 directions*/
  int range_start[3] = {0, 0, 0};
  int range_end[3] = {0, 0, 0};
  /** type identifier for the boundary*/
  int type_label = 0;
  /**> the normal direction of the face, i-0, j-1, k-2*/
  int face = 0;
  /**> is the face normal positive or negative?
   *Default: +1, which means the
   * normal points to the direction that the coordinate increases*/
  int direction = 1;
};

struct InnerFace {
  InnerFace(int x1, int x2, int y1, int y2, int z1, int z2, int tx1, int tx2, int ty1, int ty2, int tz1, int tz2,
    int block_id);

  /**
   * \brief establish the corresponding relation of the faces
   * \param ngg number of ghost grids in each direction
   * \param dim dimension
   */
  void register_boundary(int ngg, int dim);

  int range_start[3]{0, 0, 0};
  int range_end[3]{0, 0, 0}; // the coordinate range of 3 directions
  int face = 0;              // The normal direction of the face, i-0, j-1, k-2
  // is the face normal positive or negative. Default: +1, which means the
  // normal points to the direction that the coordinate increases
  int direction = 1;
  int target_start[3]{0, 0, 0},
      target_end[3]{0, 0, 0};      // The coordinate range of target block
  int target_block = 0;            // The target block number.
  int target_face = 0;             // the normal direction of the target face, i-0, j-1, k-2
  int target_direction = 1;        // Is the target face normal positive or negative? Default: +1
  int src_tar[3]{0, 0, 0};         // the corresponding relation between this source face and the target face
  int loop_dir[3]{1, 1, 1};        // The direction that when looping over the face, i,j,k increment +1/-1
  int target_loop_dir[3]{1, 1, 1}; // The direction that when looping over the face, i,j,k increment +1/-1
  int n_point[3]{0, 0, 0};
};

struct ParallelFace {
  ParallelFace(int x1, int x2, int y1, int y2, int z1, int z2, int proc_id, int flag_s, int flag_r);

  ParallelFace() = default;

  /**
   * \brief establish the value passing order of the face.
   * \param dim dimension
   * \details The faces are stored in a fixed order @loop_order.
   * The first face is the matched face, the second one is the face with positive index,
   * and the last face is the one with negative face.
   */
  void register_boundary(int dim);

  int range_start[3]{0, 0, 0}; // The index of starting point in 3 directions of the current face.
  int range_end[3]{0, 0, 0};   // The index of ending point in 3 directions of the current face.
  int face = 0;                // the normal direction of the face, i-0, j-1, k-2.
  /**
   * \brief The face normal positive or negative.
   * Default: +1, which means the normal points to the direction that the
   * coordinate increases.
   */
  int direction = 1;
  int target_process = 0; // The target block number.
  int flag_send = 0, flag_receive = 0;
  /**
   * \brief When sending a message, the order of data put into the buffer.
   * \details The label with the same number is first iterated, then the positive
   * one, and the negative direction last. When getting data out of the
   * received, also in the same order.
   */
  int loop_order[3]{0, 0, 0};
  /**
   * \brief The direction for iteration of each coordinate, +1/-1
   * \details The order is not directly in i/j/k directions but in the order of
   * @loop_order.
   */
  int loop_dir[3]{1, 1, 1};
  int n_point[3]{0, 0, 0};
};

class Parameter;

/**
 * \brief The class of a grid block.
 * \details A geometric class, all contained messages are about geometric
 * information.
 */
class Block {
public:
  explicit Block(int _mx, int _my, int _mz, int _ngg, int _id, const Parameter &parameter);

  /**
   * \brief compute the jacobian and metric matrix of the current block
   * \param myid process number
   */
  void compute_jac_metric(int myid);

  template<int ORDER> void compute_jac_metric_symmetric_conservative(int myid);

  void trim_abundant_ghost_mesh();

private:
  /**
    * \brief create the header of the error log about negative jacobian values.
    * \param myid current process id
    * \param i the i-th grid point
    * \param j the j-th grid point
    * \param k the k-th grid point
    *
    */
  void log_negative_jacobian(int myid, int i, int j, int k) const;

public:
  int mx = 1, my = 1, mz = 1;
  int n_grid = 1;
  int block_id = 0;
  int ngg = 2;
  gxl::Array3D<real> x, y, z;
  std::vector<Boundary> boundary;
  std::vector<InnerFace> inner_face;
  std::vector<ParallelFace> parallel_face;
  gxl::Array3D<real> jacobian;
  /**
   * \brief array of metrics of the grid points.
   * \details The metric matrix consists of
   *         \f[
   *         \left[\begin{array}{ccc}
   *             \xi_x  &\xi_y  &\xi_z \\
   *             \eta_x &\eta_y &\eta_z \\
   *             \zeta_x&\zeta_y&\zeta_z
   *             \end{array}\right]
   *             \f]
   */
//  gxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric;
  gxl::VectorField3D<real> metric;
  gxl::Array2D<int> bType_il, bType_ir, bType_jl, bType_jr, bType_kl, bType_kr;
  gxl::Array3D<real> des_scale;

  void compute_des_scale(const Parameter &parameter);
};

class Mesh {
public:
  int dimension = 3;
  int n_block = 1;
  int n_grid = 1;
  int n_grid_total = 1;
  int n_block_total = 1;
  int n_proc = 1;
  int ngg = 2;
  int *nblk = nullptr;
  int *mx_blk = nullptr;
  int *my_blk = nullptr;
  int *mz_blk = nullptr;

private:
  std::vector<Block> block;

public:
  explicit Mesh(Parameter &parameter);

  Block &operator[](size_t i);

  const Block &operator[](size_t i) const;

  //  ~Mesh();

private:
  void read_grid(int myid, const Parameter &parameter);

  /**
   * \brief read the physical boundary of the current process
   * \param myid the process id of the current process
   * \param need_extend_boundary
   * \param periodic_label
   */
  void read_boundary(int myid, bool need_extend_boundary, int periodic_label);

  /**
   * \brief read the inner face communication message of the current process
   * \param myid the process id of the current process
   */
  void read_inner_interface(int myid/*, int ngg*/);

  /**
   * \brief read the parallel boundary coordinates. Do not read the target face or match them, left for solver initialization
   * \param myid process number, used for identify which file to read
   */
  void read_parallel_interface(int myid/*, int ngg*/);

  /**
   * \brief scale all coordinates (x/y/z) to unit of meters.
   * \param scale the scale of the coordinates
   * \details for example, if the grid is drawn in unit mm, when we compute it in meters, it should be multiplied by 0.001
   *  first, where the 0.001 is @scale.
   */
  void scale(real scale);

  /**
   * \brief initialize the ghost grids of the simulation
   * \param myid process number, used for identify which file to read
   * \param parallel if the computation is conducted in parallel
   */
  void init_ghost_grid(int myid, bool parallel/*, int ngg*/);

  /**
   * \brief called by @init_ghost_grid, used for initializing ghost grids of the inner faces
   */
  void init_inner_ghost_grid();

  /**
   * \brief called by @init_ghost_grid, initialize the ghost grids of parallel communication faces
   */
  void init_parallel_ghost_grid();
};
}

#include <iostream>
#include <fstream>
#include "Parameter.h"
#include "gxl_lib/Array.hpp"
#include <sstream>
#include <cstdint>
#include <filesystem>

int main() {
  cfd::Parameter parameter("input.txt");

  int GridgenOrPointwise = parameter.get_int("GridgenOrPointwise");

  int dimension = parameter.get_int("dimension");
  std::string gridFileName = parameter.get_string("gridFile");
  std::string boundFileName = parameter.get_string("boundaryFile");
  int n_proc = parameter.get_int("n_proc");
  bool isBinary = parameter.get_bool("isBinary");

  // Read grid file, find the maximum of ni, nj, nk
  int n_block{0};
  std::vector<int> ni, nj, nk;
  std::vector<gxl::Array3D<double>> x, y, z;
  std::vector<int64_t> nGrid;
  std::vector<int> order;

  bool setZ{parameter.get_bool("setZ")};
  real zValue{0};
  if (setZ) {
    if (dimension == 2)
      zValue = parameter.get_real("zValue");
    else
      setZ = false;
  }
  if (isBinary) {
    // Read binary form of grid file
    FILE *gridFile = fopen(gridFileName.c_str(), "rb");
    if (gridFile == nullptr) {
      std::cerr << "Error: Cannot open grid file " << gridFileName << std::endl;
      return 1;
    }

    fread(&n_block, sizeof(int), 1, gridFile);
    if (n_block < n_proc) {
      std::cerr << "Error: The number of blocks is smaller than the number of processes.\n";
      return 1;
    }
    ni.resize(n_block);
    nj.resize(n_block);
    nk.resize(n_block);
    x.resize(n_block);
    y.resize(n_block);
    z.resize(n_block);
    nGrid.resize(n_block, 0);
    order.resize(n_block, -1);

    if (GridgenOrPointwise == 0) {
      // Gridgen
      for (int b = 0; b < n_block; ++b) {
        fread(&ni[b], sizeof(int), 1, gridFile);
        fread(&nj[b], sizeof(int), 1, gridFile);
        fread(&nk[b], sizeof(int), 1, gridFile);
        nGrid[b] = ni[b] * nj[b] * nk[b];
      }
    }

    for (int b = 0; b < n_block; ++b) {
      if (GridgenOrPointwise == 1) {
        // Pointwise
        fread(&ni[b], sizeof(int), 1, gridFile);
        fread(&nj[b], sizeof(int), 1, gridFile);
        fread(&nk[b], sizeof(int), 1, gridFile);
        nGrid[b] = ni[b] * nj[b] * nk[b];
      }

      printf("Block %d to be read: ni = %d, nj = %d, nk = %d, nGrid = %lld\n", b, ni[b], nj[b], nk[b], nGrid[b]);

      for (int i = 0; i < b; ++i) {
        if (nGrid[b] > nGrid[order[i]]) {
          for (int j = b; j > i; --j) {
            order[j] = order[j - 1];
          }
          order[i] = b;
          break;
        }
      }
      if (order[b] == -1) {
        order[b] = b;
      }

      x[b].resize(ni[b], nj[b], nk[b], 0);
      y[b].resize(ni[b], nj[b], nk[b], 0);
      z[b].resize(ni[b], nj[b], nk[b], 0);
      fread(x[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
      fread(y[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
      fread(z[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
      if (setZ) {
        if (abs(zValue) < 1e-10) {
          memset(z[b].data(), 0, ni[b] * nj[b] * nk[b] * sizeof(double));
        } else {
          for (int k = 0; k < nk[b]; ++k) {
            for (int j = 0; j < nj[b]; ++j) {
              for (int i = 0; i < ni[b]; ++i) {
                z[b](i, j, k) = zValue;
              }
            }
          }
        }
      }
    }
  } else {
    std::ifstream gridFile(gridFileName);
    if (!gridFile.is_open()) {
      std::cerr << "Error: Cannot open grid file " << gridFileName << std::endl;
      return 1;
    }

    gridFile >> n_block;
    if (n_block < n_proc) {
      std::cerr << "Error: The number of blocks is smaller than the number of processes.\n";
      return 1;
    }
    ni.resize(n_block);
    nj.resize(n_block);
    nk.resize(n_block);
    x.resize(n_block);
    y.resize(n_block);
    z.resize(n_block);
    nGrid.resize(n_block, 0);
    order.resize(n_block, -1);

    if (GridgenOrPointwise == 0) {
      // Gridgen
      for (int b = 0; b < n_block; ++b) {
        gridFile >> ni[b] >> nj[b] >> nk[b];
        nGrid[b] = ni[b] * nj[b] * nk[b];
      }
    }

    for (int b = 0; b < n_block; ++b) {
      if (GridgenOrPointwise == 1) {
        // Pointwise
        gridFile >> ni[b] >> nj[b] >> nk[b];
        nGrid[b] = ni[b] * nj[b] * nk[b];
      }
      for (int i = 0; i < b; ++i) {
        if (nGrid[b] > nGrid[order[i]]) {
          for (int j = b; j > i; --j) {
            order[j] = order[j - 1];
          }
          order[i] = b;
          break;
        }
      }
      if (order[b] == -1) {
        order[b] = b;
      }

      printf("Block %d to be read: ni = %d, nj = %d, nk = %d, nGrid = %lld\n", b, ni[b], nj[b], nk[b], nGrid[b]);

      x[b].resize(ni[b], nj[b], nk[b], 0);
      y[b].resize(ni[b], nj[b], nk[b], 0);
      z[b].resize(ni[b], nj[b], nk[b], 0);
      for (int k = 0; k < nk[b]; ++k) {
        for (int j = 0; j < nj[b]; ++j) {
          for (int i = 0; i < ni[b]; ++i) {
            gridFile >> x[b](i, j, k);
          }
        }
      }
      for (int k = 0; k < nk[b]; ++k) {
        for (int j = 0; j < nj[b]; ++j) {
          for (int i = 0; i < ni[b]; ++i) {
            gridFile >> y[b](i, j, k);
          }
        }
      }
      for (int k = 0; k < nk[b]; ++k) {
        for (int j = 0; j < nj[b]; ++j) {
          for (int i = 0; i < ni[b]; ++i) {
            gridFile >> z[b](i, j, k);
          }
        }
      }
      if (setZ) {
        if (abs(zValue) < 1e-10) {
          memset(z[b].data(), 0, ni[b] * nj[b] * nk[b] * sizeof(double));
        } else {
          for (int k = 0; k < nk[b]; ++k) {
            for (int j = 0; j < nj[b]; ++j) {
              for (int i = 0; i < ni[b]; ++i) {
                z[b](i, j, k) = zValue;
              }
            }
          }
        }
      }
    }
  }
  printf("Finish reading grid.\n");
  if (setZ) {
    printf("Set z value to %e.\n", zValue);
  }

  // Find the total label of communications
  std::vector<int> n_inter(n_block, 0);
  std::vector<std::vector<int>> inter_id(n_block);

  std::vector<std::vector<std::array<int, 6>>> range_boundary(n_block);
  std::vector<std::vector<int>> boundary_type(n_block);
  std::vector<std::vector<std::array<int, 6>>> range_interior(n_block);
  std::vector<std::vector<std::array<int, 6>>> range_interior_target(n_block);
  std::vector<std::vector<int>> interior_target_id(n_block);
  std::ifstream boundFile(boundFileName);
  int line1OfBFile, line2OfBFile;
  std::string input{};
  boundFile >> line1OfBFile >> line2OfBFile;
  for (int b = 0; b < n_block; ++b) {
    std::getline(boundFile, input);
    n_inter[b] = 0;
    std::getline(boundFile, input);
    std::getline(boundFile, input);
    std::getline(boundFile, input);
    std::istringstream iss(input);
    int n_bound;
    iss >> n_bound;

    for (int f = 0; f < n_bound; ++f) {
      int x1, x2, y1, y2, z1, z2, iType;
      if (dimension == 2) {
        boundFile >> x1 >> x2 >> y1 >> y2 >> iType;
      } else {
        boundFile >> x1 >> x2 >> y1 >> y2 >> z1 >> z2 >> iType;
      }
      if (iType < 0) {
        if (dimension == 2) {
          range_interior[b].push_back({x1, x2, y1, y2, 1, 1});
          boundFile >> x1 >> x2 >> y1 >> y2 >> iType;
          range_interior_target[b].push_back({x1, x2, y1, y2, 1, 1});
        } else {
          range_interior[b].push_back({x1, x2, y1, y2, z1, z2});
          boundFile >> x1 >> x2 >> y1 >> y2 >> z1 >> z2 >> iType;
          range_interior_target[b].push_back({x1, x2, y1, y2, z1, z2});
        }
        interior_target_id[b].push_back(iType - 1);
        n_inter[b] += 1;
        inter_id[b].push_back(iType);
        // Check if this communication label has been recorded, make the info same.
        iType -= 1;
        const int ifThis = n_inter[b] - 1;
        if (iType < b) {
          // The info has been recorded in previous blocks
          // We find the face with exactly the same dimensions
          int found_face = -1;
          for (int tf = 0; tf < n_inter[iType]; ++tf) {
            // The il and ir may be exchanged, but the abs(value) are the same. The same for jl and jr, kl and kr.
            const int il1 = std::abs(range_interior[iType][tf][0]), ir1 = std::abs(range_interior[iType][tf][1]);
            const int jl1 = std::abs(range_interior[iType][tf][2]), jr1 = std::abs(range_interior[iType][tf][3]);
            const int kl1 = std::abs(range_interior[iType][tf][4]), kr1 = std::abs(range_interior[iType][tf][5]);
            const int il2 = std::abs(range_interior_target[b][ifThis][0]);
            const int ir2 = std::abs(range_interior_target[b][ifThis][1]);
            const int jl2 = std::abs(range_interior_target[b][ifThis][2]);
            const int jr2 = std::abs(range_interior_target[b][ifThis][3]);
            const int kl2 = std::abs(range_interior_target[b][ifThis][4]);
            const int kr2 = std::abs(range_interior_target[b][ifThis][5]);
            if ((((il1 == il2) && (ir1 == ir2)) || ((il1 == ir2) && (ir1 == il2))) &&
                (((jl1 == jl2) && (jr1 == jr2)) || ((jl1 == jr2) && (jr1 == jl2)))) {
              if (dimension == 2) {
                found_face = tf;
                break;
              }
              if (((kl1 == kl2) && (kr1 == kr2)) || ((kl1 == kr2) && (kr1 == kl2))) {
                found_face = tf;
                break;
              }
            }
          }
          if (found_face != -1) {
            // Found the face with exactly the same dimensions
            // Make the info same
            range_interior_target[b][ifThis] = range_interior[iType][found_face];
            range_interior[b][ifThis] = range_interior_target[iType][found_face];
          }
        }
      } else {
        if (dimension == 2) {
          range_boundary[b].push_back({x1 - 1, x2 - 1, y1 - 1, y2 - 1, 0, 0});
        } else {
          range_boundary[b].push_back({x1 - 1, x2 - 1, y1 - 1, y2 - 1, z1 - 1, z2 - 1});
        }
        boundary_type[b].push_back(iType);
      }
    }
  }
  boundFile.close();

  // Allocate the grid blocks into processes.

  // n_block_proc: the number of blocks in each process
  std::vector<int> n_block_proc(n_proc, 0);
  // block_grid_num: the number of grids in each process
  std::vector<int64_t> block_grid_num(n_proc, 0);
  // block_id: the original id of blocks in each process
  std::vector<std::vector<int>> block_id(n_proc);
  std::vector<int> block_in_which_proc(n_block);
  std::vector<int> block_is_which_label_in_its_proc(n_block);

  // Put the blocks with the most grid points into the first n_proc processes, according to the order
  for (int p = 0; p < n_proc; ++p) {
    block_id[p].push_back(order[p]);
    n_block_proc[p] = 1;
    block_grid_num[p] = nGrid[order[p]];
    block_in_which_proc[order[p]] = p;
    block_is_which_label_in_its_proc[order[p]] = 0;
  }
  if (n_proc < n_block) {
    for (int b = n_proc; b < n_block; ++b) {
      int p = 0;
      for (int i = 1; i < n_proc; ++i) {
        if (block_grid_num[i] < block_grid_num[p]) {
          p = i;
        }
      }
      block_id[p].push_back(order[b]);
      n_block_proc[p] += 1;
      block_grid_num[p] += nGrid[order[b]];
      block_in_which_proc[order[b]] = p;
      block_is_which_label_in_its_proc[order[b]] = n_block_proc[p] - 1;
    }
  }
  // In each process, the blocks are re-ordered according to its global index.
  // Therefore, the blocks in each process are sorted by its global index in ascending order,
  // while the block_is_which_label_in_its_proc is also re-ordered accordingly.
  for (int p = 0; p < n_proc; ++p) {
    for (int m = 0; m < n_block_proc[p]; ++m) {
      for (int n = m + 1; n < n_block_proc[p]; ++n) {
        if (block_id[p][m] > block_id[p][n]) {
          std::swap(block_id[p][m], block_id[p][n]);
          std::swap(block_is_which_label_in_its_proc[block_id[p][m]], block_is_which_label_in_its_proc[block_id[p][n]]);
        }
      }
    }
  }

  // 生成边界条件文件夹和网格文件夹
  if (!std::filesystem::exists("boundary_condition"))
    std::filesystem::create_directory("boundary_condition");
  if (!std::filesystem::exists("grid"))
    std::filesystem::create_directory("grid");

  // 将所有边界条件写入文件中
  for (int p = 0; p < n_proc; ++p) {
    char *fileName = new char[100];
    sprintf(fileName, "boundary_condition/boundary%4d.txt", p);
    // Output the boundary condition
    FILE *boundaryFile = fopen(fileName, "w");
    for (int m = 0; m < n_block_proc[p]; ++m) {
      const int b = block_id[p][m];
      auto n_boundary = (int) boundary_type[b].size();
      fprintf(boundaryFile, "%7d\n", n_boundary);
      for (int f = 0; f < n_boundary; ++f) {
        fprintf(boundaryFile, "%7d%7d%7d%7d%7d%7d%7d\n", range_boundary[b][f][0], range_boundary[b][f][1],
                range_boundary[b][f][2], range_boundary[b][f][3], range_boundary[b][f][4], range_boundary[b][f][5],
                boundary_type[b][f]);
      }
    }
    fclose(boundaryFile);
  }
  printf("Finishing writing boundary condition files.\n");

  if (!parameter.get_int("writeBinary")) {
    // write the grid in ascii format
    for (int p = 0; p < n_proc; ++p) {
      char *fileName = new char[100];
      sprintf(fileName, "grid/grid%4d.grd", p);
      FILE *gridFile = fopen(fileName, "w");
      printf("Writing grid file %d.\n", p);
      fprintf(gridFile, "%d\n", n_block_proc[p]);
      for (int m = 0; m < n_block_proc[p]; ++m) {
        const int b = block_id[p][m];
        fprintf(gridFile, "%d\t%d\t%d\n", ni[b], nj[b], nk[b]);
      }
      for (int m = 0; m < n_block_proc[p]; ++m) {
        const int b = block_id[p][m];
        for (int k = 0; k < nk[b]; ++k) {
          for (int j = 0; j < nj[b]; ++j) {
            for (int i = 0; i < ni[b]; ++i) {
              fprintf(gridFile, "%e\t", x[b](i, j, k));
            }
          }
        }
        for (int k = 0; k < nk[b]; ++k) {
          for (int j = 0; j < nj[b]; ++j) {
            for (int i = 0; i < ni[b]; ++i) {
              fprintf(gridFile, "%e\t", y[b](i, j, k));
            }
          }
        }
        for (int k = 0; k < nk[b]; ++k) {
          for (int j = 0; j < nj[b]; ++j) {
            for (int i = 0; i < ni[b]; ++i) {
              fprintf(gridFile, "%e\t", z[b](i, j, k));
            }
          }
        }
      }
      fclose(gridFile);
      printf("Finish writing grid file %d.\n", p);
    }
  } else {
    // 将所有网格用C的文件指针以二进制写入文件中
    for (int p = 0; p < n_proc; ++p) {
      char *fileName = new char[100];
      sprintf(fileName, "grid/grid%4d.dat", p);
      FILE *gridFile = fopen(fileName, "wb");
      printf("Writing grid file %d.\n", p);
      fwrite(&n_block_proc[p], sizeof(int), 1, gridFile);
      for (int m = 0; m < n_block_proc[p]; ++m) {
        const int b = block_id[p][m];
        fwrite(&ni[b], sizeof(int), 1, gridFile);
        fwrite(&nj[b], sizeof(int), 1, gridFile);
        fwrite(&nk[b], sizeof(int), 1, gridFile);
      }
      for (int m = 0; m < n_block_proc[p]; ++m) {
        const int b = block_id[p][m];
        fwrite(x[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
        fwrite(y[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
        fwrite(z[b].data(), sizeof(double), ni[b] * nj[b] * nk[b], gridFile);
      }
      fclose(gridFile);
      printf("Finish writing grid file %d.\n", p);
    }
  }

  std::ofstream info_file("grid_allocate_info.txt");
  for (int p = 0; p < n_proc; ++p) {
    info_file << p << " grid file:\n" << "total grid number is " << block_grid_num[p] << "\nblock number is "
        << n_block_proc[p] << "\nblock id is ";
    for (int m = 0; m < n_block_proc[p]; ++m) {
      info_file << block_id[p][m] << ' ';
    }
    info_file << '\n';
  }
  info_file.close();
  // 统计总网格量，检验算法是否重复或漏分了块
  int64_t total_grid = 0;
  for (int p = 0; p < n_proc; ++p) {
    total_grid += block_grid_num[p];
  }
  printf("Total grid number is %lld.\n", total_grid);

  // 检验每个块的id
  std::vector<int> blockIdTest(n_block, 0);
  for (int p = 0; p < n_proc; ++p) {
    for (int b = 0; b < n_block_proc[p]; ++b) {
      blockIdTest[block_id[p][b]] += 1;
    }
  }
  for (int b = 0; b < n_block; ++b) {
    if (blockIdTest[b] != 1) {
      printf("Error: block %d is allocated to %d processes.\n", b, blockIdTest[b]);
    }
  }

  printf("Grid file has been saved successfully!!\n");

  // 进行内外边界整理
  std::vector<int> n_outer(n_block, 0), n_inner(n_block, 0);
  std::vector<std::vector<int>> communicate_label(n_block);
  std::vector<std::vector<int>> outerFace_label(n_block);
  std::vector<std::vector<int>> innerFace_label(n_block);
  int outer_counter{0};
  for (int b = 0; b < n_block; ++b) {
    n_outer[b] = 0;
    n_inner[b] = 0;
    int n_face = n_inter[b];
    for (int f = 0; f < n_face; ++f) {
      if (block_in_which_proc[interior_target_id[b][f]] == block_in_which_proc[b]) {
        innerFace_label[b].push_back(f);
        n_inner[b] += 1;
      } else {
        outerFace_label[b].push_back(f);
        n_outer[b] += 1;
        communicate_label[b].push_back(outer_counter);
        ++outer_counter;
      }
    }
  }

  for (int p = 0; p < n_proc; ++p) {
    int n_send{0};
    for (int b = 0; b < n_block_proc[p]; ++b) {
      n_send += n_outer[block_id[p][b]];
    }

    char *fileName = new char[100];
    sprintf(fileName, "boundary_condition/parallel%4d.txt", p);
    // Output the outer info
    FILE *outerFile = fopen(fileName, "w");
    fprintf(outerFile, "%9d%9d\n", n_block_proc[p], n_send);
    for (int b = 0; b < n_block_proc[p]; ++b) {
      fprintf(outerFile, "%9d", n_outer[block_id[p][b]]);
    }
    fprintf(outerFile, "\niMin  iMax  jMin  jMax  kMin  kMax   s_r_id  send_flag  recv_flag\n");
    for (int b_in_p = 0; b_in_p < n_block_proc[p]; ++b_in_p) {
      int b = block_id[p][b_in_p];
      for (int counter = 0; counter < n_outer[b]; ++counter) {
        int f = outerFace_label[b][counter];
        fprintf(outerFile, "%9d%9d%9d%9d%9d%9d%9d%9d", range_interior[b][f][0], range_interior[b][f][1],
                range_interior[b][f][2], range_interior[b][f][3], range_interior[b][f][4], range_interior[b][f][5],
                block_in_which_proc[interior_target_id[b][f]], communicate_label[b][counter]);
        int target_block = interior_target_id[b][f];
        int target_face = -1;
        for (int i = 0; i < n_outer[target_block]; ++i) {
          int f_tar = outerFace_label[target_block][i];
          if (interior_target_id[target_block][f_tar] == b) {
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][4] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][5]) {
              // completely the same
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][4] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][5]) {
              // reverse the x index
              // swap the x indexes of the target face
              std::swap(range_interior[target_block][f_tar][0], range_interior[target_block][f_tar][1]);
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][4] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][5]) {
              // reverse the y index
              // swap the y indexes of the target face
              std::swap(range_interior[target_block][f_tar][2], range_interior[target_block][f_tar][3]);
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][5] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][4]) {
              // reverse the z index
              // swap the z indexes of the target face
              std::swap(range_interior[target_block][f_tar][4], range_interior[target_block][f_tar][5]);
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][4] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][5]) {
              // reverse the x and y index
              // swap the x and y indexes of the target face
              std::swap(range_interior[target_block][f_tar][0], range_interior[target_block][f_tar][1]);
              std::swap(range_interior[target_block][f_tar][2], range_interior[target_block][f_tar][3]);
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][5] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][4]) {
              // reverse the x and z index
              // swap the x and z indexes of the target face
              std::swap(range_interior[target_block][f_tar][0], range_interior[target_block][f_tar][1]);
              std::swap(range_interior[target_block][f_tar][4], range_interior[target_block][f_tar][5]);
              target_face = i;
              break;
            }
            if (range_interior[target_block][f_tar][0] == range_interior_target[b][f][0] &&
                range_interior[target_block][f_tar][1] == range_interior_target[b][f][1] &&
                range_interior[target_block][f_tar][2] == range_interior_target[b][f][3] &&
                range_interior[target_block][f_tar][3] == range_interior_target[b][f][2] &&
                range_interior[target_block][f_tar][4] == range_interior_target[b][f][5] &&
                range_interior[target_block][f_tar][5] == range_interior_target[b][f][4]) {
              // reverse the y and z index
              // swap the y and z indexes of the target face
              std::swap(range_interior[target_block][f_tar][2], range_interior[target_block][f_tar][3]);
              std::swap(range_interior[target_block][f_tar][4], range_interior[target_block][f_tar][5]);
              target_face = i;
              break;
            }
          }
        }
        if (target_face == -1) {
          printf("Error: Cannot find the target face of block %d, face %d.\n", b, f);
        }
        fprintf(outerFile, "%9d\n", communicate_label[target_block][target_face]);
      }
    }
    fclose(outerFile);
  }

  // inner boundary files
  for (int p = 0; p < n_proc; ++p) {
    int n_send{0};
    for (int b = 0; b < n_block_proc[p]; ++b) {
      n_send += n_inner[block_id[p][b]];
    }

    char *fileName = new char[100];
    sprintf(fileName, "boundary_condition/inner%4d.txt", p);
    // Output the inner info in C style with integer width 7
    FILE *innerFile = fopen(fileName, "w");
    fprintf(innerFile, "%9d\n", n_send);
    for (int b = 0; b < n_block_proc[p]; ++b) {
      fprintf(innerFile, "%9d\t", n_inner[block_id[p][b]]);
    }
    fprintf(innerFile, "\niMin  iMax  jMin  jMax  kMin  kMax   id\n");
    if (n_send > 0) {
      for (int b_in_p = 0; b_in_p < n_block_proc[p]; ++b_in_p) {
        int b = block_id[p][b_in_p];
        for (int i = 0; i < n_inner[b]; ++i) {
          int f = innerFace_label[b][i];
          fprintf(innerFile, "%9d%9d%9d%9d%9d%9d%9d\n", range_interior[b][f][0], range_interior[b][f][1],
                  range_interior[b][f][2], range_interior[b][f][3], range_interior[b][f][4], range_interior[b][f][5],
                  block_is_which_label_in_its_proc[b] + 1);
          fprintf(innerFile, "%9d%9d%9d%9d%9d%9d%9d\n", range_interior_target[b][f][0], range_interior_target[b][f][1],
                  range_interior_target[b][f][2], range_interior_target[b][f][3], range_interior_target[b][f][4],
                  range_interior_target[b][f][5], block_is_which_label_in_its_proc[interior_target_id[b][f]] + 1);
        }
      }
    }
    fclose(innerFile);
  }

  return 0;
}

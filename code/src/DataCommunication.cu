#include "DataCommunication.cuh"

__global__ void cfd::setup_data_to_be_sent(const DZone *zone, int i_face, real *data, const DParameter *param) {
  const auto &f = zone->parFace[i_face];
  int n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  int idx[3];
  for (int ijk: f.loop_order) {
    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
  }

  const int n_var{param->n_scalar + 6}, ngg{zone->ngg};
  int bias = n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);

  const auto &bv = zone->bv;
  #pragma unroll
  for (int l = 0; l < 6; ++l) {
    data[bias + l] = bv(idx[0], idx[1], idx[2], l);
  }
  const auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
  }

  for (int ig = 1; ig <= ngg; ++ig) {
    idx[f.face] -= f.direction;
    bias += n_var;
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
      data[bias + l] = bv(idx[0], idx[1], idx[2], l);
    }
    for (int l = 0; l < param->n_scalar; ++l) {
      data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
    }
  }
}

__global__ void cfd::setup_data_to_be_sent(const DZone *zone, int i_face, real *data, const DParameter *param,
  int task) {
  const auto &f = zone->parFace[i_face];
  int n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  int idx[3];
  for (int ijk: f.loop_order) {
    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
  }

  if (task == 1) {
    const int n_var{param->n_var - 1}, ngg{zone->ngg};
    const int bias_0 = 3 * n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);

    auto bias = bias_0;
    for (int l = 0; l < n_var; ++l) {
      data[bias] = zone->fFlux(idx[0], idx[1], idx[2], l);
      ++bias;
      data[bias] = zone->gFlux(idx[0], idx[1], idx[2], l);
      ++bias;
      data[bias] = zone->hFlux(idx[0], idx[1], idx[2], l);
      ++bias;
    }
    for (int ig = 0; ig < ngg; ++ig) {
      idx[f.face] -= f.direction;
      for (int l = 0; l < n_var; ++l) {
        data[bias] = zone->fFlux(idx[0], idx[1], idx[2], l);
        ++bias;
        data[bias] = zone->gFlux(idx[0], idx[1], idx[2], l);
        ++bias;
        data[bias] = zone->hFlux(idx[0], idx[1], idx[2], l);
        ++bias;
      }
    }
  } else if (task == 2) {
    const int ngg{zone->ngg};
    int bias = (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);
    data[bias] = zone->shock_sensor(idx[0], idx[1], idx[2]);
    ++bias;
    for (int ig = 0; ig < ngg; ++ig) {
      idx[f.face] -= f.direction;
      data[bias] = zone->shock_sensor(idx[0], idx[1], idx[2]);
      ++bias;
    }
  }
}

__global__ void cfd::assign_data_received(DZone *zone, int i_face, const real *data, DParameter *param, int task) {
  const auto &f = zone->parFace[i_face];
  int n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  int idx[3];
  idx[0] = f.range_start[0] + n[0] * f.loop_dir[0];
  idx[1] = f.range_start[1] + n[1] * f.loop_dir[1];
  idx[2] = f.range_start[2] + n[2] * f.loop_dir[2];

  if (task == 1) {
    const int n_var{param->n_var - 1}, ngg{zone->ngg};
    const int bias_0 = 3 * n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);
    int bias = bias_0;
    for (int l = 0; l < n_var; ++l) {
      zone->fFlux(idx[0], idx[1], idx[2], l) = data[bias];
      ++bias;
      zone->gFlux(idx[0], idx[1], idx[2], l) = data[bias];
      ++bias;
      zone->hFlux(idx[0], idx[1], idx[2], l) = data[bias];
      ++bias;
    }
    for (int ig = 0; ig < ngg; ++ig) {
      idx[f.face] += f.direction;
      for (int l = 0; l < n_var; ++l) {
        zone->fFlux(idx[0], idx[1], idx[2], l) = data[bias];
        ++bias;
        zone->gFlux(idx[0], idx[1], idx[2], l) = data[bias];
        ++bias;
        zone->hFlux(idx[0], idx[1], idx[2], l) = data[bias];
        ++bias;
      }
    }
  } else if (task == 2) {
    const int ngg{zone->ngg};
    int bias = (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);
    zone->shock_sensor(idx[0], idx[1], idx[2]) = max(data[bias], zone->shock_sensor(idx[0], idx[1], idx[2]));
    ++bias;
    for (int ig = 0; ig < ngg; ++ig) {
      idx[f.face] += f.direction;
      zone->shock_sensor(idx[0], idx[1], idx[2]) = data[bias];
      ++bias;
    }
  }
}

void cfd::exchange_value(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter,
  DParameter *param, int task) {
  // -1 - inner faces
  for (auto blk = 0; blk < mesh.n_block; ++blk) {
    auto &inF = mesh[blk].inner_face;
    const auto n_innFace = inF.size();
    auto v = field[blk].d_ptr;
    const auto ngg = mesh[blk].ngg;
    for (auto l = 0; l < n_innFace; ++l) {
      // reference to the current face
      const auto &fc = mesh[blk].inner_face[l];
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; ++j) {
        tpb[j] = fc.n_point[j] <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};

      // variables of the neighbor block
      auto nv = field[fc.target_block].d_ptr;
      inner_exchange<<<BPG, TPB>>>(v, nv, l, param, task);
    }
  }

  // Parallel communication via MPI
  if (parameter.get_bool("parallel")) {
    parallel_exchange(mesh, field, parameter, param, task);
  }

  // Periodic conditions
  for (auto blk = 0; blk < mesh.n_block; ++blk) {
    const auto &block = mesh[blk];
    const auto nb = static_cast<int>(block.boundary.size());
    for (int i = 0; i < nb; i++) {
      const auto &hf = block.boundary[i];
      if (hf.type_label == parameter.get_int("periodic_label")) {
        const auto ngg = block.ngg;
        uint tpb[3], bpg[3];
        for (size_t j = 0; j < 3; j++) {
          const auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
          tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
          bpg[j] = (n_point - 1) / tpb[j] + 1;
        }
        dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
        periodic_exchange<<<BPG, TPB>>>(field[blk].d_ptr, param, task, i);
      }
    }
  }
}

__global__ void cfd::inner_exchange(DZone *zone, DZone *tar_zone, int i_face, DParameter *param, int task) {
  const auto &f = zone->innerFace[i_face];
  uint n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  int idx[3], idx_tar[3], d_idx[3];
  for (int i = 0; i < 3; ++i) {
    d_idx[i] = f.loop_dir[i] * static_cast<int>(n[i]);
    idx[i] = f.range_start[i] + d_idx[i];
  }
  for (int i = 0; i < 3; ++i) {
    idx_tar[i] = f.target_start[i] + f.target_loop_dir[i] * d_idx[f.src_tar[i]];
  }

  // The face direction: which of i(0)/j(1)/k(2) is the coincided face.
  const auto face_dir{f.direction > 0 ? f.range_start[f.face] : f.range_end[f.face]};

  if (task == 1) {
    // Exchange the viscous fluxes, used in 8th-order CDS
    if (idx[f.face] == face_dir) {
      for (int l = 0; l < param->n_var - 1; ++l) {
        real ave_v = 0.5 * (zone->fFlux(idx[0], idx[1], idx[2], l) + tar_zone->fFlux(idx_tar[0], idx_tar[1], idx_tar[2], l));
        zone->fFlux(idx[0], idx[1], idx[2], l) = ave_v;
        tar_zone->fFlux(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave_v;
        ave_v = 0.5 * (zone->gFlux(idx[0], idx[1], idx[2], l) + tar_zone->gFlux(idx_tar[0], idx_tar[1], idx_tar[2], l));
        zone->gFlux(idx[0], idx[1], idx[2], l) = ave_v;
        tar_zone->gFlux(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave_v;
        ave_v = 0.5 * (zone->hFlux(idx[0], idx[1], idx[2], l) + tar_zone->hFlux(idx_tar[0], idx_tar[1], idx_tar[2], l));
        zone->hFlux(idx[0], idx[1], idx[2], l) = ave_v;
        tar_zone->hFlux(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave_v;
      }
    } else {
      for (int l = 0; l < param->n_var - 1; ++l) {
        zone->fFlux(idx[0], idx[1], idx[2], l) = tar_zone->fFlux(idx_tar[0], idx_tar[1], idx_tar[2], l);
        zone->gFlux(idx[0], idx[1], idx[2], l) = tar_zone->gFlux(idx_tar[0], idx_tar[1], idx_tar[2], l);
        zone->hFlux(idx[0], idx[1], idx[2], l) = tar_zone->hFlux(idx_tar[0], idx_tar[1], idx_tar[2], l);
      }
    }
  } else if (task == 2) {
    // Exchange the shock sensor
    if (idx[f.face] == face_dir) {
      const real val = max(zone->shock_sensor(idx[0], idx[1], idx[2]),
                           tar_zone->shock_sensor(idx_tar[0], idx_tar[1], idx_tar[2]));
      zone->shock_sensor(idx[0], idx[1], idx[2]) = val;
      tar_zone->shock_sensor(idx_tar[0], idx_tar[1], idx_tar[2]) = val;
    } else {
      zone->shock_sensor(idx[0], idx[1], idx[2]) = tar_zone->shock_sensor(idx_tar[0], idx_tar[1], idx_tar[2]);
    }
  }
}

void cfd::parallel_exchange(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter,
  DParameter *param, int task) {
  const int n_block{mesh.n_block};
  const int ngg{mesh[0].ngg};
  //Add up to the total face number
  size_t total_face = 0;
  for (int m = 0; m < n_block; ++m) {
    total_face += mesh[m].parallel_face.size();
  }

  int n_trans{0};
  switch (task) {
    case 1:
      n_trans = (parameter.get_int("n_var") - 1) * 3; // The three viscous fluxes
      break;
    case 2:
      n_trans = 1;
      break;
    default:
      n_trans = 0;
  }
  //A 2-D array which is the cache used when using MPI to send/recv messages. The first dimension is the face index
  //while the second dimension is the coordinate of that face, 3 consecutive number represents one position.
  const auto temp_s = new real *[total_face], temp_r = new real *[total_face];
  const auto length = new int[total_face];

  //Added with iterating through faces and will equal to the total face number when the loop ends
  int fc_num = 0;
  //Compute the array size of different faces and allocate them. Different for different faces.
  for (int blk = 0; blk < n_block; ++blk) {
    auto &B = mesh[blk];
    const int fc = static_cast<int>(B.parallel_face.size());
    for (int f = 0; f < fc; ++f) {
      const auto &face = B.parallel_face[f];
      //The length of the array is ${number of grid points of the face}*(ngg+1)*n_trans
      //ngg is the number of layers to communicate, n_trans for n_trans variables
      const int len = n_trans * (ngg + 1) * (std::abs(face.range_start[0] - face.range_end[0]) + 1)
                      * (std::abs(face.range_end[1] - face.range_start[1]) + 1)
                      * (std::abs(face.range_end[2] - face.range_start[2]) + 1);
      length[fc_num] = len;
      cudaMalloc(&temp_s[fc_num], len * sizeof(real));
      cudaMalloc(&temp_r[fc_num], len * sizeof(real));
      ++fc_num;
    }
  }

  // Create array for MPI_ISEND/IRecv
  // MPI_REQUEST is an array representing whether the face sends/recvs successfully
  const auto s_request = new MPI_Request[total_face], r_request = new MPI_Request[total_face];
  const auto s_status = new MPI_Status[total_face], r_status = new MPI_Status[total_face];
  fc_num = 0;

  for (int m = 0; m < n_block; ++m) {
    auto &B = mesh[m];
    const int f_num = static_cast<int>(B.parallel_face.size());
    for (int f = 0; f < f_num; ++f) {
      //Iterate through the faces
      const auto &fc = B.parallel_face[f];

      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; ++j) {
        tpb[j] = fc.n_point[j] <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      setup_data_to_be_sent<<<BPG, TPB>>>(field[m].d_ptr, f, &temp_s[fc_num][0], param, task);
      cudaDeviceSynchronize();
      //Send and receive. Take care of the first address!
      // The buffer is on GPU; thus we require a CUDA-aware MPI, such as OpenMPI.
      MPI_Isend(&temp_s[fc_num][0], length[fc_num], MPI_DOUBLE, fc.target_process, fc.flag_send, MPI_COMM_WORLD,
                &s_request[fc_num]);
      MPI_Irecv(&temp_r[fc_num][0], length[fc_num], MPI_DOUBLE, fc.target_process, fc.flag_receive, MPI_COMM_WORLD,
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
    auto &B = mesh[blk];
    const size_t f_num = B.parallel_face.size();
    for (size_t f = 0; f < f_num; ++f) {
      const auto &fc = B.parallel_face[f];
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; ++j) {
        tpb[j] = fc.n_point[j] <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      assign_data_received<<<BPG, TPB>>>(field[blk].d_ptr, f, &temp_r[fc_num][0], param, task);
      cudaDeviceSynchronize();
      fc_num++;
    }
  }

  //Free dynamic allocated memory
  delete[]s_status;
  delete[]r_status;
  delete[]s_request;
  delete[]r_request;
  for (int i = 0; i < fc_num; ++i) {
    cudaFree(&temp_s[i][0]);
    cudaFree(&temp_r[i][0]);
  }
  delete[]temp_s;
  delete[]temp_r;
  delete[]length;
}

__global__ void cfd::periodic_exchange(DZone *zone, DParameter *param, int task, int i_face) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  int idx_other[3]{i, j, k};
  switch (b.face) {
    case 0: // i face
      idx_other[0] = b.direction < 0 ? zone->mx - 1 : 0;
      break;
    case 1: // j face
      idx_other[1] = b.direction < 0 ? zone->my - 1 : 0;
      break;
    case 2: // k face
    default:
      idx_other[2] = b.direction < 0 ? zone->mz - 1 : 0;
      break;
  }

  if (task == 1) {
    auto &fv = zone->fFlux, &gv = zone->gFlux, &hv = zone->hFlux;
    for (int l = 0; l < param->n_var - 1; ++l) {
      real ave = 0.5 * (fv(i, j, k, l) + fv(idx_other[0], idx_other[1], idx_other[2], l));
      fv(i, j, k, l) = ave;
      fv(idx_other[0], idx_other[1], idx_other[2], l) = ave;
      ave = 0.5 * (gv(i, j, k, l) + gv(idx_other[0], idx_other[1], idx_other[2], l));
      gv(i, j, k, l) = ave;
      gv(idx_other[0], idx_other[1], idx_other[2], l) = ave;
      ave = 0.5 * (hv(i, j, k, l) + hv(idx_other[0], idx_other[1], idx_other[2], l));
      hv(i, j, k, l) = ave;
      hv(idx_other[0], idx_other[1], idx_other[2], l) = ave;
    }
    for (int g = 1; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      const int ii{idx_other[0] + g * dir[0]}, ij{idx_other[1] + g * dir[1]}, ik{idx_other[2] + g * dir[2]};
      for (int l = 0; l < param->n_var - 1; ++l) {
        fv(gi, gj, gk, l) = fv(ii, ij, ik, l);
        gv(gi, gj, gk, l) = gv(ii, ij, ik, l);
        hv(gi, gj, gk, l) = hv(ii, ij, ik, l);
      }
    }
  } else if (task == 2) {
    auto &ss = zone->shock_sensor;
    for (int g = 0; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      const int ii{idx_other[0] + g * dir[0]}, ij{idx_other[1] + g * dir[1]}, ik{idx_other[2] + g * dir[2]};
      ss(gi, gj, gk) = ss(ii, ij, ik);
    }
  }
}

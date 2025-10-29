#include "Monitor.cuh"
#include "gxl_lib/MyString.h"
#include <filesystem>
#include "Parallel.h"
#include "Field.h"
#include "Mesh.h"
#include <fstream>

namespace cfd {
Monitor::Monitor(const Parameter &parameter, const Species &species, const Mesh &mesh_) :
  block_monitor{parameter, mesh_}, output_file{parameter.get_int("output_file")},
  n_block{parameter.get_int("n_block")}, n_point(n_block, 0), mesh(mesh_) {
  if (!parameter.get_int("if_monitor_points")) {
    return;
  }

  h_ptr = new DeviceMonitorData;

  // Set up the labels to monitor
  auto var_name_found{setup_labels_to_monitor(parameter, species)};
  const auto myid{parameter.get_int("myid")};
  if (myid == 0) {
    printf("The following variables will be monitored:\n");
    for (const auto &name: var_name_found) {
      printf("%s\t", name.c_str());
    }
    printf("\n");
  }

  // Read the points to be monitored
  auto monitor_file_name{parameter.get_string("monitor_file")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  int counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    int pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    int i, j, k, b;
    line_stream >> b >> i >> j >> k;
    is_h.push_back(i);
    js_h.push_back(j);
    ks_h.push_back(k);
    bs_h.push_back(b);
    printf("Process %d monitors block %d, point (%d, %d, %d).\n", myid, b, i, j, k);
    ++n_point[b];
    ++counter;
  }
  // copy the indices to GPU
  cudaMalloc(&h_ptr->bs_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->is_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->js_d, sizeof(int) * counter);
  cudaMalloc(&h_ptr->ks_d, sizeof(int) * counter);
  cudaMemcpy(h_ptr->bs_d, bs_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->is_d, is_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->js_d, js_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->ks_d, ks_h.data(), sizeof(int) * counter, cudaMemcpyHostToDevice);
  n_point_total = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point_total);
  disp.resize(parameter.get_int("n_block"), 0);
  for (int b = 1; b < n_block; ++b) {
    disp[b] = disp[b - 1] + n_point[b - 1];
  }
  cudaMalloc(&h_ptr->disp, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->disp, disp.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);
  cudaMalloc(&h_ptr->n_point, sizeof(int) * n_block);
  cudaMemcpy(h_ptr->n_point, n_point.data(), sizeof(int) * n_block, cudaMemcpyHostToDevice);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point_total, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point_total);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point_total, nullptr);
  for (int l = 0; l < n_point_total; ++l) {
    std::string file_name{
      "/monitor_" + std::to_string(myid) + '_' + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' +
      std::to_string(js_h[l]) + '_' + std::to_string(ks_h[l]) + ".dat"
    };
    std::filesystem::path whole_name_path{out_dir.string() + file_name};
    if (!exists(whole_name_path)) {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
      fprintf(files[l], "variables=step,");
      for (const auto &name: var_name_found) {
        fprintf(files[l], "%s,", name.c_str());
      }
      fprintf(files[l], "time\n");
    } else {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
    }
  }
}

std::vector<std::string> Monitor::setup_labels_to_monitor(const Parameter &parameter, const Species &species) {
  const auto n_spec{species.n_spec};
  auto &spec_list{species.spec_list};

  const auto var_name{parameter.get_string_array("monitor_var")};

  std::vector<int> bv_idx, sv_idx;
  auto n_found{0};
  std::vector<std::string> var_name_found;
  for (auto name: var_name) {
    name = gxl::to_upper(name);
    if (name == "DENSITY" || name == "RHO") {
      bv_idx.push_back(0);
      var_name_found.emplace_back("Density");
      ++n_found;
    } else if (name == "U") {
      bv_idx.push_back(1);
      var_name_found.emplace_back("U");
      ++n_found;
    } else if (name == "V") {
      bv_idx.push_back(2);
      var_name_found.emplace_back("V");
      ++n_found;
    } else if (name == "W") {
      bv_idx.push_back(3);
      var_name_found.emplace_back("W");
      ++n_found;
    } else if (name == "PRESSURE" || name == "P") {
      bv_idx.push_back(4);
      var_name_found.emplace_back("Pressure");
      ++n_found;
    } else if (name == "TEMPERATURE" || name == "T") {
      bv_idx.push_back(5);
      var_name_found.emplace_back("Temperature");
      ++n_found;
    } else if (name == "TKE") {
      sv_idx.push_back(n_spec);
      var_name_found.emplace_back("TKE");
      ++n_found;
    } else if (name == "OMEGA") {
      sv_idx.push_back(n_spec + 1);
      var_name_found.emplace_back("Omega");
      ++n_found;
    } else if (name == "MIXTUREFRACTION" || name == "Z") {
      sv_idx.push_back(n_spec + 2);
      var_name_found.emplace_back("MixtureFraction");
      ++n_found;
    } else if (name == "MIXTUREFRACTIONVARIANCE") {
      sv_idx.push_back(n_spec + 3);
      var_name_found.emplace_back("MixtureFractionVariance");
      ++n_found;
    } else if (n_spec > 0) {
      auto it = spec_list.find(name);
      if (it != spec_list.end()) {
        sv_idx.push_back(it->second);
        var_name_found.emplace_back(name);
        ++n_found;
      } else {
        if (parameter.get_int("myid") == 0) {
          printf("The variable %s is not found in the variable list.\n", name.c_str());
        }
      }
    } else if (parameter.get_int("n_ps") > 0) {
      const int n_ps{parameter.get_int("n_ps")};
      const int i_ps{parameter.get_int("i_ps")};
      if (name == "PS") {
        for (int i = 0; i < n_ps; ++i) {
          sv_idx.push_back(i_ps + i);
          var_name_found.emplace_back("PS" + std::to_string(i + 1));
          ++n_found;
        }
      } else {
        if (parameter.get_int("myid") == 0) {
          printf("The variable %s is not found in the variable list.\n", name.c_str());
        }
      }
    } else {
      if (parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  n_bv = static_cast<int>(bv_idx.size());
  n_sv = static_cast<int>(sv_idx.size());
  // The +1 is for physical time
  n_var = n_bv + n_sv + 1;
  h_ptr->n_bv = n_bv;
  h_ptr->n_sv = n_sv;
  h_ptr->n_var = n_var;
  cudaMalloc(&h_ptr->bv_label, sizeof(int) * n_bv);
  cudaMalloc(&h_ptr->sv_label, sizeof(int) * n_sv);
  cudaMemcpy(h_ptr->bv_label, bv_idx.data(), sizeof(int) * n_bv, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->sv_label, sv_idx.data(), sizeof(int) * n_sv, cudaMemcpyHostToDevice);

  return var_name_found;
}

Monitor::~Monitor() {
  for (const auto fp: files) {
    fclose(fp);
  }
}

void Monitor::monitor_point(int step, real physical_time, const std::vector<Field> &field) {
  if (counter_step == 0)
    step_start = step;

  for (int b = 0; b < n_block; ++b) {
    if (n_point[b] > 0) {
      constexpr auto tpb{128};
      const auto bpg{(n_point[b] - 1) / tpb + 1};
      record_monitor_data<<<bpg, tpb>>>(field[b].d_ptr, d_ptr, b, counter_step % output_file, physical_time);
    }
  }
  ++counter_step;
}

void Monitor::output_point_monitors() {
  cudaMemcpy(mon_var_h.data(), h_ptr->data.data(), sizeof(real) * n_var * output_file * n_point_total,
             cudaMemcpyDeviceToHost);

  for (int p = 0; p < n_point_total; ++p) {
    for (int l = 0; l < counter_step; ++l) {
      fprintf(files[p], "%d\t", step_start + l);
      for (int k = 0; k < n_var; ++k) {
        fprintf(files[p], "%e\t", mon_var_h(k, l, p));
      }
      fprintf(files[p], "\n");
    }
  }
  counter_step = 0;
}

__global__ void
record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, int blk_id, int counter_pos, real physical_time) {
  const auto idx = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  if (idx >= monitor_info->n_point[blk_id])
    return;
  const auto idx_tot = monitor_info->disp[blk_id] + idx;
  const auto i = monitor_info->is_d[idx_tot];
  const auto j = monitor_info->js_d[idx_tot];
  const auto k = monitor_info->ks_d[idx_tot];

  auto &data = monitor_info->data;
  const auto bv_label = monitor_info->bv_label;
  const auto sv_label = monitor_info->sv_label;
  const auto n_bv{monitor_info->n_bv};
  int var_counter{0};
  for (int l = 0; l < n_bv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->bv(i, j, k, bv_label[l]);
    ++var_counter;
  }
  for (int l = 0; l < monitor_info->n_sv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->sv(i, j, k, sv_label[l]);
    ++var_counter;
  }
  data(var_counter, counter_pos, idx_tot) = physical_time;
}

BlockMonitor::BlockMonitor(const Parameter &parameter, const Mesh &mesh_) {
  if (!parameter.get_int("if_monitor_blocks")) {
    return;
  }

  // First, get the variables to monitor.
  setup_labels_to_monitor(parameter);

  // for every group
  const auto monitor_file_name{parameter.get_string("monitor_block_file")};
  const auto myid{parameter.get_int("myid")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  int counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    int pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    int b, il, ir, jl, jr, kl, kr;
    line_stream >> b >> il >> ir >> jl >> jr >> kl >> kr;
    group_range.emplace_back(std::array<int, 7>{b, il, ir, jl, jr, kl, kr});
    ++n_block_mon;
    ++counter;
  }
  const std::filesystem::path out_dir("output/monitor");
  if (exists(out_dir)) {
    // If we are starting a new simulation, we need to rename the old directory to avoid overwriting.
    if (parameter.get_int("initial") == 0) {
      if (parameter.get_int("myid") == 0) {
        std::filesystem::path old_dir = out_dir;
        old_dir += "_old";
        if (exists(old_dir)) {
          remove_all(old_dir);
        }
        std::filesystem::rename(out_dir, old_dir);
        printf("The output directory %s already exists. The old directory is renamed to %s.\n",
               out_dir.string().c_str(), old_dir.string().c_str());
      }
    }
  }
  if (!exists(out_dir) && parameter.get_int("myid") == 0) {
    create_directories(out_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // write the mesh and jacobian
  ty = new MPI_Datatype[n_block_mon];
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];
    const auto &b = mesh_[bid];

    MPI_Datatype ty1;
    const int l_size[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const int il{group_range[blk][1]}, ir{group_range[blk][2]}, jl{group_range[blk][3]}, jr{group_range[blk][4]},
        kl{group_range[blk][5]}, kr{group_range[blk][6]};
    const auto nx{ir - il + 1}, ny{jr - jl + 1}, nz{kr - kl + 1};
    const int small_size[3]{nx, ny, nz};
    const int start_idx[3]{b.ngg + il, b.ngg + jl, b.ngg + kl};
    MPI_Type_create_subarray(3, l_size, small_size, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty1);
    MPI_Type_commit(&ty1);
    ty[blk] = ty1;

    MPI_File fp;
    char file_name[1024];
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-mesh.bin", out_dir.string().c_str(), myid, bid, il, ir, jl,
            jr, kl, kr);
    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
    MPI_Status status;
    // write x, y, z
    MPI_File_write(fp, &nx, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &ny, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &nz, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, &n_var, 1, MPI_INT32_T, &status);
    MPI_File_write(fp, b.x.data(), 1, ty1, &status);
    MPI_File_write(fp, b.y.data(), 1, ty1, &status);
    MPI_File_write(fp, b.z.data(), 1, ty1, &status);
    MPI_File_write(fp, b.jacobian.data(), 1, ty1, &status);
    MPI_File_close(&fp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void BlockMonitor::output_data(const Parameter &parameter, const std::vector<Field> &field, real t) const {
  if (n_block_mon <= 0)
    return;

  const std::filesystem::path out_dir("output/monitor");
  const int myid{parameter.get_int("myid")};
  for (int blk = 0; blk < n_block_mon; ++blk) {
    const auto bid = group_range[blk][0];
    const auto &f = field[bid];

    const int il{group_range[blk][1]}, ir{group_range[blk][2]};

    MPI_File fp;
    char file_name[1024];
    sprintf(file_name, "%s/P%dB%dIL%dIR%dJL%dJR%dKL%dKR%d-data.bin", out_dir.string().c_str(), myid, bid, il, ir,
            group_range[blk][3], group_range[blk][4], group_range[blk][5], group_range[blk][6]);
    // check the size of the file, if it is larger than 10GB, rename it with the current date and time
    if (std::filesystem::exists(file_name)) {
      constexpr unsigned long long maxFileSize = 10ull * 1024ull * 1024ull * 1024ull; // 10GB
      if (std::filesystem::file_size(file_name) > maxFileSize) {
        // The date should be in format YYYYMMDD_HHMMSS, and it should be in front of '.bin'
        // Get the current time
        auto now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        const std::tm *tm = std::localtime(&now_c);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm);

        std::string new_file_name = file_name;
        new_file_name += "_" + std::string(buffer);
        std::filesystem::rename(file_name, new_file_name);
      }
    }

    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &fp);
    MPI_Status status;
    MPI_File_write(fp, &t, 1, MPI_DOUBLE, &status);
    const auto &ty1 = ty[blk];
    for (int l = 0; l < n_bv; ++l) {
      MPI_File_write(fp, f.bv[bv_label[l]], 1, ty1, &status);
    }
    for (int l = 0; l < n_sv; ++l) {
      MPI_File_write(fp, f.sv[sv_label[l]], 1, ty1, &status);
    }
    for (int l = 0; l < n_ov; ++l) {
      MPI_File_write(fp, f.ov[ov_label[l]], 1, ty1, &status);
    }
    MPI_File_close(&fp);
  }
}

void BlockMonitor::setup_labels_to_monitor(const Parameter &parameter) {
  const auto &var_names = parameter.VNs;

  const auto mon_var = parameter.get_string_array("monitor_block_var");
  std::vector<int> var_found;
  const int n_ps{parameter.get_int("n_ps")};
  for (auto name: mon_var) {
    name = gxl::to_upper(name);
    bool found{false};
    for (auto &[N, l]: var_names) {
      if (N == name) {
        var_found.push_back(l);
        found = true;
        break;
      }
    }
    if (!found) {
      if (n_ps > 0 && name == "PS") {
        for (int i = 0; i < n_ps; ++i) {
          var_found.push_back(100 + i);
        }
        found = true;
      }
      if (!found && parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  const auto ov_labels = parameter.get_int_array("ov_labels");
  for (auto l: var_found) {
    if (l < 6) {
      ++n_bv;
      bv_label.push_back(l);
    } else if (l >= 1000) {
      ++n_sv;
      sv_label.push_back(l - 1000);
    } else if (l >= 100) {
      ++n_sv;
      sv_label.push_back(l - 100);
    } else {
      for (int ll = 0; ll < ov_labels.size(); ++ll) {
        if (l == ov_labels[ll]) {
          ov_label.push_back(ll);
          ++n_ov;
          break;
        }
      }
    }
  }
  n_var = n_bv + n_sv + n_ov;
}
} // cfd

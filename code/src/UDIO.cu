/**
 * This is the file that every user needs to edit.
 */

#include "FieldIO.h"
#include "UDIO.h"

// First, which variables to output, we need the names of them.
int cfd::add_other_variable_name(std::vector<std::string> &var_name, const Parameter &parameter) {
  auto nv = static_cast<int>(var_name.size());

  /**********************************************************************************************/
  // TODO: Here is what you need to modify
  const std::array<std::string, UserDefineIO::n_static_auxiliary> static_auxiliary_var_name =
      {/*"wall_distance"*/};
  const std::array<std::string, UserDefineIO::n_dynamic_auxiliary> dynamic_auxiliary_var_name =
      {/*"fd""StrainRate", "Vorticity", "F2"*/};
  /**********************************************************************************************/

  auto z_iter = var_name.cbegin() + 3;
  int counter{0};
  for (const auto &name: static_auxiliary_var_name) {
    var_name.insert(z_iter, name);
    ++counter;
    z_iter = var_name.cbegin() + 3 + counter;
    ++nv;
  }

  for (const auto &name: dynamic_auxiliary_var_name) {
    var_name.push_back(name);
    ++nv;
  }

  return nv;
}

// Next, the variables that would not change during the whole simulation, such as, wall distance.
MPI_Offset cfd::write_static_max_min(MPI_Offset offset, const Field &field, int ngg, MPI_File &fp) {
  if constexpr (UserDefineIO::n_static_auxiliary <= 0)
    return offset;

  real min_val = field.ov(-ngg, -ngg, -ngg, 0);
  real max_val = field.ov(-ngg, -ngg, -ngg, 0);
  auto &b = field.block;
  for (int k = -ngg; k < b.mz + ngg; ++k) {
    for (int j = -ngg; j < b.my + ngg; ++j) {
      for (int i = -ngg; i < b.mx + ngg; ++i) {
        min_val = std::min(min_val, field.ov(i, j, k, 0));
        max_val = std::max(max_val, field.ov(i, j, k, 0));
      }
    }
  }
  MPI_Status status;
  MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
  offset += 8;
  MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
  offset += 8;
  return offset;
}

// Next, the variables that would not change during the whole simulation, such as, wall distance.
MPI_Offset
cfd::write_dynamic_max_min_first_step(MPI_Offset offset, const cfd::Field &field, int ngg, MPI_File &fp) {
  constexpr real zero{0};
  MPI_Status status;
  for (int l = 0; l < UserDefineIO::n_dynamic_auxiliary; ++l) {
    MPI_File_write_at(fp, offset, &zero, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &zero, 1, MPI_DOUBLE, &status);
    offset += 8;
  }
  return offset;
}

MPI_Offset cfd::write_dynamic_max_min(MPI_Offset offset, const cfd::Field &field, int ngg, MPI_File &fp) {
  MPI_Status status;
  const auto &b = field.block;
  for (int l = 0; l < UserDefineIO::n_dynamic_auxiliary; ++l) {
    real min_val = field.udv(-ngg, -ngg, -ngg, l);
    real max_val = field.udv(-ngg, -ngg, -ngg, l);
    for (int k = -ngg; k < b.mz + ngg; ++k) {
      for (int j = -ngg; j < b.my + ngg; ++j) {
        for (int i = -ngg; i < b.mx + ngg; ++i) {
          min_val = std::min(min_val, field.udv(i, j, k, l));
          max_val = std::max(max_val, field.udv(i, j, k, l));
        }
      }
    }

    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
  }
  return offset;
}

MPI_Offset
cfd::write_static_array(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, long long int mem_sz) {
  if constexpr (UserDefineIO::n_static_auxiliary <= 0)
    return offset;
  // This function is called only once throughout the whole simulation
  MPI_Status status;
  // Write the wall distance out
  MPI_File_write_at(fp, offset, field.ov[0], 1, ty, &status);
  offset += mem_sz;

  return offset;
}

MPI_Offset
cfd::write_dynamic_array(MPI_Offset offset, const Field &field, MPI_File &fp, MPI_Datatype ty, long long int mem_sz) {
  MPI_Status status;
  for (int l = 0; l < UserDefineIO::n_dynamic_auxiliary; ++l) {
    MPI_File_write_at(fp, offset, field.udv[l], 1, ty, &status);
    offset += mem_sz;
  }

  return offset;
}

void cfd::copy_auxiliary_data_from_device(cfd::Field &field, int size) {
  auto &udv = field.udv;
  auto &h_ptr = field.h_ptr;
  for (int l = 0; l < UserDefineIO::n_dynamic_auxiliary; ++l) {
    cudaMemcpy(udv[l], h_ptr->udv[l], size * sizeof(real), cudaMemcpyDeviceToHost);
  }
}

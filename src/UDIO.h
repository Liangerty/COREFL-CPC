#pragma once


namespace cfd {

struct UserDefineIO {
  /**********************************************************************************************/
  constexpr static int n_auxiliary = 0;
  constexpr static int n_static_auxiliary = 0;
  constexpr static int n_dynamic_auxiliary = n_auxiliary - n_static_auxiliary;
  /**********************************************************************************************/
};

struct Field;

void copy_auxiliary_data_from_device(Field &field, int size);

}
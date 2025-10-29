#pragma once

#include <vector>

#ifdef __CUDACC__

#include <cuda_runtime.h>

#endif
// #include "Define.h"

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#ifdef __CUDACC__
namespace ggxl {
template<typename T>
class MatrixDyn {
  int mx{0}, my{0};
  int sz{0};
  T *data_ = nullptr;
public:
  __device__ MatrixDyn(const int nx, const int ny, T d = T{}) : mx{nx}, my{ny}, sz{nx * ny}, data_(new T[sz]) {}

//   //__host__ MatrixDyn(const int nx, const int ny, T d = T{}) : mx{nx}, my{ny}, sz{nx * ny}, data_(nullptr) {
//   //  cudaMalloc(&data_, sz * sizeof(T));
//   //}
  MatrixDyn() = default;

  void init_with_size(const int nx, const int ny) {
    mx = nx;
    my = ny;
    sz = nx * ny;
    cudaMalloc(&data_, sz * sizeof(T));
  }

  T *data() { return data_; }

  auto size() { return sz; }

  CUDA_CALLABLE_MEMBER T &operator()(const int i, const int j) { return data_[i * my + j]; }

  CUDA_CALLABLE_MEMBER const T &operator()(const int i, const int j) const { return data_[i * my + j]; }

  __device__ void deallocate_matrix() { delete[] data_; }
};
}
#endif

namespace gxl {
template<typename T, int M, int N, int BaseNumber = 0>
class Matrix {
public:
  __host__ __device__ T &operator()(const int i, const int j) {
    return data_[i * N + j - bias_];
  }

  __host__ __device__ const T &operator()(const int i, const int j) const {
    return data_[i * N + j - bias_];
  }

  __device__ const T *data() const {
    return data_;
  }

private:
//  int size_{M * N};
  int bias_{BaseNumber * (N + 1)};
  T data_[M * N]{};
};

template<typename T>
class MatrixDyn {
  int mx{0}, my{0};
  int size{0};
  std::vector<T> data_;
public:
  explicit MatrixDyn(const int nx = 0, const int ny = 0, T d = T{}) : mx{nx}, my{ny}, size{nx * ny}, data_(size, d) {}

  T *data() { return data_.data(); }

  const T *data() const { return data_.data(); }

  T &operator()(const int i, const int j) { return data_[i * my + j]; }

  const T &operator()(const int i, const int j) const { return data_[i * my + j]; }

  /**
   * \brief Get the reference of i-th element of the mx*1 matrix or the array with mx elements
   * \details Do not use this as possible. This is designed for the case when the matrix is only one dimensional, or
   * one of the 2 dimensions is 1. Then the matrix decays to an array. And the access can be simplified as this.
   * \param i the subscript of a one-dimensional array
   * \return reference of the i-th element of the array
   */
  T &operator[](const int i) { return data_[i]; }

  /**
   * \brief Get the constant reference of the i-th element of the mx*1 matrix or the array with mx elements
   * \details Do not use this as possible. This is designed for the case when the matrix is only one dimensional, or
   * one of the 2 dimensions is 1. Then the matrix decays to an array. And the access can be simplified as this.
   * \param i the subscript of a one-dimensional array
   * \return constant reference of the i-th element of the array
   */
  const T &operator[](const int i) const { return data_[i]; }

  [[nodiscard]] auto row(const int i) const { return data_.data() + i * my; }

  void resize(const int nx, const int ny, T val) {
    mx = nx;
    my = ny;
    size = nx * ny;
    data_.resize(size, val);
  }

  void resize(const int nx, const int ny) {
    mx = nx;
    my = ny;
    size = nx * ny;
    data_.resize(size);
  }

  [[nodiscard]] int n_col() const { return my; }
};

}

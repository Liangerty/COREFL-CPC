#pragma once

#include "Array.hpp"

namespace ggxl {
template<typename T, Major major = Major::ColMajor>
class Array3D {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  __device__ const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  T *data() { return val; }

  auto size() { return sz; }

  cudaError_t deallocate_memory() {
    cudaError_t err = cudaFree(val);
    return err;
  }
};

template<typename T, Major major>
inline cudaError_t Array3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  int ng = n_ghost;
  int n1 = dim1;
  int n2 = dim2;
  int n3 = dim3;
  if constexpr (major == Major::ColMajor) {
    disp2 = n1 + 2 * ng;
  } else {
    disp2 = n3 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

template<typename T, Major major = Major::ColMajor>
class Array3DHost {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;
//  int ng = 0;
//  int n1 = 0, n2 = 0, n3 = 0;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost = 0);

  T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  T *data() { return val; }

  auto size() { return sz; }
};

template<typename T, Major major>
inline cudaError_t Array3DHost<T, major>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  int ng = n_ghost;
  int n1 = dim1;
  int n2 = dim2;
  int n3 = dim3;
  if constexpr (major == Major::ColMajor) {
    disp2 = n1 + 2 * ng;
  } else {
    disp2 = n3 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaHostAlloc(&val, sz * sizeof(T), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    printf(
        "The VectorField3DHost isn't allocated by cudaHostAlloc, not enough page-locked memory. Use malloc instead\n");
    val = (real *) malloc(sz * sizeof(T));
  }
//  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

template<typename T, Major major = Major::ColMajor>
class VectorField3D {
private:
  int disp1 = 0, disp2 = 0, n4 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  __device__ const T &operator()(const int i, const int j, const int k, const int l) const {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *operator[](int l) {
    static_assert(major == Major::ColMajor);
    return val + l * sz;
//    return &val[l * sz];
  }

  const T *operator[](int l) const {
    static_assert(major == Major::ColMajor);
    return val + l * sz;
//    return &val[l * sz];
  }

  T *data() { return val; }

  auto size() { return sz; }

  cudaError_t deallocate_memory() {
    cudaError_t err = cudaFree(val);
    return err;
  }
};

template<typename T, Major major>
inline cudaError_t
VectorField3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost) {
  int ng = n_ghost;
  int n1 = dim1;
  int n2 = dim2;
  int n3 = dim3;
  n4 = dim4;
  if constexpr (major == Major::RowMajor) {
    disp2 = (n3 + 2 * ng) * n4;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else { // Column major
    disp2 = n1 + 2 * ng;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * n4 * sizeof(T));
  return err;
}

template<typename T>
class VectorField2D {
private:
  int disp1 = 0, n3 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost);

  __device__ T &operator()(const int i, const int j, const int l) {
    return val[j * disp1 + i + dispt + l * sz];
  }

  __device__ const T &operator()(const int i, const int j, const int l) const {
    return val[j * disp1 + i + dispt + l * sz];
  }

  T *operator[](int l) {
    return val + l * sz;
  }

  const T *operator[](int l) const {
    return val + l * sz;
  }

  T *data() { return val; }

  auto size() { return sz; }

  cudaError_t deallocate_memory() {
    cudaError_t err = cudaFree(val);
    return err;
  }
};

template<typename T>
inline cudaError_t VectorField2D<T>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  int ng = n_ghost;
  int n1 = dim1;
  int n2 = dim2;
  n3 = dim3;
  // Column major
  disp1 = n1 + 2 * ng;
  dispt = (disp1 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * n3 * sizeof(T));
  return err;
}

template<typename T>
class VectorField2DHost {
private:
  int disp1 = 0, n3 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;

public:
  explicit VectorField2DHost() = default;

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost);

  T &operator()(const int i, const int j, const int l) {
    return val[j * disp1 + i + dispt + l * sz];
  }

  const T &operator()(const int i, const int j, const int l) const {
    return val[j * disp1 + i + dispt + l * sz];
  }

  T *operator[](int l) {
    return val + l * sz;
  }

  const T *operator[](int l) const {
    return val + l * sz;
  }

  T *data() { return val; }

  auto size() { return sz; }

  cudaError_t deallocate_memory() {
    cudaError_t err = cudaFreeHost(val);
    return err;
  }
};

template<typename T>
inline cudaError_t VectorField2DHost<T>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  int ng = n_ghost;
  int n1 = dim1;
  int n2 = dim2;
  n3 = dim3;
  // Column major
  disp1 = n1 + 2 * ng;
  dispt = (disp1 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng);
  if (val != nullptr)
    cudaFreeHost(val);
  cudaError_t err = cudaHostAlloc(&val, sz * n3 * sizeof(T), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    printf(
        "The VectorField2DHost isn't allocated by cudaHostAlloc, not enough page-locked memory. Use malloc instead\n");
    val = (real *) malloc(sz * n3 * sizeof(T));
  }
  return err;
}

template<typename T, Major major = Major::ColMajor>
class VectorField3DHost {
  int disp2{0}, disp1{0}, dispt{0}, n4{0}, sz{0};
  T *data_ = nullptr;
//  int ng{0}, n1{0}, n2{0}, n3{0};
//  std::vector<T> data_;
public:
//  explicit VectorField3DHost(int dim1, int dim2, int dim3, int dim4, int n_ghost);

  auto data() const { return data_; }
//  auto data() const { return data_.data(); }

  auto data() { return data_; }

//  auto data() { return data_.data(); }
  auto size() { return sz; }

  /**
   * \brief Get the l-th variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \param l variable index in a vector
   * \return the l-th variable at position (i,j,k)
   */
  T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  const T &operator()(const int i, const int j, const int k, const int l) const {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *operator[](int l) {
    static_assert(major == Major::ColMajor);
    return &data_[l * sz];
  }

  const T *operator[](int l) const {
    static_assert(major == Major::ColMajor);
    return &data_[l * sz];
  }

  void resize(int ni, int nj, int nk, int nl, int ngg);

  int n_var() const { return n4; }

  cudaError_t deallocate_memory() {
    auto err = cudaFreeHost(data_);
    return err;
  }
};

template<typename T, Major major>
void VectorField3DHost<T, major>::resize(int ni, int nj, int nk, int nl, int ngg) {
  int ng = ngg;
  int n1 = ni + 2 * ngg;
  int n2 = nj + 2 * ngg;
  int n3 = nk + 2 * ngg;
  n4 = nl;
  sz = n1 * n2 * n3;
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 * n4;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else {
    disp2 = n1;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  if (data_ != nullptr)
    cudaFreeHost(data_);
  cudaError_t err = cudaHostAlloc(&data_, sz * n4 * sizeof(T), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    printf(
        "The VectorField3DHost isn't allocated by cudaHostAlloc, not enough page-locked memory. Use malloc instead\n");
    data_ = (real *) malloc(sz * n4 * sizeof(T));
  }
  cudaMemset(data_, 0, sz * n4 * sizeof(T));
//  data_.resize(n1 * n2 * n3 * n4, t);
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

}
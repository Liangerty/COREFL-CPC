#pragma once
#include <cmath>
#include "Array.hpp"

namespace gxl {
template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
int sgn(T a) {
  return a < 0 ? -1 : 1;
}

template<typename T>
int del(T a, T b) {
  return std::abs(a) == std::abs(b) ? 1 : 0;
}

template<int DIFFERENCE_ORDER>
double Derivatives(Array3D<double> &phi, int i, int j, int k, int type) {
  constexpr int p = DIFFERENCE_ORDER;
  const auto extent = phi.extent();
  const int imax = extent[0];
  const int jmax = extent[1];
  const int kmax = extent[2];
  const int ngg = extent[3];

  if (type == 1) //Calculate the first derivative in the ξ direction.
  {
    int s = i - p / 2;
    if (i <= -ngg + p / 2)
      s = -ngg;
    else if (i >= imax - 1 + ngg - p / 2)
      s = imax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l) {
          denominator *= static_cast<double>(l - a);
        }
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(i - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(l, j, k);
    }
    return D;
  }
  if (type == 2) //Calculate the first derivative in the η direction.
  {
    int s = j - p / 2;
    if (j <= -ngg + p / 2)
      s = -ngg;
    else if (j >= jmax - 1 + ngg - p / 2)
      s = jmax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l)
          denominator *= static_cast<double>(l - a);
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(j - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, l, k);
    }
    return D;
  }
  if (type == 3) //Calculate the first derivative in the ζ direction.
  {
    int s = k - p / 2;
    if (k <= -ngg + p / 2)
      s = -ngg;
    else if (k >= kmax - 1 + ngg - p / 2)
      s = kmax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l)
          denominator *= static_cast<double>(l - a);
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(k - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, j, l);
    }
    return D;
  }
  return 0.0;
}

template<int DIFFERENCE_ORDER>
double Derivatives(VectorField3D<double> &phi, int i, int j, int k, int v, int type) {
  constexpr int p = DIFFERENCE_ORDER;
  const auto extent = phi.extent();
  const int imax = extent[0];
  const int jmax = extent[1];
  const int kmax = extent[2];
  const int nv = extent[3];
  const int ngg = extent[4];

  if (type == 1) //Calculate the first derivative in the ξ direction.
  {
    int s = i - p / 2;
    if (i <= -ngg + p / 2)
      s = -ngg;
    else if (i >= imax - 1 + ngg - p / 2)
      s = imax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l) {
          denominator *= static_cast<double>(l - a);
        }
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(i - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(l, j, k, v);
    }
    return D;
  }
  if (type == 2) //Calculate the first derivative in the η direction.
  {
    int s = j - p / 2;
    if (j <= -ngg + p / 2)
      s = -ngg;
    else if (j >= jmax - 1 + ngg - p / 2)
      s = jmax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l)
          denominator *= static_cast<double>(l - a);
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(j - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, l, k, v);
    }
    return D;
  }
  if (type == 3) //Calculate the first derivative in the ζ direction.
  {
    int s = k - p / 2;
    if (k <= -ngg + p / 2)
      s = -ngg;
    else if (k >= kmax - 1 + ngg - p / 2)
      s = kmax - 1 + ngg - p;

    double D = 0;
    for (int l = s; l <= s + p; ++l) {
      double denominator = 1; //Denominator of the interpolated polynomial coefficients.
      for (int a = s; a <= s + p; ++a) {
        if (a != l)
          denominator *= static_cast<double>(l - a);
      }
      double numerator = 0; //Numerator of the interpolating polynomial coefficients.
      for (int m = s; m <= s + p; ++m) {
        if (m != l) {
          double mul = 1;
          for (int a = s; a <= s + p; ++a) {
            if ((a != l) && (a != m))
              mul *= static_cast<double>(k - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, j, l, v);
    }
    return D;
  }
  return 0.0;
}

// Solve the inverse function of complementary error function
// Type T should be a float number
// The current implementation is based on Newton's iteration
template<typename T>
T erfcInv(T z, T eps = 1e-5) {
  constexpr int step_max{50};
  T err{1};
  int step{0};
  T x{1};
  if (z > 1) x = -1;
  if (std::abs(z - 1) < 0.02) return 0;

  const double inv_sqrt_pi = 1.0 / sqrt(3.14159265358979);

  while (step < step_max && err > eps) {
    ++step;
    T f_x = std::erfc(x) - z;
    T df_dx = -2 * inv_sqrt_pi * std::exp(-x * x);
    T x1 = x - f_x / df_dx;
    err = std::abs((x1 - x) / x);
    x = x1;
  }
  return x;
}

#ifdef __CUDACC__

// device only function, smallest power of 2 >= n
// Ref: Programming in Parallel with CUDA
template<class T=int>
__device__ int pow2ceil(int n) {
  int pow2 = 1 << (31 - __clz(n));
  if (n > pow2) pow2 = (pow2 << 1);
  return pow2;
}

#endif
}

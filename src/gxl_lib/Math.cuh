#pragma once

#include "Array.cuh"

namespace ggxl {
template<int ORDER, int type> __device__ double Derivatives(VectorField3D<double> &phi, int i, int j,
  int k, int v, int max, int ngg) {
  constexpr int p = ORDER;

  if constexpr (type == 1) { //Calculate the first derivative in the ξ direction.
    int s = i - p / 2;
    if (i <= -ngg + p / 2)
      s = -ngg;
    else if (i >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

    double D = 0.0;
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
            if (a != l && a != m)
              mul *= static_cast<double>(i - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(l, j, k, v);
    }
    return D;
  }
  if constexpr (type == 2) //Calculate the first derivative in the η direction.
  {
    int s = j - p / 2;
    if (j <= -ngg + p / 2)
      s = -ngg;
    else if (j >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

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
            if (a != l && a != m)
              mul *= static_cast<double>(j - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, l, k, v);
    }
    return D;
  }
  if constexpr (type == 3) //Calculate the first derivative in the ζ direction.
  {
    int s = k - p / 2;
    if (k <= -ngg + p / 2)
      s = -ngg;
    else if (k >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

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
            if (a != l && a != m)
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

template<int ORDER, int type> __device__ double Derivatives(Array3D<double> &phi, int i, int j, int k,
  int max, int ngg) {
  constexpr int p = ORDER;

  if constexpr (type == 1) { //Calculate the first derivative in the ξ direction.
    int s = i - p / 2;
    if (i <= -ngg + p / 2)
      s = -ngg;
    else if (i >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

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
            if (a != l && a != m)
              mul *= static_cast<double>(i - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(l, j, k);
    }
    return D;
  }
  if constexpr (type == 2) //Calculate the first derivative in the η direction.
  {
    int s = j - p / 2;
    if (j <= -ngg + p / 2)
      s = -ngg;
    else if (j >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

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
            if (a != l && a != m)
              mul *= static_cast<double>(j - a);
          }
          numerator += mul;
        }
      }
      D += numerator / denominator * phi(i, l, k);
    }
    return D;
  }
  if constexpr (type == 3) //Calculate the first derivative in the ζ direction.
  {
    int s = k - p / 2;
    if (k <= -ngg + p / 2)
      s = -ngg;
    else if (k >= max - 1 + ngg - p / 2)
      s = max - 1 + ngg - p;

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
            if (a != l && a != m)
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
}

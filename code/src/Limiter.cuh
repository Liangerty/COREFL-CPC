#pragma once
#include "Define.h"
#include "gxl_lib/Math.hpp"

namespace cfd{
// Limiter functions.
// 0-minmod
template<int Method=0>
struct Limiter{
  __device__ real apply_limiter(real a, real b);
};

template<int Method>
__device__ real Limiter<Method>::apply_limiter(real a, real b) {
  return 0.5 * (gxl::sgn(a) + gxl::sgn(b)) * min(std::abs(a), std::abs(b));
}


// https://stackoverflow.com/questions/25202250/c-template-instantiation-avoiding-long-switches
template<int...> struct IntList{};

__device__
real apply_limiter(int, IntList<>, real a, real b){}

template<int I, int...N>
__device__
real apply_limiter(int i, IntList<I,N...>, real a, real b) {
  if (I != i)
    return apply_limiter(i, IntList<N...>(), a,b);

  Limiter<I> limiter;
  return limiter.apply_limiter(a,b);
}

template<int ...N>
__device__
real apply_limiter(int i, real a, real b) {
  return apply_limiter(i, IntList<N...>(), a, b);
}

template __device__ real apply_limiter<0, 1>(int method, real a, real b);
}
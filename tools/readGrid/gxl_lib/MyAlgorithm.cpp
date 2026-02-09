#include "MyAlgorithm.h"
#include <algorithm>

namespace gxl {
template<typename T>
bool exists(const std::vector<T> &vec, T val){
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

template bool exists(const std::vector<int> &vec, int val);
} // gxl
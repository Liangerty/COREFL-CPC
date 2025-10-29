#include "MyAlgorithm.h"
#include <algorithm>
#include <string>

namespace gxl {
template<typename T>
bool exists(const std::vector<T> &vec, T val){
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

template bool exists(const std::vector<int> &vec, int val);
template bool exists(const std::vector<std::string> &vec, std::string val);
} // gxl
# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should be modified to be larger or equal to the species number.
#     However, for larger mechanism such as Grimech with more than 50 species, the resource may be limited. We only conduct tests up to 9 species 

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

add_compile_definitions(MAX_SPEC_NUMBER=9)
add_compile_definitions(MAX_REAC_NUMBER=19)
add_compile_definitions(MAX_PASSIVE_SCALAR_NUMBER=1)

project(COREFL-CPC CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(src)

include_directories(depends)

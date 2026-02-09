# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should not beyond 55, which is set by allowing the computation of CH4 in Grimech mechanism.
#     If larger mechanism is to be used, the variable should be modified to be larger or equal to the species number.
#     However, the correctness, if the resource is available for the corresponding kernel may need to be tested.
# 3. Combustion2Part or HighTempMultiPart must be used to specify the type of simulation.

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

add_compile_definitions(MAX_SPEC_NUMBER=9)
add_compile_definitions(MAX_REAC_NUMBER=19)
add_compile_definitions(MAX_PASSIVE_SCALAR_NUMBER=1)
add_compile_definitions(Combustion2Part)

project(COREFL-CPC CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(src)

include_directories(depends)

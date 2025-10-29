cmake_minimum_required(VERSION 3.28)
project(readGrid)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(readGrid main.cpp Parameter.cpp gxl_lib/MyString.cpp)
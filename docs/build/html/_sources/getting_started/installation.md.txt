# Compilation

To compile COREFL, an environment with the following compilers is required:

- **CUDA compiler**: supporting C++17 (nvcc > 11.0)
- **C++ compiler**: supporting C++20
- **MPI library**: supporting CUDA-aware MPI (E.g., OpenMPI > 1.8)
- **CMake**: supporting CUDA language

> Comments on the MPI library: Only a few vendors' MPI support CUDA-aware MPI, and only on Linux systems. Therefore, only Linux system supports the parallel running of COREFL. However, any MPI version supports the compilation and running in serial modes.

COREFL has been compiled and tested on Nvidia A100 GPU.
The most frequently used configuration by us on A100 is given for reference:
**CUDA 11.8 / gcc 11.3 / openmpi 4.1.5 / cmake 3.26.3**

All compilations are performed on **Linux** system. For Windows system, we successfully built the code with Visual Studio 2022 and CUDA, Microsoft MPI. You can also build it with CLion, but the toolchain must be the msvc instead of mingw. I would not include that here because large scale computations are always performed on Linux clusters.

The compilation consists of the following steps:

1. Navigate to the COREFL folder (the "`code`" folder here).
2. Modify the CMakeLists.txt:
   1. Modify the number in `set(CMAKE_CUDA_ARCHITECTURES 60)` according to the GPU compute capability. For example, this number is 60 for P100, 70 for V100, and 80 for A100.
   2. Modify `add_compile_definitions(MAX_SPEC_NUMBER=9)` according to the problem. The number should be larger than or equal to the species number to be used in computations. If no species is included, set it to 1.
   3. Modify `add_compile_definitions(MAX_REAC_NUMBER=19)` according to the problem. The number should be larger than or equal to the reaction number to be used in computations. If no reaction is included, set it to 1.
   4. Modify `add_compile_definitions(Combustion2Part)` according to the problem. If a general chemical reaction case is considered (combustion), set it to `Combustion2Part`. If a high-temerature air chemistry is to be simulated, set to `HighTempMultiPart`. Only the 5 species 6 reactions mechanism of air is supported currently when using `HighTempMultiPart`.
3. Load the compilation environment. For example, `module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1 cmake/3.26.3`
4. `cmake -Bbuild -DCMAKE_BUILD_TYPE=Release`
5. `cmake --build build --parallel 16`

The executable COREFL should appear in ths current folder.

list(APPEND src_file
    BoundCond.h BoundCond.cpp BoundCond.cu BoundCond.cuh BoundaryIO.h
    ChemData.h ChemData.cpp Constants.h
    DataCommunication.cuh DataCommunication.cu Define.h DParameter.cu DParameter.cuh DPLUR.cuh DPLUR.cu Driver.cu Driver.cuh DualTimeStepping.cuh DualTimeStepping.cu
    Element.h Element.cpp
    Field.h Field.cu FieldOperation.cuh FieldOperation.cu FiniteRateChem.cu FiniteRateChem.cuh FieldIO.h Fluctuation.cu Fluctuation.cuh
    ImplicitTreatmentHPP.cuh Initialize.cuh Initialize.cu InviscidScheme.cuh InviscidScheme.cu IOManager.h InviscidHybrid.cu
    kernels.cuh kernels.cu kernels.h
    Limiter.cuh Mesh.cu Mesh.cuh
    main.cu Mesh.h Mesh.cpp MixtureFraction.cpp MixtureFraction.h Monitor.cuh Monitor.cu MixingLayer.cuh MixingLayer.cu
    Parallel.h Parallel.cpp Parameter.h Parameter.cpp PostProcess.h PostProcess.cu ParameterSet.cpp ParameterSet.h
    Reconstruction.cu Reconstruction.cuh Residual.cu Residual.cuh RK.cuh RiemannSolver.cuh
    SchemeSelector.cuh SteadySim.cuh Simulate.cuh
    SpongeLayer.cu SpongeLayer.cuh SinglePointStat.cu SinglePointStat.cuh StrangSplitting.cuh StrangSplitting.cu
    Transport.cuh Transport.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Thermo.cuh Thermo.cu TurbMethod.hpp
    DigitalFilter.cu DigitalFilter.cuh
    UDIO.h UDIO.cu
    ViscousScheme.cuh ViscousScheme.cu
    WallDistance.cuh WENO.cu wenoOpt.cu

    stat_lib/TkeBudget.cu stat_lib/TkeBudget.cuh stat_lib/SpeciesStat.cu stat_lib/SpeciesStat.cuh

    gxl_lib/Array.hpp gxl_lib/Array.cuh
    gxl_lib/MyString.cpp gxl_lib/MyAlgorithm.cpp gxl_lib/MyAlgorithm.h gxl_lib/Math.cuh
    gxl_lib/Time.cpp gxl_lib/Time.h
    gxl_lib/MyAtomic.cuh
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_policy(SET CMP0074 NEW)
find_package(MPI REQUIRED)

add_executable(corefl ${src_file})

set_target_properties(corefl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

set_target_properties(corefl
    PROPERTIES
    CXX_STANDARD 20 # A newer version CMake 3.25.2 would support this
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(corefl
    PUBLIC
    ${CMAKE_SOURCE_DIR}/depends/include/
)

target_link_directories(corefl
    PUBLIC#PRIVATE
    "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
    "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
)

target_link_libraries(corefl
    PUBLIC#PRIVATE
    "$<$<CONFIG:Debug>:fmtd>"
    "$<$<CONFIG:Release>:fmt>"
    MPI::MPI_CXX
)

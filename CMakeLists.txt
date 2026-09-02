cmake_minimum_required(VERSION 3.18)
project(matmax LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(matmax STATIC
  lib/matx.cpp
  lib/cuda/matx_ops.cu
)

target_include_directories(matmax
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(matmax PUBLIC CUDA::cudart)

add_executable(tests
  tests/main.cpp
)

target_link_libraries(tests PRIVATE matmax)

add_executable(logreg
  examples/ml/logistic-regression.cpp
)

target_link_libraries(logreg PRIVATE matmax)

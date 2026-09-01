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

add_executable(matmax
  main.cpp
  test.cpp
  lib/matx.cpp
  lib/cuda/matx_ops.cu
)

target_include_directories(matmax PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(matmax PRIVATE CUDA::cudart)

add_executable(logreg
  ml/logistic-regression.cpp
  lib/matx.cpp
  lib/cuda/matx_ops.cu
)

target_include_directories(logreg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(logreg PRIVATE CUDA::cudart)

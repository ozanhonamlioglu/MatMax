#pragma once

#include <iostream>
#include <cuda_runtime.h>

int threadsPerBlock = 256;

inline int blocksPerGrid(int N) {
  return (N + threadsPerBlock - 1) / threadsPerBlock;
}

#define CUDA_CHECK(ans) do { gpuAssert(ans, __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if(code != cudaSuccess) {
    std::cerr << "GPU Error: " << cudaGetErrorString(code) << " in " << file << ":" << line << std::endl;
    if (abort) exit(code);
  }
}
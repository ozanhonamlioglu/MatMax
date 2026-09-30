#include "utils.cuh"
#include "matx_chain_ops.cuh"
#include <curand_kernel.h>
#include <iostream>

// Returns a device pointer holding M's data. Host data is copied into a new
// allocation (owned = true, caller must free it); an existing device pointer
// is reused as is (owned = false, it belongs to the chain's collector).
static float* to_device(const matx::ChainMatrix& M, size_t size, bool& owned) {
  if(M.mtx != nullptr) {
    float *d_M;
    CUDA_CHECK(cudaMalloc(&d_M, size));
    CUDA_CHECK(cudaMemcpy(d_M, M.mtx, size, cudaMemcpyHostToDevice));
    owned = true;
    return d_M;
  }

  owned = false;
  return M.device;
}

__global__ void cuda_matx_chain_add(float *A, float *B, float *Buffer, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index < N) {
    Buffer[index] = A[index] + B[index];
  }
}

float* matx_chain_add(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  int N = A.dims * A.rows;
  size_t size = N * sizeof(float);
  float *Buffer;
  bool ownA, ownB;

  float *d_A = to_device(A, size, ownA);
  float *d_B = to_device(B, size, ownB);
  CUDA_CHECK(cudaMalloc(&Buffer, size));

  int blocks = blocksPerGrid(N);
  cuda_matx_chain_add<<<blocks, threadsPerBlock>>>(d_A, d_B, Buffer, N);
  CUDA_CHECK(cudaGetLastError());

  if(ownA) CUDA_CHECK(cudaFree(d_A));
  if(ownB) CUDA_CHECK(cudaFree(d_B));

  return Buffer;
}

__global__ void cuda_matx_chain_sub(float *A, float *B, float *Buffer, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index < N) {
    Buffer[index] = A[index] - B[index];
  }
}

float* matx_chain_sub(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  int N = A.dims * A.rows;
  size_t size = N * sizeof(float);
  float *Buffer;
  bool ownA, ownB;

  float *d_A = to_device(A, size, ownA);
  float *d_B = to_device(B, size, ownB);
  CUDA_CHECK(cudaMalloc(&Buffer, size));

  int blocks = blocksPerGrid(N);
  cuda_matx_chain_sub<<<blocks, threadsPerBlock>>>(d_A, d_B, Buffer, N);
  CUDA_CHECK(cudaGetLastError());

  if(ownA) CUDA_CHECK(cudaFree(d_A));
  if(ownB) CUDA_CHECK(cudaFree(d_B));

  return Buffer;
}

__global__ void cuda_matx_chain_scale(float *A, float *Buffer, float scalar, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index < N) {
    Buffer[index] = A[index] * scalar;
  }
}

float* matx_chain_scale(const matx::ChainMatrix& A, float scalar) {
  int N = A.dims * A.rows;
  size_t size = N * sizeof(float);
  float *Buffer;
  bool ownA;

  float *d_A = to_device(A, size, ownA);
  CUDA_CHECK(cudaMalloc(&Buffer, size));

  int blocks = blocksPerGrid(N);
  cuda_matx_chain_scale<<<blocks, threadsPerBlock>>>(d_A, Buffer, scalar, N);
  CUDA_CHECK(cudaGetLastError());

  if(ownA) CUDA_CHECK(cudaFree(d_A));

  return Buffer;
}

// A is M x K, B is K x P, Buffer (output) is M x P
__global__ void cuda_matx_chain_mul(float *A, float *B, float *Buffer, int M, int K, int P) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index < M * P) {
    int row = index / P;
    int col = index % P;

    float sum = 0.0f;
    for(int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * P + col];
    }

    Buffer[index] = sum;
  }
}

float* matx_chain_mul(const matx::ChainMatrix& A, const matx::ChainMatrix& B) {
  int M = A.rows;
  int K = A.dims;
  int P = B.dims;
  float *Buffer;
  bool ownA, ownB;

  float *d_A = to_device(A, M * K * sizeof(float), ownA);
  float *d_B = to_device(B, K * P * sizeof(float), ownB);
  CUDA_CHECK(cudaMalloc(&Buffer, M * P * sizeof(float)));

  int blocks = blocksPerGrid(M * P);
  cuda_matx_chain_mul<<<blocks, threadsPerBlock>>>(d_A, d_B, Buffer, M, K, P);
  CUDA_CHECK(cudaGetLastError());

  if(ownA) CUDA_CHECK(cudaFree(d_A));
  if(ownB) CUDA_CHECK(cudaFree(d_B));

  return Buffer;
}

// A is Row x Col, Buffer (output) is Col x Row
__global__ void cuda_matx_chain_transpose(float *A, float *Buffer, int Row, int Col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index < Row * Col) {
    int row = index / Col;
    int col = index % Col;

    Buffer[col * Row + row] = A[index];
  }
}

float* matx_chain_transpose(const matx::ChainMatrix& A) {
  int N = A.dims * A.rows;
  size_t size = N * sizeof(float);
  float *Buffer;
  bool ownA;

  float *d_A = to_device(A, size, ownA);
  CUDA_CHECK(cudaMalloc(&Buffer, size));

  int blocks = blocksPerGrid(N);
  cuda_matx_chain_transpose<<<blocks, threadsPerBlock>>>(d_A, Buffer, A.rows, A.dims);
  CUDA_CHECK(cudaGetLastError());

  if(ownA) CUDA_CHECK(cudaFree(d_A));

  return Buffer;
}

void matx_chain_complete(const std::vector<matx::DeviceMemCollector>& collector, float *Host) {
  const matx::DeviceMemCollector& last = collector.back();
  size_t size = last.dims * last.rows * sizeof(float);

  CUDA_CHECK(cudaMemcpy(Host, last.buffer, size, cudaMemcpyDeviceToHost));

  matx_chain_free(collector);
}

void matx_chain_free(const std::vector<matx::DeviceMemCollector>& collector) {
  for(const matx::DeviceMemCollector& item : collector) {
    CUDA_CHECK(cudaFree(item.buffer));
  }
}

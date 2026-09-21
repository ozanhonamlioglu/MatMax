#include "utils.cuh"
#include <curand_kernel.h>
#include <iostream>
#include <ctime>
#include <chrono>

__global__ void cuda_matx_add(float* A, float* B, float* Buffer, int N) {
  // Calculate global thread index
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Prevent out-of-bound errors if array isn't a clean multiple of block size
  if (index < N) {
    Buffer[index] = A[index] + B[index];
  }
}

void matx_add(float* A, float* B, float* Buffer, int N) {
  size_t size = N * sizeof(float);
  float *d_A, *d_B, *d_Buffer;

  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_Buffer, size));

  // Copy data from host (vectors) to device (GPU)
  CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  int blocks = blocksPerGrid(N);
  cuda_matx_add<<<blocks, threadsPerBlock>>>(d_A, d_B, d_Buffer, N);
  CUDA_CHECK(cudaGetLastError());

  // Copy final result back to the host vector memory
  CUDA_CHECK(cudaMemcpy(Buffer, d_Buffer, size, cudaMemcpyDeviceToHost));

  // Clean up GPU memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Buffer));
}

__global__ void cuda_matx_sub(float* A, float* B, float* Buffer, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < N) {
    Buffer[index] = A[index] - B[index];
  }
}

void matx_sub(float* A, float* B, float* Buffer, int N) {
  size_t size = N * sizeof(float);
  float *d_A, *d_B, *d_Buffer;

  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_Buffer, size));

  CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  int blocks = blocksPerGrid(N);
  cuda_matx_sub<<<blocks, threadsPerBlock>>>(d_A, d_B, d_Buffer, N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(Buffer, d_Buffer, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Buffer));
}

__global__ void cuda_matx_scale(float* A, float* Buffer, float scalar, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < N) {
    Buffer[index] = A[index] * scalar;
  }
}

void matx_scale(float* A, float* Buffer, float scalar, int N) {
  size_t size = N * sizeof(float);
  float *d_A, *d_Buffer;

  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_Buffer, size));

  CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

  int blocks = blocksPerGrid(N);
  cuda_matx_scale<<<blocks, threadsPerBlock>>>(d_A, d_Buffer, scalar, N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(Buffer, d_Buffer, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_Buffer));
}

// A is M x K, B is K x P, Buffer (output) is M x P
__global__ void cuda_matx_mul(float* A, float* B, float* Buffer, int M, int K, int P) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < M * P) {
    int row = index / P;
    int col = index % P;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * P + col];
    }

    Buffer[index] = sum;
  }
}

void matx_mul(float* A, float* B, float* Buffer, int M, int K, int P) {
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * P * sizeof(float);
  size_t size_Buffer = M * P * sizeof(float);

  float *d_A, *d_B, *d_Buffer;

  CUDA_CHECK(cudaMalloc(&d_A, size_A));
  CUDA_CHECK(cudaMalloc(&d_B, size_B));
  CUDA_CHECK(cudaMalloc(&d_Buffer, size_Buffer));

  CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

  int blocks = blocksPerGrid(M * P);
  cuda_matx_mul<<<blocks, threadsPerBlock>>>(d_A, d_B, d_Buffer, M, K, P);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(Buffer, d_Buffer, size_Buffer, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Buffer));
}

__global__ void cuda_matx_randomf(float* matrix, int length, unsigned long long seed) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < length) {
    curandState state;
    curand_init(seed, index, 0, &state);
    matrix[index] = curand_uniform(&state);
  }
}

void matx_randomf(int length, float* Buffer) {
  unsigned long long seed = static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  
  float* A;
  size_t size = length * sizeof(float);

  CUDA_CHECK(cudaMalloc(&A, size));
  
  int blocks = blocksPerGrid(length);
  cuda_matx_randomf<<<blocks, threadsPerBlock>>>(A, length, seed);

  CUDA_CHECK(cudaMemcpy(Buffer, A, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(A));
}
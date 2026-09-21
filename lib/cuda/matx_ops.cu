#include <iostream>

#include "utils.cuh"

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

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Assuming your actual global kernel is named matx_add_kernel
  cuda_matx_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Buffer, N);
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

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cuda_matx_sub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Buffer, N);
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

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cuda_matx_scale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_Buffer, scalar, N);
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

  int threadsPerBlock = 256;
  int blocksPerGrid = (M * P + threadsPerBlock - 1) / threadsPerBlock;

  cuda_matx_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Buffer, M, K, P);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(Buffer, d_Buffer, size_Buffer, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Buffer));
}
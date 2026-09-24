#include "utils.cuh"
#include <curand_kernel.h>
#include <iostream>
#include <ctime>

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

void matx_randomf(int length, float* Buffer, unsigned long long seed) {
  float* A;
  size_t size = length * sizeof(float);

  CUDA_CHECK(cudaMalloc(&A, size));
  
  int blocks = blocksPerGrid(length);
  cuda_matx_randomf<<<blocks, threadsPerBlock>>>(A, length, seed);

  CUDA_CHECK(cudaMemcpy(Buffer, A, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(A));
}

__global__ void cuda_matx_equal(float* A, float* B, bool* d_is_equal, int length) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if(index < length) {
    if(A[index] != B[index]) {
      *d_is_equal = false;
    }
  }
}

bool matx_equal(float* A, float* B, int length) {
  size_t size = length * sizeof(float);
  float *d_A, *d_B;

  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));

  CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  bool* d_is_equal;
  CUDA_CHECK(cudaMallocManaged(&d_is_equal, sizeof(bool)));

  *d_is_equal = true;

  int blocks = blocksPerGrid(length);
  cuda_matx_equal<<<blocks, threadsPerBlock>>>(d_A, d_B, d_is_equal, length);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());

  bool is_equal = *d_is_equal;

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_is_equal));

  return is_equal;
}

__global__ void cuda_matx_transpose(float* A, int Row, int Col, float* Buffer) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // A is Row x Col, Buffer (output) is Col x Row
  if (index < Row * Col) {
    int row = index / Col;
    int col = index % Col;

    Buffer[col * Row + row] = A[index];
  }
}

void matx_transpose(float* A, int Row, int Col, float* Buffer) {
  int length = Row * Col;
  size_t size = length * sizeof(float);
  
  float *d_A, *Out;

  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&Out, size));

  int blocks = blocksPerGrid(length);
  cuda_matx_transpose<<<blocks, threadsPerBlock>>>(d_A, Row, Col, Out);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(Buffer, Out, size, cudaMemcpyDeviceToHost));

  cudaFree(Out);
  cudaFree(d_A);
}
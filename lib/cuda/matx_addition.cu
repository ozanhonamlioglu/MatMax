
__global__ void cuda_matx_add(float* A, float* B, float* Buffer, int N) {
  // Calculate global thread index
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Prevent out-of-bound errors if array isn't a clean multiple of block size
  if (index < N) {
    Buffer[index] = A[index] + B[index];
  }
}

extern "C" void matx_add(float* A, float* B, float* Buffer, int N) {
  size_t size = N * sizeof(float);
  float *d_A, *d_B, *d_Buffer;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_Buffer, size);

  // Copy data from host (vectors) to device (GPU)
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // 3. Define execution configuration and launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Assuming your actual global kernel is named matx_add_kernel
  cuda_matx_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Buffer, N);

  // Copy final result back to the host vector memory
  cudaMemcpy(Buffer, d_Buffer, size, cudaMemcpyDeviceToHost);

  // Clean up GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_Buffer);
}
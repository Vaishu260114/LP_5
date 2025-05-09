#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_multiply(float *a, float *b, float *c, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  if (row < n && col < n)
  {
    for (int i = 0; i < n; ++i)
    {
      sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}

int main()
{
  int n = 1024;
  size_t size = n * n * sizeof(float);

  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  cudaEvent_t start, stop;
  float elapsed_time;

  // Allocate host memory
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  // Initialize matrices
  for (int i = 0; i < n * n; ++i)
  {
    a[i] = i % n;
    b[i] = i % n;
  }

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Configure kernel launch parameters
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

  // Launch and time the kernel
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  matrix_multiply<<<blocks, threads>>>(d_a, d_b, d_c, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Copy result to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  printf("Elapsed time for matrix multiplication: %.2f ms\n", elapsed_time);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);

  return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Added error checking wrapper
#define CUDA_CHECK(err)                                    \
  do                                                       \
  {                                                        \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("CUDA error: %s at %s:%d\n",                  \
             cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                             \
    }                                                      \
  } while (0)

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
  cudaError_t err;

  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  cudaEvent_t start, stop;
  float elapsed_time;

  // Allocate and initialize host memory
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  if (!a || !b || !c)
  {
    printf("Host memory allocation failed\n");
    return 1;
  }

  // Initialize matrices with simpler pattern
  for (int i = 0; i < n * n; ++i)
  {
    a[i] = 1.0f;
    b[i] = 1.0f;
  }

  // Allocate device memory with error checking
  err = cudaMalloc(&d_a, size);
  CUDA_CHECK(err);
  err = cudaMalloc(&d_b, size);
  CUDA_CHECK(err);
  err = cudaMalloc(&d_c, size);
  CUDA_CHECK(err);

  // Copy data to device with error checking
  err = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  CUDA_CHECK(err);
  err = cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  CUDA_CHECK(err);

  // Configure kernel launch
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + threads.x - 1) / threads.x,
              (n + threads.y - 1) / threads.y);

  // Create events for timing
  err = cudaEventCreate(&start);
  CUDA_CHECK(err);
  err = cudaEventCreate(&stop);
  CUDA_CHECK(err);

  // Time the kernel execution
  err = cudaEventRecord(start);
  CUDA_CHECK(err);
  matrix_multiply<<<blocks, threads>>>(d_a, d_b, d_c, n);
  err = cudaEventRecord(stop);
  CUDA_CHECK(err);
  err = cudaEventSynchronize(stop);
  CUDA_CHECK(err);
  err = cudaEventElapsedTime(&elapsed_time, start, stop);
  CUDA_CHECK(err);

  // Check for kernel errors
  err = cudaGetLastError();
  CUDA_CHECK(err);

  // Copy result back
  err = cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  CUDA_CHECK(err);

  printf("Matrix multiplication completed in %.2f ms\n", elapsed_time);

  // Simple verification (check first element)
  printf("Result[0][0] = %.2f (should be %.2f)\n", c[0], (float)n);

  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
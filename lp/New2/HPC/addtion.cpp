#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * CUDA kernel for vector addition
 * @param a Input vector A
 * @param b Input vector B
 * @param c Output vector C (A + B)
 * @param n Number of elements in vectors
 */
__global__ void vectorAdd(const int *a, const int *b, int *c, int n)
{
  // Calculate global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds to avoid out-of-range memory access
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

int main()
{
  const int n = 1000000;            // Number of elements in vectors
  const int size = n * sizeof(int); // Total size in bytes

  // Host pointers
  int *h_a, *h_b, *h_c;

  // Device pointers
  int *d_a, *d_b, *d_c;

  // Allocate pinned host memory for better transfer performance
  cudaMallocHost((void **)&h_a, size);
  cudaMallocHost((void **)&h_b, size);
  cudaMallocHost((void **)&h_c, size);

  // Initialize input vectors
  for (int i = 0; i < n; i++)
  {
    h_a[i] = i;
    h_b[i] = i;
  }

  // Allocate device memory
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy data from host to device (asynchronous for potential overlap)
  cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Kernel launch configuration
  const int blockSize = 256; // Threads per block
  // Calculate grid size to cover all elements
  const int gridSize = (n + blockSize - 1) / blockSize;

  // Launch vector addition kernel
  vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  // Copy result back to host (synchronous to ensure completion)
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify and print first 10 results
  bool success = true;
  for (int i = 0; i < 10; i++)
  {
    if (h_c[i] != h_a[i] + h_b[i])
    {
      printf("Error at index %d: %d != %d + %d\n", i, h_c[i], h_a[i], h_b[i]);
      success = false;
      break;
    }
    printf("h_c[%d] = %d\n", i, h_c[i]);
  }

  // Final verification (optional, for thorough checking)
  if (success)
  {
    printf("First 10 elements correct. Verifying rest...\n");
    for (int i = 10; i < n; i++)
    {
      if (h_c[i] != h_a[i] + h_b[i])
      {
        printf("Error at index %d\n", i);
        success = false;
        break;
      }
    }
    if (success)
    {
      printf("All elements verified successfully!\n");
    }
  }

  // Cleanup - free all allocated memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);

  return success ? 0 : 1;
}
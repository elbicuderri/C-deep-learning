#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMul(int *c, const int *a, const int *b, int N) {
	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Iterate over row, and down column
	c[row * N + col] = 0;
	for (int k = 0; k < N; k++) {
		// Accumulate results for a single element
		c[row * N + col] += a[row * N + k] * b[k * N + col];
	}
}


int main() {
	// Matrix size of 3 x 3;
	int N = 3;

	// Size (in bytes) of matrix
	size_t bytes = N * N * sizeof(int);

	// Allocate host memory
	int a[] = { 1, 0, 1, 2, 1, 0, 3, 0, 1 };
	int b[] = { 3, 0, 1, 1, 0, 0, 1, 2, 2 };
	int *c = (int*)malloc(bytes);

	// Allocate device memory
	int *d_a;
	int *d_b;
	int *d_c;
	cudaMalloc((void**)&d_a, bytes);
	cudaMalloc((void**)&d_b, bytes);
	cudaMalloc((void**)&d_c, bytes);

	// Copy host-data to the device
	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	// Threads per CTA dimension
	int THREADS = 3;

	// Blocks per grid dimension (assumes THREADS divides N evenly)
	int BLOCKS = N / THREADS;

	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Launch kernel
	matrixMul << <BLOCKS, threads >> > (d_c, d_a, d_b, N);

	// Copy back to the host
	cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

	printf("a: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", a[i * N + j]);
		}
		printf("\n\n");
	}

	printf("b: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", b[i * N + j]);
		}
		printf("\n\n");
	}

	printf("c: \n\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", c[i * N + j]);
		}
		printf("\n\n");
	}

	// Check result
	//verify_result(a, b, c, N);

	//cout << "COMPLETED SUCCESSFULLY\n";

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

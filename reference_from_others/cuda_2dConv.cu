#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

#define MASK_DIM 2

#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int d_kernel[2 * 2];

__global__ void convolution_2d(int *output, int *input, int N)
{
	//, int K, int P, int Q, int t_count
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//
	//if (tid >= t_count)
	//	return;

	//int q_idx = tid % Q;
	//int idx = tid / Q;

	//int p_idx = idx % P;
	//idx /= P;

	//int k_idx = idx % K;

	//int n_idx = idx / K;

	//int sw_idx = q_idx - 1;
	//int sh_idx = p_idx - 1;

	int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	int output_col = blockIdx.x * blockDim.x + threadIdx.x;

	int input_row_start = output_row;
	int input_col_start = output_col;

	int input_row_end = input_row_start + MASK_DIM - 1;
	int input_col_end = input_col_start + MASK_DIM - 1;

	int sum = 0;
	for (int i = 0; i < MASK_DIM; i++) {
		for (int j = 0; j < MASK_DIM; j++) {
			if (input_row_start >= 0 && input_row_end < N ) {
				if (input_col_start >= 0 && input_col_end < N) {
					int m = input[(input_row_start + i) * N + (input_col_start + j)];
					int k = d_kernel[i * MASK_DIM + j];
					sum += m * k;
					printf("m: %d, k: %d, sum: %d \n\n", m, k, sum);
				}
			}
		}
	}
	output[output_row * (N - MASK_DIM + 1) + output_col] = sum;

}

void init_matrix(int *output, int dim) 
{
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			output[i * dim + j] = rand() % 10;
		}
	}

}

void print_image(int *image, int height, int width) 
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%2d ", image[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");

}

int main()
{
	int N = 3;
	int N_ = 1;
	int K = 1;
	int P = 2;
	int Q = 2;
	//int MASK_DIM = 2;

	int *h_matrix = (int*)malloc(N * N * sizeof(int));
	int *h_kernel = (int*)malloc(MASK_DIM * MASK_DIM * sizeof(int));

	init_matrix(h_matrix, N);
	init_matrix(h_kernel, MASK_DIM);

	print_image(h_matrix, N, N);
	print_image(h_kernel, MASK_DIM, MASK_DIM);

	int *d_matrix;
	cudaMalloc((void**)&d_matrix, N * N * sizeof(int));
	cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_kernel, h_kernel, MASK_DIM * MASK_DIM * sizeof(int));
		
	int *h_result = (int*)malloc(2 * 2 * sizeof(int));
	int *d_result;
	cudaMalloc((void**)&d_result, 2 * 2 * sizeof(int));

	//int THREADS = 2;
	//int BLOCKS = 256;

	//int t_count = N_ * K * P * Q;
	//int b_count = (t_count + BLOCKS - 1) / BLOCKS;

	dim3 block_dim(2, 2);
	dim3 grid_dim(1, 1);

	convolution_2d << <grid_dim, block_dim >> > (d_result, d_matrix, N);

	//convolution_2d <<<b_count, BLOCKS >>> (d_result, d_matrix, N);

	cudaMemcpy(h_result, d_result, 2 * 2 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("result: \n");
	print_image(h_result, 2, 2 );

	return 0;
}

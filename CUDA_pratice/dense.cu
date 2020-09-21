#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <float.h>

__global__ void dense_kernel(float *output, float *input, float *weight, float *bias, int N, int C, int K, int H, int W, int total_size)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	int k_idx = tid % K;
	int n_idx = tid / K;

	float sum = 0.0f;
	for (int i = 0; i < C * H * W; i++) {
			int input_index = n_idx * C * H * W + i;
			int weight_index = k_idx * C * H * W + i;
			sum += input[input_index] * weight[weight_index];
	}
	sum += bias[k_idx];
	output[tid] = sum;

}

void dense(float *output, float *input, float *weight, float *bias, int N, int C, int K, int H, int W)
{
	
	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * K;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


	dense_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (output, input, weight, bias, N, C, K, H, W, TOTAL_SIZE);
}
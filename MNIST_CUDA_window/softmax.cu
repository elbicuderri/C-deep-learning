#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>


__global__ void softmax_kernel(float *output, float *input, float *exp_sum, int batch, int channel, int total_size)
{
	int N = batch;
	int C = channel;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	int c_idx = tid % C;
	int n_idx = tid / C;

	float exp_element = expf(input[tid]);
	float exp_sum_n = exp_sum[n_idx];

	output[tid] = exp_element / exp_sum_n;


}

void softmax(float *output, float *input, float *exp_sum, int batch, int channel)
{
	int N = batch;
	int C = channel;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * C;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	softmax_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (output, input, exp_sum, N, C, TOTAL_SIZE);
}
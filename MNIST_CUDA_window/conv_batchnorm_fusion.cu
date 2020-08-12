#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

__global__ void conv_batchnorm_fusion_kernel(float *output, float *input, float *weight, float *bias,
	int batch, int in_channel, int out_channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width, int total_size)
{
	int N = batch;
	int C = in_channel;
	int K = out_channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	//tid : thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= total_size)
		return;

	//q_idx : output w-index
	int q_idx = tid % Q;
	int idx = tid / Q;

	//p_idx : output h-index
	int p_idx = idx % P;
	idx /= P;

	//k_idx : output channel-index
	int k_idx = idx % K;

	//n_idx : output batch-index
	int n_idx = idx / K;

	//output(n_idx, k_idx, p_idx, q_idx)

	float sum = 0.0f;
	for (int c_idx = 0; c_idx < C; c_idx++) {
		for (int kh_idx = 0; kh_idx < kH; kh_idx++) {
			int h_idx = p_idx * sH + kh_idx - pH;
			if (h_idx >= 0 && h_idx < H) {
				for (int kw_idx = 0; kw_idx < kW; kw_idx++) {
					int w_idx = q_idx * sW + kw_idx - pW;
					if (w_idx >= 0 && w_idx < W) {
						int input_index = n_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;
						int weight_index = k_idx * C * kH * kW + c_idx * kH * kW + kh_idx * kW + kw_idx;
						sum += input[input_index] * weight[weight_index];
					}
				}
			}
		}
	}
	sum += bias[k_idx];
	output[tid] = sum;

}


void conv_batchnorm_fusion(float *output, float *input, float *weight, float *bias,
	int batch, int in_channel, int out_channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width)
{
	int N = batch;
	int C = in_channel;
	int K = out_channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * K * P * Q;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	conv_batchnorm_fusion_kernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (output, input, weight, bias,
		N, C, K, H, W,
		kH, kW, pH, pW, sH, sW, TOTAL_SIZE);
}
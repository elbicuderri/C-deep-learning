#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>


__global__ void convolution_kernel(float *output, float *input, float *kernel,
	int batch, int in_channel, int out_channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width, int t_count)
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
	int P = (H + 2 * pH - kH) / sH + 1;
	int Q = (W + 2 * pW - kW) / sW + 1;

	// tid : thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (tid >= t_count)
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

	float sum = 0.0f;
	for (int c = 0; c < C; c++) {
		for (int kh = 0; kh < kH; kh++) {
			int h_idx = p_idx * sH + kh - pH; 
			if (h_idx >= 0 && h_idx < H) {
				for (int kw = 0; kw < kW; kw++) {
					int w_idx = q_idx * sW + kw - pW;
					if (w_idx >= 0 && w_idx < W) {
						int input_index = n_idx * C * H * W + c * H * W + h_idx * W + w_idx;
						int kernel_index = k_idx * C * kH * kW + c * kH * kW + kh * kW + kw;
						sum += input[input_index] * kernel[kernel_index];
					}
				}
			}
		}
	}
	output[tid] = sum;

}


void convolution(float *output, float *input, float *kernel,
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

	int P = (H + 2 * pH - kH ) / sH + 1;
	int Q = (W + 2 * pW - kW) / sW + 1;

	int BLOCKS = 256;
	int t_count = N * K * P * Q;
	int b_count = (t_count + BLOCKS - 1) / BLOCKS;

	convolution_kernel << <b_count, BLOCKS >> > (output, input, kernel,
		N, C, K, H, W, 
		kH, kW, pH, pW, sH, sW, t_count);
}
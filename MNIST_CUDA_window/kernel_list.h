#pragma once
void conv_batchnorm_fusion(float *output, float *input, float *weight, float *bias,
	int batch, int in_channel, int out_channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width);

void maxpooling(float *output, float *input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width);

void relu(float *output, float *input, int batch, int channel, int height, int width);

void dense(float *output, float *input, float *weight, float *bias, int N, int C, int K, int H, int W);

void softmax(float *output, float *input, float *exp_sum, int batch, int channel);
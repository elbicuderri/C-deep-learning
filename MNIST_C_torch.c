#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

void load_data(float *output, const char *name, int size);

void print_image(float *image, int batch, int height, int width, int channel);

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size, float constant);

void conv_batchnorm_fusion(float *output, float *input, float *kernel, float *gamma, float *beta, float *mean, float *variance, float epsilon,
	int batch, int in_channel, int out_channel, int input_height, int input_width,
	int kernel_height, int kernel_width,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	int stride_height, int stride_width);

void maxpooling(float *output, float *input, int batch, int channel, int input_height, int input_width, 
	int kernel_height, int kernel_width,
	int pad_top, int pad_bottom, int pad_left, int pad_right, int stride_height, int stride_width);

void dense(float *output, float *input, float *weight, float *bias, int batch, int in_channel, int out_channel, int input_height, int input_width);

void relu(float*output, float*input, int batch, int dim);

float exponential_sum(float *input, int length, int start);

void softmax(float *output, float *input, int batch, int dim);


int main()
{
	const int N = 1;
	const int image_size = 784;
	const int data_size = N * (image_size + 1);
	const float m = 255.0f;
	const int K = 5;
	const int C = 1;
	const int H = 28;
	const int W = 28;
	const int kH = 5;
	const int kW = 5;
	const int pH = 2;
	const int pW = 2;
	const int sH = 1;
	const int sW = 1;
	const int P = ((H + 2 * pH - kH) / sH) + 1; // output_height
	const int Q = ((W + 2 * pW - kW) / sW) + 1; // output_width
	const int maxpool_kH = 2;
	const int maxpool_kW = 2;
	const int maxpool_H = 14;
	const int maxpool_W = 14;
	const int maxpool_pH = 0;
	const int maxpool_pW = 0;
	const int maxpool_sH = 2;
	const int maxpool_sW = 2;
	const int dense1_units = 120;
	const int dense2_units = 10;
	const float epsilon = 0.001f;

	char *data_file = "mnist_test_float.bin";


	float *data = (float*)malloc(data_size * sizeof(float));
	load_data(data, data_file, data_size);

	//float *data = (float*)malloc(data_size * sizeof(float));
	//load_data(data, "mnist_test_float.bin", data_size);

	float *kernel = (float*)malloc(K * C * kH * kW * sizeof(float));
	load_data(kernel, "kernel_torch.bin", K * C * kH * kW);

	float *gamma = (float*)malloc(K * sizeof(float));
	load_data(gamma, "gamma_torch.bin", K);

	float *beta = (float*)malloc(K * sizeof(float));;
	load_data(beta, "beta_torch.bin", K);

	float *mean = (float*)malloc(K * sizeof(float));;
	load_data(mean, "mean_torch.bin", K);

	float *variance = (float*)malloc(K * sizeof(float));;
	load_data(variance, "variance_torch.bin", K);

	float *W1 = (float*)malloc(K * maxpool_H * maxpool_W * dense1_units * sizeof(float));;
	load_data(W1, "W1_torch.bin", K * maxpool_H * maxpool_W * dense1_units);

	float *b1 = (float*)malloc(dense1_units * sizeof(float));;
	load_data(b1, "b1_torch.bin", dense1_units);

	float *W2 = (float*)malloc(dense1_units * dense2_units * sizeof(float));;
	load_data(W2, "W2_torch.bin", dense1_units * dense2_units);

	float *b2 = (float*)malloc(dense2_units * sizeof(float));;
	load_data(b2, "b2_torch.bin", dense2_units);

	//====================================================================
	// original file to compare

	float *conv_fusion_origin = (float*)malloc(N * K * P * Q * sizeof(float));
	load_data(conv_fusion_origin, "batchnorm_torch.bin", N * K * H * W);

	float *maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(maxpool_origin, "maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *relu_maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(relu_maxpool_origin, "relu_maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));;
	load_data(dense1_origin, "dense1_torch.bin", N * dense1_units);

	float *relu_dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(relu_dense1_origin, "relu_dense1_torch.bin", N * dense1_units);

	float *dense2_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(dense2_origin, "dense2_torch.bin", N * dense2_units);

	float *result_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(result_origin, "result_torch.bin", N * dense2_units);

	//====================================================================

	//conv, maxpool, batchnorm, relu_batchnorm, dense1, relu_dense1, dense2, result

	//====================================================================

	float *image = (float*)malloc(N * image_size * sizeof(float));
	float *label = (float*)malloc(N * sizeof(float));
	split_image_label_normalization(image, label, data, N, image_size, m);

	float *conv_fusion = (float*)malloc(N * K * H * W * sizeof(float));
	conv_batchnorm_fusion(conv_fusion, image, kernel, gamma, beta, mean, variance, epsilon, N, C, K, H, W, kH, kW, pH, pH, pW, pW, sH, sW);

	float *maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	maxpooling(maxpool, conv_fusion, N, K, P, Q, maxpool_kH, maxpool_kW, maxpool_pH, maxpool_pH, maxpool_pW, maxpool_pW, maxpool_sH, maxpool_sW);

	float *relu_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	relu(relu_maxpool, maxpool, N, K * maxpool_H * maxpool_W);

	float *dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	dense(dense1, relu_maxpool, W1, b1, N, K, dense1_units, maxpool_H, maxpool_W);

	float *relu_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	relu(relu_dense1, dense1, N, dense1_units);

	float *dense2 = (float*)malloc(N * dense1_units * dense2_units * sizeof(float));
	dense(dense2, relu_dense1, W2, b2, N, dense1_units, dense2_units, 1, 1);

	float *result = (float*)malloc(N * dense2_units * sizeof(float));
	softmax(result, dense2, N, dense2_units);


	printf("before softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = dense2[index] - dense2_origin[index];
			printf("my answer: %.8f, real answer: %.8f, difference: %.8f\n\n", dense2[index], dense2_origin[index], diff);
		}
		printf("\n");
	}

	printf("===============================================================================================================================================================================================================================\n\n");

	printf("after softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = result[index] - result_origin[index];
			printf("my answer: %.8f, real answer: %.8f, difference: %.8f\n\n", result[index], result_origin[index], diff);
		}
		printf("\n\n");
	}

	free(data);
	free(kernel);
	free(gamma);
	free(beta);
	free(mean);
	free(variance);
	free(W1);
	free(b1);
	free(W2);
	free(b2);
	free(image);
	free(label);
	free(conv_fusion_origin);
	free(conv_fusion);
	free(maxpool_origin);
	free(maxpool);
	free(relu_maxpool_origin);
	free(relu_maxpool);
	free(dense1_origin);
	free(dense1);
	free(relu_dense1_origin);
	free(relu_dense1);
	free(dense2_origin);
	free(dense2);
	free(result_origin);
	free(result);

	return 0;

}

void load_data(float *output, const char *name, int size)
{
	FILE *pFile = fopen(name, "rb");

	if (pFile == NULL) {
		printf("cannot find %s\n", name);
		exit(-1);
	}

	size_t sizet = fread(output, size * sizeof(float), 1, pFile);

	if (sizet != 1) {
		printf("read error!\n");
		exit(-1);
	}

	fclose(pFile);
}

void print_image(float *image, int batch, int channel, int height, int width)
{
	int N = batch;
	int C = channel;
	int H = height;
	int W = width;

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					printf("%6.2f ", image[n * C * H * W + c * H * W + h * W + w]);
				}
				printf("\n\n");
			}
			printf("\n===============================================================================================\n");
		}
	}

}

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size, float constant)
{
	int N = total_number;
	int S = size;
	float m = constant;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < S; j++) {
			int index = i * S + j;
			image[index] = input[i * (S + 1) + (j + 1)] / m;
		}
	}

	for (int k = 0; k < N; k++) {
		label[k] = input[k * (S + 1)];
	}
}

void conv_batchnorm_fusion(float *output, float *input, float *kernel, float *gamma, float *beta, float *mean, float *variance, float epsilon,
	int batch, int in_channel, int out_channel, int input_height, int input_width,
	int kernel_height, int kernel_width,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	int stride_height, int stride_width)
{
	int N = batch;
	int C = in_channel;
	int K = out_channel;
	int H = input_height;
	int W = input_width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pT = pad_top;
	int pB = pad_bottom;
	int pL = pad_left;
	int pR = pad_right;
	int sH = stride_height;
	int sW = stride_width;
	int pH = H + pT + pB;
	int pW = W + pL + pR;
	int P = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1; // output_height
	int Q = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1; // output_width

	//set weight
	float *weight = (float*)malloc(K * C * kH * kW * sizeof(float));

	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int kh = 0; kh < kH; kh++) {
				for (int kw = 0; kw < kW; kw++) {
					int index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
					weight[index] = (gamma[k] * kernel[index]) / (sqrtf(variance[k] + epsilon));
				}
			}
		}
	}

	//set bias
	float *bias = (float*)malloc(K * sizeof(float));

	for (int k = 0; k < K; k++) {
		bias[k] = beta[k] - ((gamma[k] * mean[k]) / (sqrtf(variance[k] + epsilon)));
	}

	//convolution + batchnormalization
	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			for (int p = 0; p < P; p++) { //image_row
				for (int q = 0; q < Q; q++) { //image_column
					float sum = 0.0f;
					for (int c = 0; c < C; c++) {
						for (int kh = 0; kh < kH; kh++) {//kernel_height
							int input_h_index = p * sH + kh - pT;
							if (input_h_index >= 0 && input_h_index < H) {
								for (int kw = 0; kw < kW; kw++) { //kernel_width
									int input_w_index = q * sW + kw - pL;
									if (input_w_index >= 0 && input_w_index < W) {
										int input_index = n * C * H * W + c * H * W + input_h_index * W + input_w_index;
										int weight_index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
										float s = weight[weight_index] * input[input_index];
										sum += s;
									}
								}
							}
						}
					}
					int output_index = n * K * P * Q + k * P * Q + p * Q + q;
					sum += bias[k];
					output[output_index] = sum;
				}
			}
		}
	}

	free(weight);
	free(bias);

}


void maxpooling(float *output, float *input, int batch, int channel, int input_height, int input_width, int kernel_height, int kernel_width, int pad_top, int pad_bottom, int pad_left, int pad_right, int stride_height, int stride_width)
{
	int N = batch;
	int C = channel;
	int H = input_height;
	int W = input_width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pT = pad_top;
	int pB = pad_bottom;
	int pL = pad_left;
	int pR = pad_right;
	int sH = stride_height;
	int sW = stride_width;
	int P = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1;
	int Q = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1;

	//maxpooling
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int p = 0; p < P; p++) {
				for (int q = 0; q < Q; q++) {
					float max = -FLT_MAX;
					for (int kh = 0; kh < kH; kh++) {
						int h_idx = p * sH + kh - pT;
						if (h_idx >= 0 && h_idx < H) {
							for (int kw = 0; kw < kW; kw++) {
								int w_idx = q * sW + kw - pL;
								if (w_idx >= 0 && w_idx < W) {
									int index = n * C * H * W + c * H * W + h_idx * W + w_idx;
									if (input[index] > max) {
										max = input[index];
									}
								}
							}
						}
					}
					int output_index = n * C * P * Q + c * P * Q + p * Q + q;
					output[output_index] = max;
				}
			}
		}
	}

}

void dense(float *output, float *input, float *weight, float *bias, int batch, int in_channel, int out_channel, int input_height, int input_width)
{
	int N = batch;
	int H = input_height;
	int W = input_width;
	int C = in_channel;
	int K = out_channel;
	int L = C * H * W;

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = 0.0f;
			for (int i = 0; i < L; i++) {
				int input_index = n * L + i;
				int weight_index = k * L + i;
				float s = input[input_index] * weight[weight_index];
				sum += s;
			}
			sum += bias[k];
			int output_index = n * K + k;
			output[output_index] = sum;

		}
	}

}

void relu(float *output, float *input, int batch, int dim)
{
	int N = batch;
	int D = dim;

	for (int n = 0; n < N; n++) {
		for (int i = 0; i < D; i++) {
			int index = n * D + i;
			if (input[index] > 0.0f) {
				output[index] = input[index];
			}
			else {
				output[index] = 0.0f;
			}
		}
	}

}


float exponential_sum(float *input, int length, int start)
{
	int L = length;
	int S = start;

	float sum = 0.0f;
	for (int i = 0; i < L; i++) {
		float element = input[i + S];
		float element_exponential = expf(element);
		sum += element_exponential;
	}

	return sum;

}

void softmax(float *output, float *input, int batch, int length)
{
	int N = batch;
	int L = length;

	for (int i = 0; i < N; i++) {
		float exp_sum = exponential_sum(input, L, i * L);
		for (int j = 0; j < L; j++) {
			int index = i * L + j;
			output[index] = expf(input[index]) / exp_sum;
		}
	}

}

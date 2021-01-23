#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

void load_data(float *output, const char *name, int size);

void print_image(float *image, int batch, int height, int width, int channel);

void split_image_label(float *image, float *label, float *input, int total_number, int size);

void normalization(float *output, float *input, int size);

void convolution(float*output, float*image, float*kernel, int batch, int in_channel, int out_channel, int image_Height, int image_Width,
		 int kernel_Height, int kernel_Width, int pad_top, int pad_bottom, int pad_left, int pad_right, int stride_height, int stride_width);

void maxpooling(float*output, float*input, int batch, int channel, int input_height, int input_width,
		int kernel_height, int kernel_width, int pad_top, int pad_bottom, int pad_left, int pad_right, int stride_height, int stride_width);

void dense(float *output, float *input, float *weight, float *bias, int batch, int input_height, int input_width, int channel, int output_dim);

void relu(float*output, float*input, int batch, int dim);

void batchnormalization(float *output, float *input, float *gamma, float *beta, float *mean, float *variance, float epsilon,
			int batch, int input_height, int input_width, int channel);

float exponential_sum(float *input, int length, int start);

void softmax(float *output, float *input, int batch, int dim);

void transpose(float *output, float *input, int batch, int height, int width, int channel);

void conv_batchnorm(float *output, float *input, float *kernel, float *gamma, float *beta, float *mean, float *variance, float epsilon,
	int batch, int in_channel, int input_height, int input_width, int out_channel,
	int kernel_height, int kernel_width,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	int stride_height, int stride_width);


int main()
{
	const int N = 1;
	const int image_size = 784;
	const int data_size = N * (image_size + 1);
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


	float *data = (float*)malloc(data_size * sizeof(float));
	load_data(data, "data/mnist_test_float.bin", data_size);

	float *kernel = (float*)malloc(kH * kW * C * K * sizeof(float));
	load_data(kernel, "data/kernel_tfone.bin", kH * kW * C * K);

	//printf("kernel: \n");
	//print_image(kernel, C, K, kH, kW);

	float *gamma = (float*)malloc(K * sizeof(float));
	load_data(gamma, "data/gamma_tfone.bin", K);

	//printf("gamma: \n");
	//print_image(gamma, 1, 1, 1, K);

	float *beta = (float*)malloc(K * sizeof(float));;
	load_data(beta, "data/beta_tfone.bin", K);

	//printf("beta: \n");
	//print_image(beta, 1, 1, 1, K);

	float *mean = (float*)malloc(K * sizeof(float));;
	load_data(mean, "data/mean_tfone.bin", K);

	//printf("mean: \n");
	//print_image(mean, 1, 1, 1, K);

	float *variance = (float*)malloc(K * sizeof(float));;
	load_data(variance, "data/variance_tfone.bin", K);

	//printf("variance: \n");
	//print_image(variance, 1, 1, 1, K);

	float *W1 = (float*)malloc(maxpool_H * maxpool_W * K * dense1_units * sizeof(float));;
	load_data(W1, "data/W1_tfone.bin", maxpool_H * maxpool_W * K * dense1_units);

	float *b1 = (float*)malloc(dense1_units * sizeof(float));;
	load_data(b1, "data/b1_tfone.bin", dense1_units);

	float *W2 = (float*)malloc(dense1_units * dense2_units * sizeof(float));;
	load_data(W2, "data/W2_tfone.bin", dense1_units * dense2_units);

	float *b2 = (float*)malloc(dense2_units * sizeof(float));;
	load_data(b2, "data/b2_tfone.bin", dense2_units);

	//printf("b2: \n");
	//print_image(b2, 1, 1, 1, dense2_units);

	//====================================================================

	//conv, maxpool, batchnorm, relu_batchnorm, dense1, relu_dense1, dense2, result

	//====================================================================

	float *image = (float*)malloc(N *  image_size * sizeof(float));

	float *label = (float*)malloc(N * sizeof(float));

	split_image_label(image, label, data, N, image_size);

	//print_image(label, N, 1, 1, 1);

	float *image_norm = (float*)malloc(N *  image_size * sizeof(float));

	normalization(image_norm, image, N *  image_size);

	float *image_transpose = (float*)malloc(N *  image_size * sizeof(float));

	transpose(image_transpose, image_norm, N, H, W, C);

	//printf("transpose_image: \n");
	//print_image(image_transpose, N, H, W, C);

	float *conv_origin = (float*)malloc(N * H * W * K * sizeof(float));
	load_data(conv_origin, "data/conv_tfone.bin", N * H * W * C);

	float *conv = (float*)malloc(N * H * W * K * sizeof(float));
	convolution(conv, image_transpose, kernel, N, C, K, H, W, kH, kW, pH, pH, pW, pW, sH, sW);

	//printf("conv_origin: \n");
	//print_image(conv_origin, n, 28, 28, 5);

	//printf("conv : \n");
	//print_image(conv, n, 28, 28, 5);

	float *batchnorm_origin = (float*)malloc(N * H * W * K * sizeof(float));
	load_data(batchnorm_origin, "data/batchnorm_tfone.bin", N * H * W * K);

	float *batchnorm = (float*)malloc(N * H * W * K * sizeof(float));
	batchnormalization(batchnorm, conv, gamma, beta, mean, variance, epsilon, N, H, W, K);

	float *conv_batchnorm_fusion = (float*)malloc(N * H * W * K * sizeof(float));
	conv_batchnorm(conv_batchnorm_fusion, image_transpose, kernel, gamma, beta, mean, variance, epsilon, N, C, H, W, K, kH, kW, pH, pH, pW, pW, sH, sW);

	//printf("batchnorm_origin: \n");
	//print_image(batchnorm_origin, N, H, W, K);

	//printf("batchnorm: \n");
	//print_image(batchnorm, N, H, W, K);

	//printf("conv_batchnorm_fusion: \n");
	//print_image(conv_batchnorm_fusion, N, H, W, K);

	float *maxpool_origin = (float*)malloc(N * maxpool_H * maxpool_W * K * sizeof(float));
	load_data(maxpool_origin, "data/maxpool_tfone.bin", N * maxpool_H * maxpool_W * K);

	float *maxpool = (float*)malloc(N * maxpool_H * maxpool_W * K * sizeof(float));
	maxpooling(maxpool, conv_batchnorm_fusion, N, K, H, W, maxpool_kH, maxpool_kW, maxpool_pH, maxpool_pH, maxpool_pW, maxpool_pW, maxpool_sH, maxpool_sW);

	//float *maxpool = (float*)malloc(N * maxpool_H * maxpool_W * K * sizeof(float));
	//maxpooling(maxpool, batchnorm, N, K, H, W, maxpool_kH, maxpool_kW, maxpool_pH, maxpool_pH, maxpool_pW, maxpool_pW, maxpool_sH, maxpool_sW);

	//printf("maxpool_origin: \n");
	//print_image(maxpool_origin, n, 14, 14, 5);

	//printf("maxpool: \n");
	//print_image(maxpool, n, 14, 14, 5);

	float *relu_maxpool_origin = (float*)malloc(N * maxpool_H * maxpool_W * K * sizeof(float));
	load_data(relu_maxpool_origin, "data/relu_maxpool_tfone.bin", N * maxpool_H * maxpool_W * K);

	float *relu_maxpool = (float*)malloc(N * maxpool_H * maxpool_W * K * sizeof(float));
	relu(relu_maxpool, maxpool, N, maxpool_H * maxpool_H  * K);

	//printf("relu_maxpool_origin: \n");
	//print_image(relu_maxpool_origin, n, 14, 14, 5);

	//printf("relu_maxpool: \n");
	//print_image(relu_maxpool, n, 14, 14, 5);

	float *dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));;
	load_data(dense1_origin, "data/dense1_tfone.bin", N * dense1_units);

	float *dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	dense(dense1, relu_maxpool, W1, b1, N, maxpool_H, maxpool_W, K, dense1_units);

	//printf("dense1_origin: \n");
	//print_image(dense1_origin, n, 12, 10, 1);

	//printf("dense1: \n");
	//print_image(dense1, n, 12, 10, 1);

	float *relu_dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(relu_dense1_origin, "data/relu_dense1_tfone.bin", N * dense1_units);

	float *relu_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	relu(relu_dense1, dense1, N, dense1_units);

	float *dense2_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(dense2_origin, "data/dense2_tfone.bin", N * dense2_units);

	float *dense2 = (float*)malloc(N * dense1_units * dense2_units * sizeof(float));
	dense(dense2, relu_dense1, W2, b2, N, 1, dense1_units, 1, dense2_units);

	//printf("before softmax origin: \n");
	//print_image(dense2_origin, n, 1, dense2_units, 1);

	//printf("before softmax : \n");
	//print_image(dense2, n, 1, dense2_units, 1);
	
	float *result_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(result_origin, "data/result_tfone.bin", N * dense2_units);

	float *result = (float*)malloc(N * dense2_units * sizeof(float));
	softmax(result, dense2, N, dense2_units);

	//printf("after softmax origin : \n");
	//print_image(result_origin, N, 1, dense2_units, 1);

	//printf("after softmax : \n");
	//print_image(result, N, 1, dense2_units, 1);

	printf("before softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i+1);
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
	free(image_norm);
	free(image_transpose);
	free(conv_origin);
	free(conv);
	free(batchnorm_origin);
	free(batchnorm);
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

void print_image(float *image, int batch, int height, int width, int channel)
{
	int N = batch;
	int H = height;
	int W = width;
	int C = channel;

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++){
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
				printf("%6.2f ", image[n * H * W * C + h * W * C + w * C + c]);
				}
				printf("\n\n");
			}
			printf("\n===============================================================================================================================================================================================================================\n");
		}
	}

}

void split_image_label(float *image, float *label, float *input, int batch, int size)
{
	int N = batch;
	int S = size;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < S; j++) {
			int index = i * S + j;
			image[index] = input[i * (S + 1) + (j + 1)];
		}
	}

	for (int i = 0; i < N; i++) {
		label[i] = input[i * (S + 1)];
	}
}

void normalization(float *output, float *input, int size)
{
	float m = 255.0f;
	for (int i = 0; i < size; i++) {
		output[i] = input[i] / m;
	}
}

void transpose(float *output, float *input, int batch, int height, int width, int channel)
{
	int N = batch;
	int H = height;
	int W = width;
	int C = channel;

	for (int n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				for (int c = 0; c < C; c++) {
					int index_NHWC = n * H * W * C + h * W * C + w * C + c;
					int index_NCHW = n * C * H * W + c * H * W + h * W + w;
					output[index_NHWC] = input[index_NCHW];
				}
			}
		}
	}

}

void convolution(float *output, float *input, float *kernel, int batch, int in_channel, int out_channel, int input_height, int input_width, int kernel_height, int kernel_width, int pad_top, int pad_bottom, int pad_left, int pad_right, int stride_height, int stride_width)
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
	int oH = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1; // output_height
	int oW = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1; // output_width

	////pad input
	float *pad_input;
	pad_input = (float*)malloc(N * pH * pW * C * sizeof(float));

	//fill pad input
	for (int n = 0; n < N; n++) {
		for (int ph = 0; ph < pH; ph++) {
			for (int pw = 0; pw < pW; pw++) {
				for (int c = 0; c < C; c++) {
					int pad_index = n * pH * pW * C + ph * pW * C + pw * C + c;
					if (ph < pT || ph >= H + pT || pw < pL || pw >= W + pL) {
						pad_input[pad_index] = 0.f;
					}
					else {
						int input_index = n * H * W * C + (ph - pT) * W * C + (pw - pL) * C + c;
						pad_input[pad_index] = input[input_index];
					}
				}
			}
		}
	}

	//printf("pad_image during convolution: \n");
	//print_image(pad_input, N, pH, pW, C);

	//convolution
	for (int n = 0; n < N; n++) {
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				for (int k = 0; k < K; k++) {
					float sum = 0;
					for (int kh = 0; kh < kH; kh++) {
						for (int kw = 0; kw < kW; kw++) {
							for (int c = 0; c < C; c++) {
								int kernel_index = kh * kW * C * K + kw * C * K + c * K + k;
								int pad_index = n * pH * pW * C + (oh * sH + kh) * pW * C + (ow * sW + kw)* C + c;
								float in = pad_input[pad_index];
								float w = kernel[kernel_index];
								float s = in * w;
								sum += s;
							}
						}
					}
					int output_index = n * oH * oW * K + oh * oW * K + ow * K + k;
					output[output_index] = sum;
				}
			}
		}
	}

	free(pad_input);

	//printf("output after convolution: \n");
	//print_image(output, N, oH, oW, K);
	//printf("convolution finished.\n");

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
	int pH = H + pT + pB;
	int pW = W + pL + pR;
	int oH = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1;
	int oW = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1;

	float *pad_input;
	pad_input = (float*)malloc((N * pH * pW * C) * sizeof(float));

	//fill pad image
	for (int n = 0; n < N; n++) {
		for (int ph = 0; ph< pH; ph++) {
			for (int pw = 0; pw < pW; pw++) {
				for (int c = 0; c < C; c++) {
					int pad_index = n * pH * pW * C + ph * pW * C + pw * C + c;
					if (ph < pT || ph >= H + pT || pw < pL || pw >= W + pL) {
						pad_input[pad_index] = 0.f;
					}
					else {
						int input_index = n * H * W * C + (ph - pT) * pW * C + (pw - pL) * C + c;
						pad_input[pad_index] = input[input_index];
					}
				}
			}
		}
	}

	//printf("pad_image during maxpooling: \n");
	//print_image(pad_input, N , pH,  pW, C);

	//maxpooling
	for (int n = 0; n < N; n++) {
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				for (int c= 0; c < C; c++) {
					float max = - FLT_MAX;
					for (int kh = 0; kh < kH; kh++) {
						for (int kw = 0; kw < kW; kw++) {
							int pad_index = n * pH * pW * C + (oh * sH + kh) * pW * C + (ow * sW + kw) * C + c;
							if (pad_input[pad_index] > max) {
								max = pad_input[pad_index];
							}
						}
					}
					int output_index = n * oH * oW * C +  oh * oW * C + ow * C + c;
					output[output_index] = max;
				}
			}
		}
	}

	free(pad_input);

	//printf("output after maxpooling: \n");
	//print_image(output, N * C * oH, oW);
}

void dense(float *output, float *input, float *weight, float *bias, int batch, int input_height, int input_width, int channel, int output_dim)
{
	int N = batch;
	int H = input_height;
	int W = input_width;
	int C = channel;
	int K = output_dim; 

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = 0;
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					for (int c = 0; c < C; c++) {
						int input_index = n * H * W * C + h * W * C + w * C + c;
						int weight_index = h * W * C * K + w * C * K + c * K + k;
						float s = input[input_index] * weight[weight_index];
						sum += s;
					}
				}
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
			if (input[index] > 0.f) {
				output[index] = input[index];
			}
			else {
				output[index] = 0.f;
			}
		}
	}

}

void batchnormalization(float *output, float *input, float *gamma, float *beta, float *mean, float *variance, float epsilon, int input_height, int input_width, int batch, int channel)
{
	int N = batch;
	int H = input_height;
	int W = input_width;
	int C = channel;

	for (int n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				for (int c = 0; c < C; c++){
				float result = ((gamma[c] * (input[n * H * W * C + h * W * C + w * C + c] - mean[c]))/ (sqrtf(variance[c] + epsilon))) + beta[c];
				output[n * H * W * C + h * W * C + w * C + c] = result;
				}
			}
		}
	}

}

float exponential_sum(float *input, int length, int start)
{
	int L = length;
	int S = start;

	float sum = 0.f;
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


void conv_batchnorm(float *output, float *input, float *kernel, float *gamma, float *beta, float *mean, float *variance, float epsilon,
	int batch, int in_channel, int input_height, int input_width, int out_channel,
	int kernel_height, int kernel_width,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	int stride_height, int stride_width
)
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
	int oH = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1; // output_height
	int oW = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1; // output_width

	////pad input
	float *pad_input;
	pad_input = (float*)malloc((N * pH * pW * C) * sizeof(float));

	//fill pad input
	for (int n = 0; n < N; n++) {
		for (int ph = 0; ph < pH; ph++) {
			for (int pw = 0; pw < pW; pw++) {
				for (int c = 0; c < C; c++) {
					int pad_index = n * pH * pW * C + ph * pW * C + pw * C + c;
					if (ph < pT || ph >= H + pT || pw < pL || pw >= W + pL) {
						pad_input[pad_index] = 0.f;
					}
					else {
						int input_index = n * H * W * C + (ph - pT) * W * C + (pw - pL) * C + c;
						pad_input[pad_index] = input[input_index];
					}
				}
			}
		}
	}

	//set weight
	float *weight = (float*)malloc(K * C * kH * kW * sizeof(float));

	for (int kh = 0; kh < kH; kh++) {
		for (int kw = 0; kw < kW; kw++) {
			for (int c = 0; c < C; c++) {
				for (int k = 0; k < K; k++) {
					int index = kh * kW * C * K + kw * C * K + c * K + k;
					weight[index] = (gamma[k] * kernel[index]) / (sqrtf(variance[k] + epsilon));
				}
			}
		}
	}

	//printf("weight: \n");
	//print_image(weight, K, kH, kW, C);

	//set bias
	float *bias = (float*)malloc(K * sizeof(float));

	for (int k = 0; k < K; k++) {
		bias[k] = beta[k] - ((gamma[k] * mean[k]) / (sqrtf(variance[k] + epsilon)));
	}

	//printf("bias: \n");
	//print_image(bias, 1, 1, K, 1);

	//printf("pad_image during convolution: \n");
	//print_image(pad_input, N, pH, pW, C);

	//convolution
	for (int n = 0; n < N; n++) {
		for (int oh = 0; oh < oH; oh++) {
			for (int ow = 0; ow < oW; ow++) {
				for (int k = 0; k < K; k++) {
					float sum = 0.f;
					for (int kh = 0; kh < kH; kh++) {
						for (int kw = 0; kw < kW; kw++) {
							for (int c = 0; c < C; c++) {
								int weight_index = kh * kW * C * K + kw * C * K + c * K + k;
								int pad_index = n * pH * pW * C + (oh * sH + kh) * pW * C + (ow * sW + kw)* C + c;
								float s = weight[weight_index] * pad_input[pad_index];
								sum += s;
							}
						}
					}
					int output_index = n * oH * oW * K + oh * oW * K + ow * K + k;
					sum += bias[k];
					output[output_index] = sum;
				}
			}
		}
	}

	free(pad_input);
	free(weight);
	free(bias);

}

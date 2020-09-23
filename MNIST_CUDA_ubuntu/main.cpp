#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>

void conv_batchnorm_fusion(float *output, float *input, float *weight, float *bias,
	int batch, int in_channel, int out_channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width);

void maxpooling(float *output, float *input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width);

void relu(float *output, float *input, int batch, int channel, int height, int width);

void dense(float *output, float *input, float *weight, float *bias, int N, int C, int K, int H, int W);

void softmax(float *output, float *input, float *exp_sum, int batch, int channel);

void load_data(float *output, const char *name, int size);

void print_image(float *image, int batch, int height, int width, int channel);

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size);

void exponential_sum(float *output, float *input, int batch, int channel);

int main()
{
	const int N = 10;
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
	const int P = ((H + 2 * pH - kH) / sH) + 1;
	const int Q = ((W + 2 * pW - kW) / sW) + 1;
	const float epsilon = 0.001f;
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

	//Data and Weight
	float *data = (float*)malloc(data_size * sizeof(float));
	load_data(data, "data/mnist_test_float.bin", data_size);

	float *kernel = (float*)malloc(K * C * kH * kW * sizeof(float));
	load_data(kernel, "data/kernel_torch.bin", K * C * kH * kW);

	float *gamma = (float*)malloc(K * sizeof(float));
	load_data(gamma, "data/gamma_torch.bin", K);

	float *beta = (float*)malloc(K * sizeof(float));;
	load_data(beta, "data/beta_torch.bin", K);

	float *mean = (float*)malloc(K * sizeof(float));;
	load_data(mean, "data/mean_torch.bin", K);

	float *variance = (float*)malloc(K * sizeof(float));;
	load_data(variance, "data/variance_torch.bin", K);

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

	float *bias = (float*)malloc(K * sizeof(float));
	for (int k = 0; k < K; k++) {
		bias[k] = beta[k] - ((gamma[k] * mean[k]) / (sqrtf(variance[k] + epsilon)));
	}

	float *W1 = (float*)malloc(K * maxpool_H * maxpool_W * dense1_units * sizeof(float));;
	load_data(W1, "data/W1_torch.bin", K * maxpool_H * maxpool_W * dense1_units);

	float *b1 = (float*)malloc(dense1_units * sizeof(float));;
	load_data(b1, "data/b1_torch.bin", dense1_units);

	float *W2 = (float*)malloc(dense1_units * dense2_units * sizeof(float));;
	load_data(W2, "data/W2_torch.bin", dense1_units * dense2_units);

	float *b2 = (float*)malloc(dense2_units * sizeof(float));;
	load_data(b2, "data/b2_torch.bin", dense2_units);

	float *image = (float*)malloc(N * image_size * sizeof(float));
	float *label = (float*)malloc(N * sizeof(float));
	split_image_label_normalization(image, label, data, N, image_size);

	//origin value to compare
	float *conv_fusion_origin = (float*)malloc(N * K * P * Q * sizeof(float));
	load_data(conv_fusion_origin, "data/batchnorm_torch.bin", N * K * P * Q);

	float *maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(maxpool_origin, "data/maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *relu_maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(relu_maxpool_origin, "data/relu_maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(dense1_origin, "data/dense1_torch.bin", N * dense1_units);

	float *relu_dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(relu_dense1_origin, "data/relu_dense1_torch.bin", N * dense1_units);

	float *dense2_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(dense2_origin, "data/dense2_torch.bin", N * dense2_units);

	float *logit_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(logit_origin, "data/result_torch.bin", N * dense2_units);

	//copy data to the device
	float *d_image;
	cudaMalloc((void**)&d_image, N * C * H * W * sizeof(float));
	cudaMemcpy(d_image, image, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

	float *d_kernel;
	cudaMalloc((void**)&d_kernel, K * C * kH * kW * sizeof(float));
	cudaMemcpy(d_kernel, kernel, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

	float *d_weight;
	cudaMalloc((void**)&d_weight, K * C * kH * kW * sizeof(float));
	cudaMemcpy(d_weight, weight, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

	float *d_bias;
	cudaMalloc((void**)&d_bias, K * sizeof(float));
	cudaMemcpy(d_bias, bias, K * sizeof(float), cudaMemcpyHostToDevice);

	float *d_W1;
	cudaMalloc((void**)&d_W1, K * maxpool_H * maxpool_W * dense1_units * sizeof(float));
	cudaMemcpy(d_W1, W1, K * maxpool_H * maxpool_W * dense1_units * sizeof(float), cudaMemcpyHostToDevice);

	float *d_b1;
	cudaMalloc((void**)&d_b1, dense1_units * sizeof(float));
	cudaMemcpy(d_b1, b1, dense1_units * sizeof(float), cudaMemcpyHostToDevice);

	float *d_W2;
	cudaMalloc((void**)&d_W2, dense1_units * dense2_units * sizeof(float));
	cudaMemcpy(d_W2, W2, dense1_units * dense2_units * sizeof(float), cudaMemcpyHostToDevice);

	float *d_b2;
	cudaMalloc((void**)&d_b2, dense2_units * sizeof(float));
	cudaMemcpy(d_b2, b2, dense2_units * sizeof(float), cudaMemcpyHostToDevice);

	//allocate to the device
	float *d_conv_fusion;
	cudaMalloc((void**)&d_conv_fusion, N * K * P * Q * sizeof(float));

	float *d_maxpool;
	cudaMalloc((void**)&d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float));

	float *d_relu_maxpool;
	cudaMalloc((void**)&d_relu_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float));

	float *d_dense1;
	cudaMalloc((void**)&d_dense1, N * dense1_units * sizeof(float));

	float *d_relu_dense1;
	cudaMalloc((void**)&d_relu_dense1, N * dense1_units * sizeof(float));

	float *d_dense2;
	cudaMalloc((void**)&d_dense2, N * dense2_units * sizeof(float));

	//CNN
	conv_batchnorm_fusion(d_conv_fusion, d_image, d_weight, d_bias, N, C, K, H, W, kH, kW, pH, pW, sH, sW);

	maxpooling(d_maxpool, d_conv_fusion, N, K, P, Q, maxpool_kH, maxpool_kW, maxpool_pH, maxpool_pW, maxpool_sH, maxpool_sW);

	relu(d_relu_maxpool, d_maxpool, N, K, maxpool_H, maxpool_W);

	dense(d_dense1, d_relu_maxpool, d_W1, d_b1, N, K, dense1_units, maxpool_H, maxpool_W);

	relu(d_relu_dense1, d_dense1, N, dense1_units, 1, 1);

	dense(d_dense2, d_relu_dense1, d_W2, d_b2, N, dense1_units, dense2_units, 1, 1);

	//copy data to the host
	float *conv_fusion = (float*)malloc(N * K * P * Q * sizeof(float));
	cudaMemcpy(conv_fusion, d_conv_fusion, N * K * P * Q * sizeof(float), cudaMemcpyDeviceToHost);

	float *maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	cudaMemcpy(maxpool, d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float), cudaMemcpyDeviceToHost);

	float *relu_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	cudaMemcpy(relu_maxpool, d_relu_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float), cudaMemcpyDeviceToHost);

	float *dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	cudaMemcpy(dense1, d_dense1, N * dense1_units, cudaMemcpyDeviceToHost);

	float *relu_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	cudaMemcpy(relu_dense1, d_relu_dense1, N * dense1_units * sizeof(float), cudaMemcpyDeviceToHost);

	float *dense2 = (float*)malloc(N * dense2_units * sizeof(float));
	cudaMemcpy(dense2, d_dense2, N * dense2_units * sizeof(float), cudaMemcpyDeviceToHost);

	//softmax
	float *exp_sum = (float*)malloc(N * sizeof(float));
	exponential_sum(exp_sum, dense2, N, dense2_units);

	float *d_exp_sum;
	cudaMalloc((void**)&d_exp_sum, N * sizeof(float));
	cudaMemcpy(d_exp_sum, exp_sum, N * sizeof(float), cudaMemcpyHostToDevice);

	float *d_logit;
	cudaMalloc((void**)&d_logit, N * dense2_units * sizeof(float));

	softmax(d_logit, d_dense2, d_exp_sum, N, dense2_units);

	float *logit = (float*)malloc(N * dense2_units * sizeof(float));
	cudaMemcpy(logit, d_logit, N * dense2_units * sizeof(float), cudaMemcpyDeviceToHost);

	//float *conv = (float*)malloc(N * K * P * Q * sizeof(float));
	//cudaMemcpy(conv, d_conv, N * K * P * Q * sizeof(float), cudaMemcpyDeviceToHost);

	//printf("conv_origin: \n\n");
	//print_image(conv_origin, N, K, P, Q);

	//printf("conv : \n\n");
	//print_image(conv, N, K, P, Q);

	//printf("conv_fusion_origin: \n\n");
	//print_image(batchnorm_origin, N, K, P, Q);

	//printf("conv_fusion : \n\n");
	//print_image(conv_fusion, N, K, P, Q);

	//printf("maxpool_origin: \n\n");
	//print_image(maxpool_origin, N, K, maxpool_H, maxpool_W);

	//printf("maxpool : \n\n");
	//print_image(maxpool, N, K, maxpool_H, maxpool_W);

	//printf("relu_maxpool_origin: \n\n");
	//print_image(relu_maxpool_origin, N, K, maxpool_H, maxpool_W);

	//printf("relu_maxpool : \n\n");
	//print_image(relu_maxpool, N, K, maxpool_H, maxpool_W);

	//printf("dense1_origin: \n\n");
	//print_image(dense1_origin, N, 1, 1, 30);

	//printf("dense1 : \n\n");
	//print_image(dense1, N, 1, 1, 30);

	//printf("relu_dense1_origin: \n\n");
	//print_image(relu_dense1_origin, N, 1, 1, 30);

	//printf("relu_dense1 : \n\n");
	//print_image(relu_dense1, N, 1, 1, 30);

	//printf("dense2_origin: \n\n");
	//print_image(dense2_origin, N, 1, 1, 10);

	//printf("dense2 : \n\n");
	//print_image(dense2, N, 1, 1, 10);

	//printf("logit_origin: \n\n");
	//print_image(logit_origin, N, 1, 1, dense2_units);

	//printf("logit: \n\n");
	//print_image(logit, N, 1, 1, dense2_units);

	//for (int i = 0; i < 10; i++) {
	//	//printf("%f\n", dense2[i]);
	//	cout << i << "th value: " << dense2[i] << endl;
	//}

	printf("before softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = dense2[index] - dense2_origin[index];
			printf("my answer: %.6f, real answer: %.6f, difference: %.6f\n\n", dense2[index], dense2_origin[index], diff);
		}
		printf("\n");
	}

	printf("===============================================================================================================\n\n");

	printf("after softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = logit[index] - logit_origin[index];
			printf("my answer: %.6f, real answer: %.6f, difference: %.6f\n\n", logit[index], logit_origin[index], diff);
		}
		printf("\n\n");
	}

	cudaFree(d_image);
	cudaFree(d_kernel);
	cudaFree(d_weight);
	cudaFree(d_bias);
	cudaFree(d_W1);
	cudaFree(d_b1);
	cudaFree(d_W2);
	cudaFree(d_b2);
	cudaFree(d_conv_fusion);
	cudaFree(d_maxpool);
	cudaFree(d_relu_maxpool);
	cudaFree(d_dense1);
	cudaFree(d_relu_dense1);
	cudaFree(d_dense2);
	cudaFree(d_exp_sum);
	cudaFree(d_logit);

	free(data);
	free(kernel);
	free(image);
	free(label);
	free(gamma);
	free(beta);
	free(mean);
	free(variance);
	free(weight);
	free(bias);
	free(W1);
	free(b1);
	free(W2);
	free(b2);
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
	free(logit_origin);
	free(logit);
	free(exp_sum);

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
		printf("%dth batch: \n\n", n + 1);
		for (int c = 0; c < C; c++) {
			printf("%dth channel: \n\n", c + 1);
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					printf("%.5f ", image[n * C * H * W + c * H * W + h * W + w]);
				}
				printf("\n\n");
			}
			printf("\n===========================================================================\n");
		}
	}

}

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size)
{
	int N = total_number;
	int S = size;
	float m = 255.0f;

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

void exponential_sum(float *output, float *input, int batch, int channel)
{
	int N = batch;
	int C = channel;

	for (int n = 0; n < N; n++) {
		float sum = 0.0f;
		for (int c = 0; c < C; c++) {
			float exp_element = expf(input[n * C + c]);
			sum += exp_element;
		}
		output[n] = sum;
	}

}


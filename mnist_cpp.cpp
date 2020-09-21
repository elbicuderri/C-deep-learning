#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "cnn_functions.h"

void load_data(float *output, const char *name, int size);

void print_image(float *image, int batch, int height, int width, int channel);

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size, float constant);

int main()
{
	clock_t start = clock();
	const int N = 3;
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

	const char *data_file = "mnist_test_float.bin";

	float* data = new float[data_size];
	load_data(data, data_file, data_size);

	float *kernel = (float*)malloc(K * C * kH * kW * sizeof(float));
	load_data(kernel, "kernel_pytorch.bin", K * C * kH * kW);

	float *gamma = (float*)malloc(K * sizeof(float));
	load_data(gamma, "gamma_pytorch.bin", K);

	float *beta = (float*)malloc(K * sizeof(float));;
	load_data(beta, "beta_pytorch.bin", K);

	float *mean = (float*)malloc(K * sizeof(float));;
	load_data(mean, "mean_pytorch.bin", K);

	float *variance = (float*)malloc(K * sizeof(float));;
	load_data(variance, "variance_pytorch.bin", K);

	float *W1 = (float*)malloc(K * maxpool_H * maxpool_W * dense1_units * sizeof(float));;
	load_data(W1, "W1_pytorch.bin", K * maxpool_H * maxpool_W * dense1_units);

	float *b1 = (float*)malloc(dense1_units * sizeof(float));;
	load_data(b1, "b1_pytorch.bin", dense1_units);

	float *W2 = (float*)malloc(dense1_units * dense2_units * sizeof(float));;
	load_data(W2, "W2_pytorch.bin", dense1_units * dense2_units);

	float *b2 = (float*)malloc(dense2_units * sizeof(float));;
	load_data(b2, "b2_pytorch.bin", dense2_units);

	//====================================================================
	// original file to compare

	float *conv_fusion_origin = (float*)malloc(N * K * P * Q * sizeof(float));
	load_data(conv_fusion_origin, "batchnorm_pytorch.bin", N * K * H * W);

	float *maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(maxpool_origin, "maxpool_pytorch.bin", N * K * maxpool_H * maxpool_W);

	float *relu_maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(relu_maxpool_origin, "relu_maxpool_pytorch.bin", N * K * maxpool_H * maxpool_W);

	float *dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));;
	load_data(dense1_origin, "dense1_pytorch.bin", N * dense1_units);

	float *relu_dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(relu_dense1_origin, "relu_dense1_pytorch.bin", N * dense1_units);

	float *dense2_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(dense2_origin, "dense2_pytorch.bin", N * dense2_units);

	float *result_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(result_origin, "result_pytorch.bin", N * dense2_units);

	//====================================================================

	//conv, maxpool, batchnorm, relu_batchnorm, dense1, relu_dense1, dense2, result

	//====================================================================

	float *image = (float*)malloc(N * image_size * sizeof(float));
	float *label = (float*)malloc(N * sizeof(float));
	split_image_label_normalization(image, label, data, N, image_size, m);

	float *conv_fusion = (float*)malloc(N * K * H * W * sizeof(float));
	MNISTCNN::conv_batchnorm_fusion(conv_fusion, image, kernel, gamma, beta, mean, variance, epsilon, N, C, K, H, W, kH, kW, pH, pH, pW, pW, sH, sW);

	float *maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	MNISTCNN::maxpooling(maxpool, conv_fusion, N, K, P, Q, maxpool_kH, maxpool_kW, maxpool_pH, maxpool_pH, maxpool_pW, maxpool_pW, maxpool_sH, maxpool_sW);

	float *relu_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	MNISTCNN::relu(relu_maxpool, maxpool, N, K * maxpool_H * maxpool_W);

	float *dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	MNISTCNN::dense(dense1, relu_maxpool, W1, b1, N, K, dense1_units, maxpool_H, maxpool_W);

	float *relu_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	MNISTCNN::relu(relu_dense1, dense1, N, dense1_units);

	float *dense2 = (float*)malloc(N * dense1_units * dense2_units * sizeof(float));
	MNISTCNN::dense(dense2, relu_dense1, W2, b2, N, dense1_units, dense2_units, 1, 1);

	float *result = (float*)malloc(N * dense2_units * sizeof(float));
	MNISTCNN::softmax(result, dense2, N, dense2_units);

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

	printf("================================================================================\n\n");

	printf("after softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = result[index] - result_origin[index];
			printf("my answer: %.6f, real answer: %.6f, difference: %.6f\n\n", result[index], result_origin[index], diff);
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

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}
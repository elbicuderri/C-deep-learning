#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <math.h>
#include <stdlib.h>
#include "function.h"

int main()
{
	const int N = 5;
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
	const int dH = 1;
	const int dW = 1;
	const int P = ((H + 2 * pH - kH) / sH) + 1;
	const int Q = ((W + 2 * pW - kW) / sW) + 1;
	const int maxpool_H = 14;
	const int maxpool_W = 14;
	const int maxpool_kH = 2;
	const int maxpool_kW = 2;
	const int maxpool_pH = 0;
	const int maxpool_pW = 0;
	const int maxpool_sH = 2;
	const int maxpool_sW = 2;
	const int dense1_units = 120;
	const int dense2_units = 10;
	const float epsilon = 0.001f;

	float *h_data = (float*)malloc(data_size * sizeof(float));
	load_data(h_data, "mnist_test_float.bin", data_size);

	//image
	float *h_image = (float*)malloc(N *  image_size * sizeof(float));
	float *h_label = (float*)malloc(N * sizeof(float));
	split_image_label_normalization(h_image, h_label, h_data, N, image_size);

	float *d_image;
	cudaMalloc((void **)&d_image, (N * image_size) * sizeof(float));
	cudaMemcpy(d_image, h_image, (N * image_size) * sizeof(float), cudaMemcpyHostToDevice);

	//kernel of convolution layer
	float *h_kernel = (float*)malloc(K * C * kH * kW * sizeof(float));
	load_data(h_kernel, "kernel_torch.bin", K * C * kH * kW);

	float *d_kernel;
	cudaMalloc((void **)&d_kernel, K * C * kH * kW * sizeof(float));
	cudaMemcpy(d_kernel, h_kernel, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

	//gamma,beta,mean,variance
	float *h_gamma = (float*)malloc(K * sizeof(float));
	load_data(h_gamma, "gamma_torch.bin", K);

	float *h_beta = (float*)malloc(K * sizeof(float));;
	load_data(h_beta, "beta_torch.bin", K);

	float *h_mean = (float*)malloc(K * sizeof(float));;
	load_data(h_mean, "mean_torch.bin", K);

	float *h_variance = (float*)malloc(K * sizeof(float));;
	load_data(h_variance, "variance_torch.bin", K);

	//=================================================================
	//get weight from gamma,beta,mean,variance
	float *h_weight = (float*)malloc(K * C * kH * kW * sizeof(float));

	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int kh = 0; kh < kH; kh++) {
				for (int kw = 0; kw < kW; kw++) {
					int index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
					h_weight[index] = (h_gamma[k] * h_kernel[index]) / (sqrtf(h_variance[k] + epsilon));
				}
			}
		}
	}

	float *d_weight;
	cudaMalloc((void **)&d_weight, K * C * kH * kW * sizeof(float));
	cudaMemcpy(d_weight, h_weight, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

	//get bias from gamma,beta,mean,variance
	float *h_bias = (float*)malloc(K * sizeof(float));

	for (int k = 0; k < K; k++) {
		h_bias[k] = h_beta[k] - ((h_gamma[k] * h_mean[k]) / (sqrtf(h_variance[k] + epsilon)));
	}

	float *d_bias;
	cudaMalloc((void **)&d_bias, K * sizeof(float));
	cudaMemcpy(d_bias, h_bias, K * sizeof(float), cudaMemcpyHostToDevice);

	//dense1 weight
	float *h_W1 = (float*)malloc(K * maxpool_H * maxpool_W * dense1_units * sizeof(float));;
	load_data(h_W1, "W1_torch.bin", K * maxpool_H * maxpool_W * dense1_units);
	float *d_W1;
	cudaMalloc((void **)&d_W1, K * maxpool_H * maxpool_W * dense1_units * sizeof(float));
	cudaMemcpy(d_W1, h_W1, K * maxpool_H * maxpool_W * dense1_units * sizeof(float), cudaMemcpyHostToDevice);

	//dense1 bias
	float *h_b1 = (float*)malloc(dense1_units * sizeof(float));;
	load_data(h_b1, "b1_torch.bin", dense1_units);
	float *d_b1;
	cudaMalloc((void **)&d_b1, dense1_units * sizeof(float));
	cudaMemcpy(d_b1, h_b1, dense1_units * sizeof(float), cudaMemcpyHostToDevice);

	//dense2 weight
	float *h_W2 = (float*)malloc(dense1_units * dense2_units * sizeof(float));;
	load_data(h_W2, "W2_torch.bin", dense1_units * dense2_units);
	float *d_W2;
	cudaMalloc((void **)&d_W2, dense1_units * dense2_units * sizeof(float));
	cudaMemcpy(d_W2, h_W2, dense1_units * dense2_units * sizeof(float), cudaMemcpyHostToDevice);

	//dense2 bias
	float *h_b2 = (float*)malloc(dense2_units * sizeof(float));;
	load_data(h_b2, "b2_torch.bin", dense2_units);
	float *d_b2;
	cudaMalloc((void **)&d_b2, dense2_units * sizeof(float));
	cudaMemcpy(d_b2, h_b2, dense2_units * sizeof(float), cudaMemcpyHostToDevice);

	//=================================================================
	// data format, type

	cudnnTensorFormat_t dataformat = CUDNN_TENSOR_NCHW;
	cudnnDataType_t datatype = CUDNN_DATA_FLOAT;
	cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;

	//=================================================================
	// HANDLE

	cudnnHandle_t HANDLE;
	cudnnCreate(&HANDLE);

	////=================================================================
	// Convolution_fusion

	cudnnTensorDescriptor_t imageDesc;
    cudnnCreateTensorDescriptor(&imageDesc);


    cudnnSetTensor4dDescriptor(
		imageDesc,
		dataformat,
		datatype,
		N,
		C,
		H,
		W);

	cudnnFilterDescriptor_t kernelDesc;
    cudnnCreateFilterDescriptor(&kernelDesc);

	cudnnSetFilter4dDescriptor(
		kernelDesc,
		datatype,
		dataformat,
		K,
		C,
		kH,
		kW);

	cudnnConvolutionDescriptor_t convDesc;
	cudnnCreateConvolutionDescriptor(&convDesc);

	cudnnSetConvolution2dDescriptor(
		convDesc,
		pH,
		pW,
		sH,
		sW,
		dH,
		dW,
		conv_mode,
		datatype);

	cudnnTensorDescriptor_t conv_imageDesc;
    cudnnCreateTensorDescriptor(&conv_imageDesc);

	cudnnSetTensor4dDescriptor(
		conv_imageDesc,
		dataformat,
		datatype,
		N,
		K,
		P,
		Q);


	cudnnConvolutionFwdAlgo_t conv_algo;
	cudnnConvolutionFwdPreference_t preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	size_t memoryLimitInBytes = 0;

	cudnnGetConvolutionForwardAlgorithm(
		HANDLE,
		imageDesc,
		kernelDesc,
		convDesc,
		conv_imageDesc,
		preference,
		memoryLimitInBytes,
		&conv_algo);


	size_t sizeInBytes;
	cudnnGetConvolutionForwardWorkspaceSize(
		HANDLE,
		imageDesc,
		kernelDesc,
		convDesc,
		conv_imageDesc,
		conv_algo,
		&sizeInBytes);

	void *d_workspace_conv_fusion;
	cudaMalloc((void **)&d_workspace_conv_fusion, sizeInBytes);

	const float alpha_conv = 1.0f;
	const float beta_conv = 0.0f;

	void *d_conv_fusion;
	cudaMalloc((void **)&d_conv_fusion, N * K * P * Q * sizeof(float)); // cuda allocate 

	cudnnConvolutionForward(
		HANDLE,
		&alpha_conv,
		imageDesc,
		d_image,
		kernelDesc,
		d_weight,
		convDesc,
		conv_algo,
		d_workspace_conv_fusion,
		sizeInBytes,
		&beta_conv,
		conv_imageDesc,
		d_conv_fusion);

	//=================================================================
	// add bias

	cudnnTensorDescriptor_t biasDesc;

	cudnnCreateTensorDescriptor(&biasDesc);

	cudnnSetTensor4dDescriptor(
		biasDesc,
		dataformat,
		datatype,
		1,
		K,
		1,
		1);

	const float alpha_bias = 1.0f;
	const float beta_bias = 1.0f;

	cudnnAddTensor(
		HANDLE,
		&alpha_bias,
		biasDesc,
		d_bias,
		&beta_bias,
		conv_imageDesc,
		d_conv_fusion);

	float *h_conv_fusion = (float*)malloc(N * K * P * Q * sizeof(float));
	cudaMemcpy(h_conv_fusion, d_conv_fusion, N * K * P * Q * sizeof(float), cudaMemcpyDeviceToHost);

	//=================================================================
	// Maxpooling

	cudnnPoolingDescriptor_t poolingDesc;

	cudnnPoolingMode_t pooling_mode = CUDNN_POOLING_MAX;

	cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;

	cudnnCreatePoolingDescriptor(&poolingDesc);

	cudnnSetPooling2dDescriptor(
		poolingDesc,
		pooling_mode,
		maxpoolingNanOpt,
		maxpool_kH,
		maxpool_kW,
		maxpool_pH,
		maxpool_pW,
		maxpool_sH,
		maxpool_sW);

	cudnnTensorDescriptor_t maxpoolDesc;

	cudnnCreateTensorDescriptor(&maxpoolDesc);

	cudnnSetTensor4dDescriptor(
		maxpoolDesc,
		dataformat,
		datatype,
		N,
		K,
		maxpool_H,
		maxpool_W);

	const float alpha_pool = 1.0f;
	const float beta_pool = 0.0f;

	void *d_maxpool;
	cudaMalloc((void **)&d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float)); // cuda allocate 

	cudnnPoolingForward(
		HANDLE,
		poolingDesc,
		&alpha_pool,
		conv_imageDesc,
		d_conv_fusion,
		&beta_pool,
		maxpoolDesc,
		d_maxpool);

	float *h_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	cudaMemcpy(h_maxpool, d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float), cudaMemcpyDeviceToHost);

	////=================================================================
	// relu

	void *d_relu_maxpool;
	cudaMalloc((void **)&d_relu_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float)); // cuda allocate 

	cudnnActivationDescriptor_t activationDesc;

	cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_RELU;
	cudnnNanPropagation_t reluNanOpt = CUDNN_PROPAGATE_NAN;

	double coef = 0;
	float alpha_activation = 1.0f;
	float beta_activation = 0.0f;

	cudnnCreateActivationDescriptor(&activationDesc);

	cudnnSetActivationDescriptor(
		activationDesc,
		activation_mode,
		reluNanOpt,
		coef);

	cudnnActivationForward(
		HANDLE,
		activationDesc,
		&alpha_activation,
		maxpoolDesc,
		d_maxpool,
		&beta_activation,
		maxpoolDesc,
		d_relu_maxpool);

	float *h_relu_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	cudaMemcpy(h_relu_maxpool, d_relu_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float), cudaMemcpyDeviceToHost);

	////=================================================================
	// dense1 layer

	cudnnTensorDescriptor_t x_dense1Desc;
	cudnnCreateTensorDescriptor(&x_dense1Desc);

	cudnnSetTensor4dDescriptor(
		x_dense1Desc,
		dataformat,
		datatype,
		N,
		K * maxpool_H * maxpool_W,
		1,
		1);

	cudnnFilterDescriptor_t W1Desc;

	cudnnCreateFilterDescriptor(&W1Desc);

	cudnnSetFilter4dDescriptor(
		W1Desc,
		datatype,
		dataformat,
		dense1_units,
		K * maxpool_H * maxpool_W,
		1,
		1);

	cudnnConvolutionDescriptor_t dense1Desc;
	cudnnCreateConvolutionDescriptor(&dense1Desc);

	cudnnSetConvolution2dDescriptor(
		dense1Desc,
		0,
		0,
		1,
		1,
		1,
		1,
		conv_mode,
		datatype);

	//y_dense1Desc dim

	int dense1_N;
	int dense1_C;
	int dense1_H;
	int dense1_W;

	cudnnGetConvolution2dForwardOutputDim(
		dense1Desc,
		x_dense1Desc,
		W1Desc,
		&dense1_N,
		&dense1_C,
		&dense1_H,
		&dense1_W);

	cudnnTensorDescriptor_t y_dense1Desc;
	cudnnCreateTensorDescriptor(&y_dense1Desc);

	cudnnSetTensor4dDescriptor(
		y_dense1Desc,
		dataformat,
		datatype,
		dense1_N,
		dense1_C,
		dense1_H,
		dense1_W);

	cudnnConvolutionFwdAlgo_t algo_dense1;
	cudnnConvolutionFwdPreference_t preference_dense1 = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	size_t memoryLimitInBytes_dense1 = 0;

	cudnnGetConvolutionForwardAlgorithm(
		HANDLE,
		x_dense1Desc,
		W1Desc,
		dense1Desc,
		y_dense1Desc,
		preference_dense1,
		memoryLimitInBytes_dense1,
		&algo_dense1);

	size_t sizeInBytes_dense1;
	cudnnGetConvolutionForwardWorkspaceSize(
		HANDLE,
		x_dense1Desc,
		W1Desc,
		dense1Desc,
		y_dense1Desc,
		algo_dense1,
		&sizeInBytes_dense1);

	void *d_workspace_dense1;
	cudaMalloc((void **)&d_workspace_dense1, sizeInBytes_dense1);

	const float alpha_dense1 = 1.0f;
	const float beta_dense1 = 0.0f;

	void *d_dense1;
	cudaMalloc((void **)&d_dense1, N * dense1_units * sizeof(float)); // cuda allocate 

	cudnnConvolutionForward(
		HANDLE,
		&alpha_dense1,
		x_dense1Desc,
		d_relu_maxpool,
		W1Desc,
		d_W1,
		dense1Desc,
		algo_dense1,
		d_workspace_dense1,
		sizeInBytes_dense1,
		&beta_dense1,
		y_dense1Desc,
		d_dense1);

	// dense1_bias

	cudnnTensorDescriptor_t b1Desc;

	cudnnCreateTensorDescriptor(&b1Desc);

	cudnnSetTensor4dDescriptor(
		b1Desc,
		dataformat,
		datatype,
		1,
		dense1_C,
		dense1_H,
		dense1_W);

	const float alpha_bias_dense1 = 1.0f;
	const float beta_bias_dense1 = 1.0f;

	cudnnAddTensor(
		HANDLE,
		&alpha_bias_dense1,
		b1Desc,
		d_b1,
		&beta_bias_dense1,
		y_dense1Desc,
		d_dense1);

	float *h_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	cudaMemcpy(h_dense1, d_dense1, N * dense1_units * sizeof(float), cudaMemcpyDeviceToHost);

	//relu_dense1
	void *d_relu_dense1;
	cudaMalloc((void **)&d_relu_dense1, N * dense1_units * sizeof(float)); // cuda allocate 

	cudnnActivationDescriptor_t activationDesc_dense1;

	cudnnActivationMode_t activation_mode_dense1 = CUDNN_ACTIVATION_RELU;
	cudnnNanPropagation_t reluNanOpt_dense1 = CUDNN_PROPAGATE_NAN;

	double coef_dense1 = 0;
	float alpha_activation_dense1 = 1.0f;
	float beta_activation_dense1 = 0.0f;

	cudnnCreateActivationDescriptor(&activationDesc_dense1);

	cudnnSetActivationDescriptor(
		activationDesc_dense1,
		activation_mode_dense1,
		reluNanOpt_dense1,
		coef_dense1);

	cudnnActivationForward(
		HANDLE,
		activationDesc_dense1,
		&alpha_activation_dense1,
		y_dense1Desc,
		d_dense1,
		&beta_activation_dense1,
		y_dense1Desc,
		d_relu_dense1);

	float *h_relu_dense1 = (float*)malloc(N * dense1_units * sizeof(float));
	cudaMemcpy(h_relu_dense1, d_relu_dense1, N * dense1_units * sizeof(float), cudaMemcpyDeviceToHost);

	//=====================================================================================================================================

	// dense2 layer

	cudnnTensorDescriptor_t x_dense2Desc;

	cudnnCreateTensorDescriptor(&x_dense2Desc);

	cudnnSetTensor4dDescriptor(
		x_dense2Desc,
		dataformat,
		datatype,
		dense1_N,
		dense1_C,
		dense1_H,
		dense1_W);

	cudnnFilterDescriptor_t W2Desc;

	cudnnCreateFilterDescriptor(&W2Desc);

	cudnnSetFilter4dDescriptor(
		W2Desc,
		datatype,
		dataformat,
		dense2_units,
		dense1_units,
		1,
		1);

	cudnnConvolutionDescriptor_t dense2Desc;

	cudnnCreateConvolutionDescriptor(&dense2Desc);

	cudnnSetConvolution2dDescriptor(
		dense2Desc,
		0,
		0,
		1,
		1,
		1,
		1,
		conv_mode,
		datatype);

	//y_dense2Desc dim

	int dense2_N;
	int dense2_C;
	int dense2_H;
	int dense2_W;

	cudnnGetConvolution2dForwardOutputDim(
		dense2Desc,
		x_dense2Desc,
		W2Desc,
		&dense2_N,
		&dense2_C,
		&dense2_H,
		&dense2_W);

	cudnnTensorDescriptor_t y_dense2Desc;

	cudnnCreateTensorDescriptor(&y_dense2Desc);

	cudnnSetTensor4dDescriptor(
		y_dense2Desc,
		dataformat,
		datatype,
		dense2_N,
		dense2_C,
		dense2_H,
		dense2_W);

	cudnnConvolutionFwdAlgo_t algo_dense2;
	cudnnConvolutionFwdPreference_t preference_dense2 = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	size_t memoryLimitInBytes_dense2 = 0;

	cudnnGetConvolutionForwardAlgorithm(
		HANDLE,
		x_dense2Desc,
		W2Desc,
		dense2Desc,
		y_dense2Desc,
		preference_dense2,
		memoryLimitInBytes_dense2,
		&algo_dense2);

	size_t sizeInBytes_dense2;
	cudnnGetConvolutionForwardWorkspaceSize(
		HANDLE,
		x_dense2Desc,
		W2Desc,
		dense2Desc,
		y_dense2Desc,
		algo_dense2,
		&sizeInBytes_dense2);

	void *d_workspace_dense2;
	cudaMalloc((void **)&d_workspace_dense2, sizeInBytes_dense2);

	const float alpha_dense2 = 1.0f;
	const float beta_dense2 = 0.0f;

	void *d_dense2;
	cudaMalloc((void **)&d_dense2, N * dense2_units * sizeof(float)); // cuda allocate 

	cudnnConvolutionForward(
		HANDLE,
		&alpha_dense2,
		x_dense2Desc,
		d_relu_dense1,
		W2Desc,
		d_W2,
		dense2Desc,
		algo_dense2,
		d_workspace_dense2,
		sizeInBytes_dense2,
		&beta_dense2,
		y_dense2Desc,
		d_dense2);

	//dense2_bias

	cudnnTensorDescriptor_t b2Desc;

	cudnnCreateTensorDescriptor(&b2Desc);

	cudnnSetTensor4dDescriptor(
		b2Desc,
		dataformat,
		datatype,
		1,
		dense2_C,
		dense2_H,
		dense2_W);

	const float alpha_bias_dense2 = 1.0f;
	const float beta_bias_dense2 = 1.0f;

	cudnnAddTensor(
		HANDLE,
		&alpha_bias_dense2,
		b2Desc,
		d_b2,
		&beta_bias_dense2,
		y_dense2Desc,
		d_dense2);

	float *h_dense2 = (float*)malloc(N * dense2_units * sizeof(float));
	cudaMemcpy(h_dense2, d_dense2, N * dense2_units * sizeof(float), cudaMemcpyDeviceToHost);

	//softmax_dense2
	void *d_result;
	cudaMalloc((void **)&d_result, N * dense2_units * sizeof(float)); // cuda allocate 

	float alpha_activation_softmax = 1.0f;
	float beta_activation_softmax = 0.0f;

	cudnnSoftmaxAlgorithm_t algorithm_softmax = CUDNN_SOFTMAX_FAST;
	cudnnSoftmaxMode_t mode_softmax = CUDNN_SOFTMAX_MODE_CHANNEL;
	cudnnSoftmaxForward(
		HANDLE,
		algorithm_softmax,
		mode_softmax,
		&alpha_activation_softmax,
		y_dense2Desc,
		d_dense2,
		&beta_activation_softmax,
		y_dense2Desc,
		d_result);

	float *h_result = (float*)malloc(N * dense2_units * sizeof(float));
	cudaMemcpy(h_result, d_result, N * dense2_units * sizeof(float), cudaMemcpyDeviceToHost);

	//=====================================================================================================================================

	// original file to compare

	//=================================================================

	float *conv_fusion_origin = (float*)malloc(N * K * P * Q * sizeof(float));
	load_data(conv_fusion_origin, "batchnorm_torch.bin", N * K * P * Q);

	float *maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(maxpool_origin, "maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *relu_maxpool_origin = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
	load_data(relu_maxpool_origin, "relu_maxpool_torch.bin", N * K * maxpool_H * maxpool_W);

	float *dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(dense1_origin, "dense1_torch.bin", N * dense1_units);

	float *relu_dense1_origin = (float*)malloc(N * dense1_units * sizeof(float));
	load_data(relu_dense1_origin, "relu_dense1_torch.bin", N * dense1_units);

	float *dense2_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(dense2_origin, "dense2_torch.bin", N * dense2_units);

	float *result_origin = (float*)malloc(N * dense2_units * sizeof(float));
	load_data(result_origin, "result_torch.bin", N * dense2_units);

	//printf("conv_fusion_origin: \n\n");
	//print_image(batchnorm_origin, N, K, P, Q);

	//printf("conv_fusion: \n\n");
	//print_image(h_conv_fusion, N, K, P, Q);

	//printf("maxpool_origin: \n\n");
	//print_image(maxpool_origin, N, K, maxpool_H, maxpool_W);

	//printf("maxpool: \n\n");
	//print_image(h_maxpool, N, K, maxpool_H, maxpool_W);

	//printf("relu_maxpool_origin: \n\n");
	//print_image(relu_maxpool_origin, N, K, maxpool_H, maxpool_W);

	//printf("relu_maxpool: \n\n");
	//print_image(h_relu_maxpool, N, K, maxpool_H, maxpool_W);

	printf("dense1_origin: \n\n");
	print_image(dense1_origin, 1, 1, 1, 30);

	printf("dense1: \n\n");
	print_image(h_dense1, 1, 1, 1, 30);

	printf("relu_dense1_origin: \n\n");
	print_image(relu_dense1_origin, 1, 1, 1, 30);

	printf("relu_dense1: \n\n");
	print_image(h_relu_dense1, 1, 1, 1, 30);

	//printf("dense2_origin: \n\n");
	//print_image(dense2_origin, N, 1, 1, 10);

	//printf("dense2: \n\n");
	//print_image(h_dense2, N, 1, 1, 10);

	printf("before softmax : \n\n\n");
	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = fabsf(h_dense2[index] - dense2_origin[index]);
			printf("my answer: %.5f, real answer: %.5f, difference: %.5f\n\n", h_dense2[index], dense2_origin[index], diff);
		}
		printf("\n");
	}
	printf("===============================================================================================================================================================================================================================\n\n");

	printf("after softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < dense2_units; j++) {
			int index = i * dense2_units + j;
			float diff = fabsf(h_result[index] - result_origin[index]);
			printf("my answer: %.5f, real answer: %.5f, difference: %.5f\n\n", h_result[index], result_origin[index], diff);
		}
		printf("\n\n");
	}

	//=================================================================

	cudnnDestroyTensorDescriptor(imageDesc);
	cudnnDestroyFilterDescriptor(kernelDesc);
	cudnnDestroyConvolutionDescriptor(convDesc);
	cudnnDestroyTensorDescriptor(conv_imageDesc);
	cudnnDestroyTensorDescriptor(biasDesc);
	cudnnDestroyPoolingDescriptor(poolingDesc);
	cudnnDestroyTensorDescriptor(maxpoolDesc);
	cudnnDestroyActivationDescriptor(activationDesc);
	cudnnDestroyTensorDescriptor(x_dense1Desc);
	cudnnDestroyTensorDescriptor(x_dense2Desc);
	cudnnDestroyTensorDescriptor(y_dense1Desc);
	cudnnDestroyTensorDescriptor(y_dense2Desc);
	cudnnDestroyFilterDescriptor(W1Desc);
	cudnnDestroyFilterDescriptor(W2Desc);
	cudnnDestroyTensorDescriptor(b1Desc);
	cudnnDestroyTensorDescriptor(b2Desc);
	cudnnDestroyConvolutionDescriptor(dense1Desc);
	cudnnDestroyConvolutionDescriptor(dense2Desc);
	cudnnDestroyActivationDescriptor(activationDesc_dense1);

	cudnnDestroy(HANDLE);

	// if not zero, cudaFree
	if (d_image)
		cudaFree(d_image);
	if (d_weight)
		cudaFree(d_weight);
	if (d_bias)
		cudaFree(d_bias);
	if (d_W1)
		cudaFree(d_W1);
	if (d_b1)
		cudaFree(d_b1);
	if (d_W2)
		cudaFree(d_W2);
	if (d_b2)
		cudaFree(d_b2);
	if (d_workspace_conv_fusion)
		cudaFree(d_workspace_conv_fusion);
	if (d_conv_fusion)
		cudaFree(d_conv_fusion);
	if (d_maxpool)
		cudaFree(d_maxpool);
	if (d_relu_maxpool)
		cudaFree(d_relu_maxpool);
	if (d_workspace_dense1)
		cudaFree(d_workspace_dense1);
	if (d_dense1)
		cudaFree(d_dense1);
	if (d_relu_dense1)
		cudaFree(d_relu_dense1);
	if (d_workspace_dense2)
		cudaFree(d_workspace_dense2);
	if (d_dense2)
		cudaFree(d_dense2);
	if (d_result)
		cudaFree(d_result);

	free(h_data);
	free(h_image);
	free(h_label);
	free(h_kernel);
	free(h_gamma);
	free(h_beta);
	free(h_mean);
	free(h_variance);
	free(h_weight);
	free(h_bias);
	free(h_W1);
	free(h_b1);
	free(h_W2);
	free(h_b2);
	free(h_conv_fusion);
	free(conv_fusion_origin);
	free(h_maxpool);
	free(maxpool_origin);
	free(h_relu_maxpool);
	free(relu_maxpool_origin);
	free(h_dense1);
	free(dense1_origin);
	free(h_relu_dense1);
	free(relu_dense1_origin);
	free(h_dense2);
	free(dense2_origin);
	free(h_result);
	free(result_origin);

	return 0;
}
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <cuda.h>
//#include <cudnn.h>
//#include <cublas.h>
//#include <math.h>
//
//#define KERNEL_SIZE 5
//
//#define KERNEL_OFFSET (KERNEL_SIZE / 2)
//
//__constant__ float c_kernel[5 * 5];
//
//void load_data(float *output, char *name, int size);
//
//void split_image_label(float *image, float *label, float *input, int batch, int size);
//
//void print_image(float *image, int batch, int channel, int height, int width);
//
//void normalization(float *output, float *input, int size);
//
//void pad_image(float *output, float *input, float *kernel,
//	int batch, int in_channel, int out_channel, int input_height, int input_width,
//	int kernel_height, int kernel_width,
//	int pad_top, int pad_bottom, int pad_left, int pad_right,
//	int stride_height, int stride_width);
//
//// 2D Convolution Kernel
//// Takes:
////  matrix: Input matrix
////  result: Convolution result
////  N:      Dimensions of the matrices
//__global__ void convolution_2d(float *output, float *input, int N) {
//	// Calculate the global thread positions
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Starting index for calculation
//	int start_r = row - KERNEL_OFFSET;
//	int start_c = col - KERNEL_OFFSET;
//
//	// Temp value for accumulating the result
//	int sum = 0;
//
//	// Iterate over all the rows
//	for (int i = 0; i < KERNEL_SIZE; i++) {
//		// Go over each column
//		for (int j = 0; j < KERNEL_SIZE; j++) {
//			// Range check for rows
//			if ((start_r + i) >= 0 && (start_r + i) < N) {
//				// Range check for columns
//				if ((start_c + j) >= 0 && (start_c + j) < N) {
//					// Accumulate result
//					sum += input[(start_r + i) * N + (start_c + j)] *
//						c_kernel[i * KERNEL_SIZE + j];
//				}
//			}
//		}
//	}
//
//	// Write back the result
//	output[row * N + col] = sum;
//}
//
//int main()
//{
//	const int N = 1;
//	const int image_size = 784;
//	const int pad_image_size = 1024;
//	const int data_size = N * (image_size + 1);
//	const int K = 5;
//	const int C = 1;
//	const int H = 28;
//	const int W = 28;
//	const int kH = 5;
//	const int kW = 5;
//	const int pH = 2;
//	const int pW = 2;
//	const int sH = 1;
//	const int sW = 1;
//	//const int dH = 1;
//	//const int dW = 1;
//	//const int maxpool_kH = 2;
//	//const int maxpool_kW = 2;
//	//const int maxpool_H = 14;
//	//const int maxpool_W = 14;
//	//const int maxpool_pH = 0;
//	//const int maxpool_pW = 0;
//	//const int maxpool_sH = 2;
//	//const int maxpool_sW = 2;
//	//const int dense1_units = 120;
//	//const int dense2_units = 10;
//	//const float epsilon = 0.001f;
//
//
//	//load data
//	float *data = (float*)malloc(data_size * sizeof(float));
//	load_data(data, "mnist_test_float.bin", data_size);
//
//	float *image = (float*)malloc(N *  image_size * sizeof(float));
//	float *label = (float*)malloc(N * sizeof(float));
//	split_image_label(image, label, data, N, image_size);
//
//	printf("image_origin: \n");
//	print_image(image, N, C, H, W);
//
//	//copy kernel to the constanst
//	float *kernel = (float*)malloc(K * C * kH * kW * sizeof(float));
//	load_data(kernel, "kernel_torch.bin", K * C * kH * kW);
//	cudaMemcpyToSymbol(c_kernel, kernel, K * C * kH * kW * sizeof(float));
//
//	printf("kernel: \n");
//	print_image(kernel, K, C, kH, kW);
//
//	float *image_norm = (float*)malloc(N * image_size * sizeof(float)); // 255.0
//	normalization(image_norm, image, N * image_size);
//
//	printf("image_normalization: \n");
//	print_image(image_norm, N, C, H, W);
//
//	float *image_padded;
//	image_padded = (float*)malloc(N * C * (H + 2 * pH) * (W + 2 * pW) * sizeof(float));
//
//	pad_image(image_padded, image_norm, kernel, N, C, K, H, W, kH, kW, pH, pH, pW, pW, sH, sW);
//
//	printf("image_padded: \n");
//	print_image(image_padded, N, C, 32, 32);
//
//	//free unused data
//	free(data);
//	free(image);
//	free(label);
//	free(kernel);
//
//	//copy x_data to the gpu 
//	float *d_x;
//	cudaMalloc((void **)&d_x, (N * pad_image_size) * sizeof(float));
//	cudaMemcpy(d_x, image_padded, (N * pad_image_size) * sizeof(float), cudaMemcpyHostToDevice);
//
//	//copy kernel to the constanst
//	cudaMemcpyToSymbol(c_kernel, kernel, K * C * kH * kW * sizeof(float));
//
//	//allocat result and d_result
//	float *result = (float*)malloc((N * image_size) * sizeof(float));
//	float *d_result;
//	cudaMalloc((void **)&d_result, (N * image_size) * sizeof(float));
//
//	// Calculate grid dimensions
//	int THREADS = 28;
//	//int BLOCKS = (H + THREADS - 1) / THREADS;
//	int BLOCKS = 1;
//
//	// Dimension launch arguments
//	dim3 block_dim(THREADS, THREADS);
//	dim3 grid_dim(BLOCKS, BLOCKS);
//
//	// Perform 2D Convolution
//	convolution_2d << <grid_dim, block_dim >> > (d_result, d_x, 32);
//
//	// Copy the result back to the CPU
//	cudaMemcpy(result, d_result, (N * image_size) * sizeof(float), cudaMemcpyDeviceToHost);
//
//	printf("after convolution: \n");
//	print_image(result, N, K, H, W);
//
//	free(image_norm);
//	free(image_padded);
//	free(result);
//
//	cudaFree(d_x);
//	cudaFree(d_result);
//
//	return 0;
//}
//
//
//void load_data(float *output, char *name, int size)
//{
//	FILE *pFile = fopen(name, "rb");
//
//	if (pFile == NULL) {
//		printf("cannot find file\n");
//		exit(-1);
//	}
//
//	size_t sizet = fread(output, size * sizeof(float), 1, pFile);
//
//	if (sizet != 1) {
//		printf("read error!\n");
//		exit(-1);
//	}
//
//	fclose(pFile);
//}
//
//
//void split_image_label(float *image, float *label, float *input, int batch, int size)
//{
//	int N = batch;
//	int S = size;
//
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < S; j++) {
//			int index = i * S + j;
//			image[index] = input[i * (S + 1) + (j + 1)];
//		}
//	}
//
//	for (int i = 0; i < N; i++) {
//		label[i] = input[i * (S + 1)];
//	}
//
//}
//
//
//void print_image(float *image, int batch, int channel, int height, int width)
//{
//	int N = batch;
//	int C = channel;
//	int H = height;
//	int W = width;
//
//	for (int n = 0; n < N; n++) {
//		for (int c = 0; c < C; c++) {
//			for (int h = 0; h < height; h++) {
//				for (int w = 0; w < width; w++) {
//					printf("%6.2f ", image[n * C * H * W + c * H * W + h * W + w]);
//				}
//				printf("\n\n");
//			}
//			printf("\n==========================================================================================================================================================================\n");
//		}
//	}
//
//}
//
//
//void normalization(float *output, float *input, int size)
//{
//	float m = 255.0;
//	for (int i = 0; i < size; i++) {
//		output[i] = input[i] / m;
//	}
//}
//
//
//void pad_image(float *output, float *input, float *kernel,
//	int batch, int in_channel, int out_channel, int input_height, int input_width,
//	int kernel_height, int kernel_width,
//	int pad_top, int pad_bottom, int pad_left, int pad_right,
//	int stride_height, int stride_width)
//{
//	int N = batch;
//	int C = in_channel;
//	int K = out_channel;
//	int H = input_height;
//	int W = input_width;
//	int kH = kernel_height;
//	int kW = kernel_width;
//	int pT = pad_top;
//	int pB = pad_bottom;
//	int pL = pad_left;
//	int pR = pad_right;
//	int pH = H + pT + pB;
//	int pW = H + pL + pR;
//	int sH = stride_height;
//	int sW = stride_width;
//	int oH = ((input_height + pT + pB - kernel_height) / stride_height) + 1; // output_height
//	int oW = ((input_width + pL + pR - kernel_width) / stride_width) + 1; // output_width
//
//	//fill pad image
//	for (int n = 0; n < N; n++) {
//		for (int c = 0; c < C; c++) {
//			for (int ph = 0; ph < pH; ph++) {
//				for (int pw = 0; pw < pW; pw++) {
//					int pad_index = n * C * pH * pW + c * pH * pW + ph * pW + pw;
//					if (ph < pT || ph >= H + pT || pw < pL || pw >= W + pL) {
//						output[pad_index] = 0.0f;
//					}
//					else {
//						int input_index = n * C * H * W + c * H * W + (ph - pT) * W + pw - pL;
//						output[pad_index] = input[input_index];
//					}
//				}
//			}
//		}
//	}
//
//}
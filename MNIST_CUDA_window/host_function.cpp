#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <corecrt.h>
#include <stdio.h>
#include <cstdlib>
#include <float.h>

void load_data(float *output, char *name, int size)
{
	FILE *pFile = fopen(name, "rb");

	if (pFile == NULL) {
		printf("cannot find file\n");
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
			printf("\n=============================================================================================================================================================================\n");
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
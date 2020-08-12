#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <math.h>
#include <stdlib.h>


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

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int image_size)
{
	int N = total_number;
	int S = image_size;
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
			printf("\n==========================================================================================================================================================================\n");
		}
	}

}
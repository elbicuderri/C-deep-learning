#pragma once
void load_data(float *output, char *name, int size);

void print_image(float *image, int batch, int height, int width, int channel);

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int size);

void exponential_sum(float *output, float *input, int batch, int channel);
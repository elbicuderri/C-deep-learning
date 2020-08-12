#pragma once

void load_data(float *output, char *name, int size);

void print_image(float *image, int batch, int channel, int height, int width);

void split_image_label_normalization(float *image, float *label, float *input, int total_number, int image_size);


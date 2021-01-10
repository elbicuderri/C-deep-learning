#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <memory>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "layers.h"

void load_data(float* output, const char* name, int size);

template<typename T>
std::vector<T> ReadData(const std::string file_name);

std::vector<float> image_normalization(const float* input, int& batch, int& image_size, const float& constant);

int main()
{
	int image_size = 28 * 28;
	int N = 10;
	const int data_size = N * (image_size + 1);
	const int C = 1;
	const int H = 28;
	const int W = 28;
	const int K_1 = 4;
	const int kernel_size = 3;
	const int padding = 1;
	const int K_2 = 8;
	const int classes = 10;
	const float epsilon = 0.00001f;

	std::string root = "C:\\GitHub\\PyTorch_Study\\soynet_lecture\\mnist_resnet\\weight\\";

	float* data = new float[data_size];
	const char* data_file = "C:\\dev\\mnist_cnn\\mnist_cnn\\data/mnist_test_float.bin";

	load_data(data, data_file, data_size);

	std::vector<float> conv0_weight = ReadData<float>(root + "conv0.0.weight_pytorch_resnet.bin");
	std::vector<float> conv0_bn_weight = ReadData<float>(root + "conv0.1.weight_pytorch_resnet.bin");
	std::vector<float> conv0_bn_bias = ReadData<float>(root + "conv0.1.bias_pytorch_resnet.bin");
	std::vector<float> conv0_bn_mean = ReadData<float>(root + "conv0.bn.mean_pytorch_resnet.bin");
	std::vector<float> conv0_bn_var = ReadData<float>(root + "conv0.bn.var_pytorch_resnet.bin");

	std::vector<float> conv1_weight = ReadData<float>(root + "conv1.0.weight_pytorch_resnet.bin");
	std::vector<float> conv1_bn_weight = ReadData<float>(root + "conv1.1.weight_pytorch_resnet.bin");
	std::vector<float> conv1_bn_bias = ReadData<float>(root + "conv1.1.bias_pytorch_resnet.bin");
	std::vector<float> conv1_bn_mean = ReadData<float>(root + "conv1.bn.mean_pytorch_resnet.bin");
	std::vector<float> conv1_bn_var = ReadData<float>(root + "conv1.bn.var_pytorch_resnet.bin");

	std::vector<float> block1_conv1_weight = ReadData<float>(root + "block1.0.weight_pytorch_resnet.bin");
	std::vector<float> block1_bn1_weight = ReadData<float>(root + "block1.1.weight_pytorch_resnet.bin");
	std::vector<float> block1_bn1_bias = ReadData<float>(root + "block1.1.bias_pytorch_resnet.bin");
	std::vector<float> block1_bn1_mean = ReadData<float>(root + "block1.bn1.mean_pytorch_resnet.bin");
	std::vector<float> block1_bn1_var = ReadData<float>(root + "block1.bn1.var_pytorch_resnet.bin");

	std::vector<float> block1_conv2_weight = ReadData<float>(root + "block1.3.weight_pytorch_resnet.bin");
	std::vector<float> block1_bn2_weight = ReadData<float>(root + "block1.4.weight_pytorch_resnet.bin");
	std::vector<float> block1_bn2_bias = ReadData<float>(root + "block1.4.bias_pytorch_resnet.bin");
	std::vector<float> block1_bn2_mean = ReadData<float>(root + "block1.bn2.mean_pytorch_resnet.bin");
	std::vector<float> block1_bn2_var = ReadData<float>(root + "block1.bn2.var_pytorch_resnet.bin");

	std::vector<float> block2_conv1_weight = ReadData<float>(root + "block2.0.weight_pytorch_resnet.bin");
	std::vector<float> block2_bn1_weight = ReadData<float>(root + "block2.1.weight_pytorch_resnet.bin");
	std::vector<float> block2_bn1_bias = ReadData<float>(root + "block2.1.bias_pytorch_resnet.bin");
	std::vector<float> block2_bn1_mean = ReadData<float>(root + "block2.bn1.mean_pytorch_resnet.bin");
	std::vector<float> block2_bn1_var = ReadData<float>(root + "block2.bn1.var_pytorch_resnet.bin");

	std::vector<float> block2_conv2_weight = ReadData<float>(root + "block2.3.weight_pytorch_resnet.bin");
	std::vector<float> block2_bn2_weight = ReadData<float>(root + "block2.4.weight_pytorch_resnet.bin");
	std::vector<float> block2_bn2_bias = ReadData<float>(root + "block2.4.bias_pytorch_resnet.bin");
	std::vector<float> block2_bn2_mean = ReadData<float>(root + "block2.bn2.mean_pytorch_resnet.bin");
	std::vector<float> block2_bn2_var = ReadData<float>(root + "block2.bn2.var_pytorch_resnet.bin");

	std::vector<float> fc_weight = ReadData<float>(root + "fc.weight_pytorch_resnet.bin");
	std::vector<float> fc_bias = ReadData<float>(root + "fc.bias_pytorch_resnet.bin");

	std::vector<float> image = image_normalization(data, N, image_size, 255.0f);

	delete[] data;

	std::vector<float> out0 = conv_bn_fusion_relu(image, conv0_weight, conv0_bn_weight, conv0_bn_bias, conv0_bn_mean, conv0_bn_var,
		epsilon, N, C, K_1, H, W, kernel_size, kernel_size, padding, 1);

	std::vector<float> res11 = conv_bn_fusion(out0, conv1_weight, conv1_bn_weight, conv1_bn_bias, conv1_bn_mean, conv1_bn_var,
		epsilon, N, K_1, K_2, H, W, kernel_size, kernel_size, padding, 2);

	std::vector<float> out111 = conv_bn_fusion_relu(out0, block1_conv1_weight, block1_bn1_weight, block1_bn1_bias, block1_bn1_mean, block1_bn1_var,
		epsilon, N, K_1, K_2, H, W, kernel_size, kernel_size, padding, 2);

	std::vector<float> out11 = conv_bn_fusion(out111, block1_conv2_weight, block1_bn2_weight, block1_bn2_bias, block1_bn2_mean, block1_bn2_var,
		epsilon, N, K_2, K_2, 14, 14, kernel_size, kernel_size, padding, 1);

	std::vector<float> out112 = add_layer(res11, out11);

	Relu(out112);

	std::vector<float> out121 = conv_bn_fusion_relu(out112, block2_conv1_weight, block2_bn1_weight, block2_bn1_bias, block2_bn1_mean, block2_bn1_var,
		epsilon, N, K_2, K_2, 14, 14, kernel_size, kernel_size, padding, 1);

	std::vector<float> out12 = conv_bn_fusion(out121, block2_conv2_weight, block2_bn2_weight, block2_bn2_bias, block2_bn2_mean, block2_bn2_var,
		epsilon, N, K_2, K_2, 14, 14, kernel_size, kernel_size, padding, 1);

	std::vector<float> out122 = add_layer(out112, out12);

	Relu(out122);

	std::vector<float> out2 = avg_pool(out122, N, K_2, 14, 14, 2, 2, 0, 2);

	int flatten_channels = 8 * 7 * 7;

	std::vector<float> last_dense = Dense(out2, fc_weight, fc_bias, N, flatten_channels, classes);

	std::vector<float> logit = log_softmax(last_dense, N, classes);

	//======================================================================================================================

	std::vector<float> last_dense_origin = ReadData<float>("C:\\GitHub\\PyTorch_Study\\soynet_lecture\\mnist_resnet\\value\\last_dense_pytorch_resnet.bin");
	std::vector<float> logit_origin = ReadData<float>("C:\\GitHub\\PyTorch_Study\\soynet_lecture\\mnist_resnet\\value\\logit_pytorch_resnet.bin");

	//======================================================================================================================

	printf("before softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < classes; j++) {
			int index = i * classes + j;
			float diff = last_dense[index] - last_dense_origin[index];
			printf("my answer: %.8f, real answer: %.8f, error: %.8f\n\n", last_dense[index], last_dense_origin[index], diff);
		}
		printf("\n");
	}

	printf("======================================================================================================================\n\n");

	printf("after softmax : \n\n\n");

	for (int i = 0; i < N; i++) {
		printf("%dth image result: \n\n", i + 1);
		for (int j = 0; j < classes; j++) {
			int index = i * classes + j;
			float diff = logit[index] - logit_origin[index];
			printf("my answer: %.8f, real answer: %.8f, error: %.8f\n\n", logit[index], logit_origin[index], diff);
		}
		printf("\n\n");
	}

	return 0;
}



template<typename T>
std::vector<T> ReadData(std::string file_name)
{
	std::ifstream input(file_name, std::ios::in | std::ios::binary);
	if (!(input.is_open()))
	{
		std::cout << "Cannot open" << file_name << "!!!" << std::endl;
		exit(-1);
	}

	std::vector<T> data;
	input.seekg(0, std::ios::end);
	auto size = input.tellg();
	input.seekg(0, std::ios::beg);

	for (int i = 0; i < size / sizeof(T); ++i)
	{
		T value;
		input.read((char*)&value, sizeof(T));
		data.emplace_back(value);
	}

	return data;
}


void load_data(float* output, const char* name, int size)
{
	FILE* pFile = fopen(name, "rb");

	if (pFile == NULL) {
		printf("cannot find %s\n", name);
		exit(-1);
	}

	size_t sizet = fread((void*)output, size * sizeof(float), 1, pFile);

	if (sizet != 1) {
		printf("%s file size error!\n", name);
		exit(-1);
	}

	fclose(pFile);
}

std::vector<float> image_normalization(const float* input, int& batch, int& image_size, const float& constant)
{
	std::vector<float> output(batch * image_size);

	for (int n = 0; n < batch; ++n)
	{
		for (int s = 0; s < image_size; ++s)
		{
			int index = n * image_size + s;
			output[index] = input[n * (image_size + 1) + (s + 1)] / constant;
		}
	}

	return output;
}
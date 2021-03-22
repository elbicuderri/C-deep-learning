#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>

void Relu(std::vector<float>& input);

std::vector<float> add_layer(const std::vector<float>& input_1, const std::vector<float>& input_2);

std::vector<float> Dense(const std::vector<float>& Input,
	const std::vector<float>& Weight,
	const std::vector<float>& Bias,
	int N,
	int C,
	int K);

std::vector<float> conv_bn_fusion(
	const std::vector<float>& input,
	const std::vector<float>& kernel,
	const std::vector<float>& gamma,
	const std::vector<float>& beta,
	const std::vector<float>& mean,
	const std::vector<float>& variance,
	float epsilon,
	int N,
	int C,
	int K,
	int H,
	int W,
	int kH,
	int kW,
	int padding,
	int stride);

std::vector<float> conv_bn_fusion_relu(
	const std::vector<float>& input,
	const std::vector<float>& kernel,
	const std::vector<float>& gamma,
	const std::vector<float>& beta,
	const std::vector<float>& mean,
	const std::vector<float>& variance,
	float epsilon,
	int N,
	int C,
	int K,
	int H,
	int W,
	int kH,
	int kW,
	int padding,
	int stride);

std::vector<float> avg_pool(const std::vector<float>& input,
	int N,
	int C,
	int H,
	int W,
	int kH,
	int kW,
	int padding,
	int stride);

std::vector<float> log_softmax(const std::vector<float>& input, int N, int C);


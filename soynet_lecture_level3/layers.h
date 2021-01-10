#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>

void Relu(std::vector<float>& input);

std::vector<float> add_layer(const std::vector<float> input_1, const std::vector<float> input_2);

std::vector<float> Dense(const std::vector<float> Input,
	const std::vector<float> Weight,
	const std::vector<float> Bias,
	int N,
	int C,
	int K);

std::vector<float> conv_bn_fusion(
	std::vector<float> input,
	std::vector<float> kernel,
	std::vector<float> gamma,
	std::vector<float> beta,
	std::vector<float> mean,
	std::vector<float> variance,
	const float& epsilon,
	const int& N,
	const int& C,
	const int& K,
	const int& H,
	const int& W,
	const int& kH,
	const int& kW,
	const int& padding,
	const int& stride);

std::vector<float> conv_bn_fusion_relu(
	std::vector<float> input,
	std::vector<float> kernel,
	std::vector<float> gamma,
	std::vector<float> beta,
	std::vector<float> mean,
	std::vector<float> variance,
	const float& epsilon,
	const int& N,
	const int& C,
	const int& K,
	const int& H,
	const int& W,
	const int& kH,
	const int& kW,
	const int& padding,
	const int& stride);

std::vector<float> avg_pool(std::vector<float> input,
	const int& N,
	const int& C,
	const int& H,
	const int& W,
	const int& kH,
	const int& kW,
	const int& padding,
	const int& stride);

std::vector<float> log_softmax(std::vector<float> input, const int& N, const int& C);


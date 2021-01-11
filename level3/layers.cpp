#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>

void Relu(std::vector<float>& input)
{
	for (auto &e : input) {
		if (e < 0.0f) { e = 0.0f; }
	}
}

std::vector<float> add_layer(const std::vector<float> input_1, const std::vector<float> input_2)
{
	//assert (input_1.size() == input_2.size());

	int Length = (int)input_1.size();
	std::vector<float> output(Length);

	for (int i = 0; i < Length; ++i)
	{
		output[i] = input_1[i] + input_2[i];
	}

	return output;
}

std::vector<float> Dense(const std::vector<float> Input,
	const std::vector<float> Weight,
	const std::vector<float> Bias,
	int N,
	int C,
	int K)
{

	std::vector<float> Output(N * K);

	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			float sum = (float)0.0f;
			for (int c = 0; c < C; ++c) {
				int Input_index = n * C + c;
				int Weight_index = k * C + c;
				float s = Input[Input_index] * Weight[Weight_index];
				sum += s;
			}
			sum += Bias[k];
			int Output_index = n * K + k;
			Output[Output_index] = sum;
		}
	}

	return Output;
}

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
	const int& stride)
{
	// N : batch
	// C : in_channel
	// K : out_channel
	// H : input_height
	// W : input_width
	// kH : kernel_height
	// kW : kernel_width

	int P = ((H + 2 * padding - kH) / stride) + 1; // output_height
	int Q = ((W + 2 * padding - kW) / stride) + 1; // output_width

	//set weight ( convolution weight + batchnorm weights )
	// float* weight = (float*)malloc(K * C * kH * kW * sizeof(float));
	float* weight = new float[K * C * kH * kW];
	// std::unique_ptr<float[]> weight = std::make_unique<float[]>(K * C * kH * kW);

	for (int k = 0; k < K; ++k) {
		for (int c = 0; c < C; ++c) {
			for (int kh = 0; kh < kH; ++kh) {
				for (int kw = 0; kw < kW; ++kw) {
					int index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
					weight[index] = (gamma[k] * kernel[index]) / (sqrtf(variance[k] + epsilon));
				}
			}
		}
	}

	//set bias ( convolution betas + batchnorm weights )
	// float *bias = (float*)malloc(K * sizeof(float));
	float* bias = new float[K];
	// std::unique_ptr<float[]> bias = std::make_unique<float[]>(K); // unique_ptr can't be initialized...

	for (int k = 0; k < K; ++k) {
		bias[k] = beta[k] - ((gamma[k] * mean[k]) / (sqrtf(variance[k] + epsilon)));
	}

	std::vector<float> output(N * K * P * Q);

	//convolution + batchnormalization
	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			for (int p = 0; p < P; ++p) { //image_row
				for (int q = 0; q < Q; ++q) { //image_column
					float sum = 0.0f;
					for (int c = 0; c < C; ++c) {
						for (int kh = 0; kh < kH; ++kh) {//kernel_height
							int input_h_index = p * stride + kh - padding;
							if (input_h_index >= 0 && input_h_index < H) {
								for (int kw = 0; kw < kW; ++kw) { //kernel_width
									int input_w_index = q * stride + kw - padding;
									if (input_w_index >= 0 && input_w_index < W) {
										int input_index = n * C * H * W + c * H * W + input_h_index * W + input_w_index;
										int weight_index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
										float s = weight[weight_index] * input[input_index];
										sum += s;
									}
								}
							}
						}
					}
					int output_index = n * K * P * Q + k * P * Q + p * Q + q;
					sum += bias[k];
					output[output_index] = sum;
					//output.emplace_back(sum);
				}
			}
		}
	}

	delete[] weight;
	delete[] bias;

	return output;

}

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
	const int& stride)
{
	// N : batch
	// C : in_channel
	// K : out_channel
	// H : input_height
	// W : input_width
	// kH : kernel_height
	// kW : kernel_width

	int P = ((H + 2 * padding - kH) / stride) + 1; // output_height
	int Q = ((W + 2 * padding - kW) / stride) + 1; // output_width

	//set weight ( convolution weight + batchnorm weights )
	float* weight = new float[K * C * kH * kW];

	for (int k = 0; k < K; ++k) {
		for (int c = 0; c < C; ++c) {
			for (int kh = 0; kh < kH; ++kh) {
				for (int kw = 0; kw < kW; ++kw) {
					int index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
					weight[index] = (gamma[k] * kernel[index]) / (sqrtf(variance[k] + epsilon));
				}
			}
		}
	}

	//set bias ( convolution betas + batchnorm weights )
	float* bias = new float[K];

	for (int k = 0; k < K; ++k) {
		bias[k] = beta[k] - ((gamma[k] * mean[k]) / (sqrtf(variance[k] + epsilon)));
	}

	std::vector<float> output(N * K * P * Q);

	//convolution + batchnormalization
	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			for (int p = 0; p < P; ++p) { //image_row
				for (int q = 0; q < Q; ++q) { //image_column
					float sum = 0.0f;
					for (int c = 0; c < C; ++c) {
						for (int kh = 0; kh < kH; ++kh) {//kernel_height
							int input_h_index = p * stride + kh - padding;
							if (input_h_index >= 0 && input_h_index < H) {
								for (int kw = 0; kw < kW; ++kw) { //kernel_width
									int input_w_index = q * stride + kw - padding;
									if (input_w_index >= 0 && input_w_index < W) {
										int input_index = n * C * H * W + c * H * W + input_h_index * W + input_w_index;
										int weight_index = k * C * kH * kW + c * kH * kW + kh * kW + kw;
										float s = weight[weight_index] * input[input_index];
										sum += s;
									}
								}
							}
						}
					}
					int output_index = n * K * P * Q + k * P * Q + p * Q + q;
					sum += bias[k];
					if (sum <= 0.0f) { output[output_index] = 0.0f; }
					else { output[output_index] = sum; }
				}
			}
		}
	}

	delete[] weight;
	delete[] bias;

	return output;

}

std::vector<float> avg_pool(std::vector<float> input,
	const int& N,
	const int& C,
	const int& H,
	const int& W,
	const int& kH,
	const int& kW,
	const int& padding,
	const int& stride)
{

	int P = ((H + 2 * padding - kH) / stride) + 1;
	int Q = ((W + 2 * padding - kW) / stride) + 1;
	const int M = kH * kW;

	std::vector<float> output(N*C*P*Q);

	//avg_pool
	for (int n = 0; n < N; ++n)
	{
		for (int c = 0; c < C; ++c)
		{
			for (int p = 0; p < P; ++p)
			{
				for (int q = 0; q < Q; ++q)
				{
					float sum = 0.0f;
					for (int kh = 0; kh < kH; ++kh)
					{
						int h_idx = p * stride + kh - padding;
						if (h_idx >= 0 && h_idx < H)
						{
							for (int kw = 0; kw < kW; ++kw)
							{
								int w_idx = q * stride + kw - padding;
								if (w_idx >= 0 && w_idx < W)
								{
									int index = n * C * H * W + c * H * W + h_idx * W + w_idx;
									sum += input[index];
								}
							}
						}
					}
					int output_index = n * C * P * Q + c * P * Q + p * Q + q;
					output[output_index] = sum / M;
					//output.emplace_back(sum);
				}
			}
		}
	}

	return output;

}



std::vector<float> log_softmax(std::vector<float> input, const int& N, const int& C)
{
	int Length = (int)input.size();
	std::vector<float> output(Length);
	// N : batch, C : classes
	for (int i = 0; i < Length; ++i) {
		int p = i / C;
		float sum = (float)0.0f;
		for (int c = 0; c < C; ++c) {
			float element = input[p * C + c];
			float element_exponential = expf(element);
			sum += element_exponential;
		}

		output[i] = logf(expf(input[i]) / sum);
	}

	return output;
}

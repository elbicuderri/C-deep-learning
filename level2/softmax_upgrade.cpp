#include <iostream>
#include <fstream>
#include <vector>

struct AllocationMetrics
{
	size_t TotalAllocated = 0;
	size_t TotalFreed = 0;

	size_t CurrentUsage() { return TotalAllocated - TotalFreed; }
};

static AllocationMetrics s_AllocationMetrics;

void* operator new(size_t size)
{

	s_AllocationMetrics.TotalAllocated += size;

	return malloc(size);

}

void operator delete(void* memory, size_t size)
{

	s_AllocationMetrics.TotalFreed += size;

	free(memory);
}

static void PrintMemoryUsage()
{
	std::cout << "Memory Usage: " << s_AllocationMetrics.CurrentUsage()
		<< " byte" << std::endl;
}

std::vector<float> Dense(const std::vector<float>& Input,
	const std::vector<float>& Weight, const std::vector<float>& Bias)
{
	const int K = (int)Bias.size();
	const int C = (int)Weight.size() / K;
	const int N = (int)Input.size() / C;

	std::vector<float> Output(N * K);

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = (float)0.0f;
			for (int c = 0; c < C; c++) {
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


std::vector<float> ReluDense(const std::vector<float>& Input,
	const std::vector<float>& Weight, const std::vector<float>& Bias)
{
	const int K = (int)Bias.size();
	const int C = (int)Weight.size() / K;
	const int N = (int)Input.size() / C;

	std::vector<float> Output(N * K);

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = (float)0.0f;
			for (int c = 0; c < C; c++) {
				int Input_index = n * C + c;
				int Weight_index = k * C + c;
				float s = Input[Input_index] * Weight[Weight_index];
				sum += s;
			}
			sum += Bias[k];
			if (sum <= (float)0.0f) { sum = (float)0.0f; }
			else {
				int Output_index = n * K + k;
				Output[Output_index] = sum;
			}
		}
	}

	return Output;
}


std::vector<float> Dense_TF(const std::vector<float> Input,
	const std::vector<float> Weight, const std::vector<float> Bias)
{
	const int K = (int)Bias.size();
	const int C = (int)Weight.size() / K;
	const int N = (int)Input.size() / C;

	std::vector<float> Output(N * K);

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = (float)0.0f;
			for (int c = 0; c < C; c++) {
				int Input_index = n * C + c;
				int Weight_index = c * K + k;  // The diff between Torch
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

std::vector<float> ReluDense_TF(const std::vector<float> Input,
	const std::vector<float> Weight, const std::vector<float> Bias)
{
	const int K = (int)Bias.size();
	const int C = (int)Weight.size() / K;
	const int N = (int)Input.size() / C;

	std::vector<float> Output(N * K);

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			float sum = (float)0.0f;
			for (int c = 0; c < C; c++) {
				int Input_index = n * C + c;
				int Weight_index = c * K + k; // The diff between Torch
				float s = Input[Input_index] * Weight[Weight_index];
				sum += s;
			}
			sum += Bias[k];
			if (sum <= (float)0.0f) { sum = (float)0.0f; }
			else {
				int Output_index = n * K + k;
				Output[Output_index] = sum;
			}
		}
	}

	return Output;
}

std::vector<float> SoftmaxV2(const std::vector<float> Tensor, const int& Classes)
{
	const int C = Classes;
	const int OutputSize = (int)Tensor.size();

	std::vector<float> Output(OutputSize);

	for (int i = 0; i < OutputSize; i++) {
		int p = i / C;
		float sum = (float)0.0f;
		for (int c = 0; c < C; c++) {
			float element = Tensor[p * C + c];
			float element_exponential = expf(element);
			sum += element_exponential;
		}
		Output[i] = expf(Tensor[i]) / sum;
	}

	return Output;
}

std::vector<float> SoftmaxV3(const std::vector<float> Tensor, const int& Classes)
{
	const int C = Classes;
	const int OutputSize = (int)Tensor.size();
	const int P = OutputSize / C;

	std::vector<float> Output(OutputSize);

	std::vector<float> ExSum(P);

	for (int p = 0; p < P; p++) {
		float sum = (float)0.0f;
		for (int c = 0; c < C; c++) {
			float element = Tensor[p * C + c];
			float element_exponential = expf(element);
			sum += element_exponential;
		}
		ExSum[p] = sum;
	}

	for (int i = 0; i < OutputSize; i++) {
		int q = i / C;
		Output[i] = expf(Tensor[i]) / ExSum[q];
	}

	return Output;
}


float ExponentialSum(std::vector<float> Tensor, int Classes)
{
	int C = Classes;

	float sum{ (float)0.0f };
	for (int c = 0; c < C; c++) {
		float element = Tensor[c];
		float element_exponential = expf(element);
		sum += element_exponential;
	}

	return sum;
}

std::vector<float> Softmax(std::vector<float> Tensor, int Classes)
{
	int C = Classes;
	int OutputSize = (int)Tensor.size();

	std::vector<float> Output(OutputSize);

	for (int i = 0; i < OutputSize; i++) {
		int p = i / C;
		Output[i] = expf(Tensor[i]) / ExponentialSum(std::vector<float>(Tensor.begin() + p * C,
			Tensor.begin() + (p * C) + C), C);
	}

	return Output;
}


std::vector<float> LoadData(const std::string& FileName)
{
	std::ifstream input(FileName, std::ios::in | std::ios::binary);
	if (!(input.is_open()))
	{
		std::cout << "Cannot open the file!" << std::endl;
		exit(-1);
	}

	std::vector<float> Data;
	input.seekg(0, std::ios::end);
	auto size = input.tellg();
	input.seekg(0, std::ios::beg);

	for (int i = 0; i < size / sizeof(float); i++) {
		float value;
		input.read((char*)&value, sizeof(float));
		Data.emplace_back(value);
	}

	return Data;
}


//PyTorch Dense Weight shape --> (out_channel, in_channel)
int main()
{
	{
	int N = 100; // 0 ~ Nth mnist test images

	const std::vector<float> W1{ LoadData("weight_torch/W1_torch.wts") }; // 784 * 16
	const std::vector<float> b1{ LoadData("weight_torch/b1_torch.wts") }; // 16

	const std::vector<float> W2{ LoadData("weight_torch/W2_torch.wts") }; // 16 * 16
	const std::vector<float> b2{ LoadData("weight_torch/b2_torch.wts") }; // 16 

	const std::vector<float> W3{ LoadData("weight_torch/W3_torch.wts") }; // 16 * 16
	const std::vector<float> b3{ LoadData("weight_torch/b3_torch.wts") }; // 16 

	const std::vector<float> W4{ LoadData("weight_torch/W4_torch.wts") }; // 16 * 16
	const std::vector<float> b4{ LoadData("weight_torch/b4_torch.wts") }; // 16 

	const std::vector<float> W5{ LoadData("weight_torch/W5_torch.wts") }; // 16 * 16
	const std::vector<float> b5{ LoadData("weight_torch/b5_torch.wts") }; // 16 

	const std::vector<float> W6{ LoadData("weight_torch/W6_torch.wts") }; // 16 * 10
	const std::vector<float> b6{ LoadData("weight_torch/b6_torch.wts") }; // 10

	std::vector<float> image{ LoadData("data/mnist_test_float32.bin") }; // 10000 mnist test images

	auto N_image = std::vector<float>(image.begin(), image.begin() + 784 * N); // 0 ~ Nth mnist test images

	for (int i = 0; i < 784 * N; ++i) {
		N_image[i] /= (float)255.0f;
	}

	std::vector<float> relu_dense1 = ReluDense(N_image, W1, b1);
	std::vector<float> relu_dense2 = ReluDense(relu_dense1, W2, b2);
	std::vector<float> relu_dense3 = ReluDense(relu_dense2, W3, b3);
	std::vector<float> relu_dense4 = ReluDense(relu_dense3, W4, b4);
	std::vector<float> relu_dense5 = ReluDense(relu_dense4, W5, b5);
	std::vector<float> dense6 = Dense(relu_dense5, W6, b6);
	std::vector<float> logit = SoftmaxV2(dense6, 10);

	std::vector<float> logit_origin{ LoadData("value_torch/logit_torch.layers") }; //logit layer

	for (int i = 0; i < 10 * N; i++) {
		if (i % 10 == 0 && i != 0) {
			printf("\n");
		}
		printf("my value: %f, origin value: %f, difference: %f \n",
			logit[i], logit_origin[i], fabsf(logit[i] - logit_origin[i]));
	}

	std::vector<float> relu_dense4_origin{ LoadData("value_torch/relu_dense4_torch.layers") }; //middle layer

	for (int i = 0; i < 16 * N; i++) {
		if (i % 10 == 0 && i != 0) {
			printf("\n");
		}
		printf("my value: %f, origin value: %f, difference: %f \n",
			relu_dense4[i], relu_dense4_origin[i], fabsf(relu_dense4[i] - relu_dense4_origin[i]));
	}

	}
	PrintMemoryUsage();

	return 0;
}


//TensorFlow Dense Weight shape --> (in_channel, out_channel)
//int main()
//{
//	const int N = 100;
//
//	auto W1{ LoadData("weight_tf/W1_tf.wts") };
//	auto b1{ LoadData("weight_tf/b1_tf.wts") };
//
//	auto W2{ LoadData("weight_tf/W2_tf.wts") };
//	auto b2{ LoadData("weight_tf/b2_tf.wts") };
//
//	auto W3{ LoadData("weight_tf/W3_tf.wts") };
//	auto b3{ LoadData("weight_tf/b3_tf.wts") };
//
//	auto W4{ LoadData("weight_tf/W4_tf.wts") };
//	auto b4{ LoadData("weight_tf/b4_tf.wts") };
//
//	auto W5{ LoadData("weight_tf/W5_tf.wts") };
//	auto b5{ LoadData("weight_tf/b5_tf.wts") };
//
//	auto W6{ LoadData("weight_tf/W6_tf.wts") };
//	auto b6{ LoadData("weight_tf/b6_tf.wts") };
//
//	auto image{ LoadData("data/mnist_test_float32.bin") };
//
//	auto N_image = std::vector<float>(image.begin(), image.begin() + 784 * N);
//
//	for (int i = 0; i < 784 * N; ++i) {
//		N_image[i] /= (float)255.0f;
//	}
//
//	auto relu_dense1 = ReluDense_TF(N_image, W1, b1);
//	auto relu_dense2 = ReluDense_TF(relu_dense1, W2, b2);
//	auto relu_dense3 = ReluDense_TF(relu_dense2, W3, b3);
//	auto relu_dense4 = ReluDense_TF(relu_dense3, W4, b4);
//	auto relu_dense5 = ReluDense_TF(relu_dense4, W5, b5);
//	auto dense6 = Dense_TF(relu_dense5, W6, b6);
//	auto logit = SoftmaxV2(dense6, 10);
//
//	auto logit_origin{ LoadData("value_tf/logit_tf.layers") };
//
//	for (int i = 0; i < 10 * N; i++) {
//		if (i % 10 == 0) {
//			printf("\n");
//		}
//		printf("my value: %f, origin value: %f, difference: %f \n",
//logit[i], logit_origin[i], fabsf(logit[i] - logit_origin[i]));
//	}
//
//	auto relu_dense3_origin{ LoadData("value_tf/relu_dense3_tf.layers") };
//		
//	for (int i = 0; i < 16 * N; i++) {
//		if (i % 10 == 0) {
//			printf("\n");
//		}
//		printf("my value: %f, origin value: %f, difference: %f \n",
//relu_dense3[i], relu_dense3_origin[i], fabsf(relu_dense3[i] - relu_dense3_origin[i]));
//	}
//
//	return 0;
//}
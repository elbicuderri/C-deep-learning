#include <iostream>
#include <vector>
#include <random>

template<typename T>
void rand_initialize(std::vector<T>& Tensor)
{
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_int_distribution<int> dis(0, Tensor.size());

	for (auto& e : Tensor) {
		e = static_cast<T>(dis(gen));
	}
}

template<typename T>
void print_vector(const std::vector<T>& Tensor)
{
	for (auto& e : Tensor) {
		std::cout << e << std::endl;
	}
}

int main()
{
	std::vector<float> input(9);
	std::vector<float> kernel(4);
	std::vector<float> output(4);

	rand_initialize<float>(input);
	rand_initialize<float>(kernel);
	rand_initialize<float>(output);

	print_vector<float>(input);


	for (int h = 0; h < 3; ++h) {
		for (int w = 0; w < 3; ++w) {
			//OneIdx input_index(0, 0, h+1, w+1);
			//int idx = input_index.idx;
			//std::cout << idx << std::endl;
			int idx = h * 3 + w;
			std::cout << input[idx] << " ";
		}
		std::cout << "\n";
	}

	

	std::cout << "\n";

	for (int h = 0; h < 2; ++h) {
		for (int w = 0; w < 2; ++w) {
			int idx = h * 2 + w;
			std::cout << kernel[idx] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "\n";

	for (int p = 0; p < 2; ++p) {
		for (int q = 0; q < 2; ++q) {
			int sum = 0;
			for (int kh = 0; kh < 2; ++kh) {
				for (int kw = 0; kw < 2; ++kw) {
					int kernel_idx = kh * 2 + kw;
					int input_idx = (p + kh) * 3 + q + kw;
					std::cout << "( " << kernel_idx << ", " << input_idx << " )" << std::endl;
					sum += input[input_idx] * kernel[kernel_idx];
				}
			}
			int output_idx = p * 2 + q;
			output[output_idx] = sum;
		}
	}

	std::cout << "\n";

	for (int h = 0; h < 2; ++h) {
		for (int w = 0; w < 2; ++w) {
			int idx = h * 2 + w;
			std::cout << output[idx] << " ";
		}
		std::cout << "\n";
	}


	return 0;
}
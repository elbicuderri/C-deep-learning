#include <iostream>
#include <vector>
#include <random>


// for one-D index struct
typedef struct OneDimIndex {

	int N, C, H, W;
	int idx;

	OneDimIndex(int n, int c, int h, int w)
		:
		N(n-1),
		C(c-1),
		H(h-1),
		W(w-1)
		{
		idx = n * C * H * W + c * H * W + h * W + w - 1;
		}

} OneIdx;

class IndexTensor {
public:
	IndexTensor(std::vector<int>& InTensor): Tensor(InTensor) {}
	~IndexTensor() {}

	//int OneDimIndex() const {}

private:
	std::vector<int> Tensor;
};

int main()
{
	// 시드값을 얻기 위한 random_device 생성.
	std::random_device rd;

	// random_device 를 통해 난수 생성 엔진을 초기화 한다.
	std::mt19937 gen(rd());

	// 0 부터 9 까지 균등하게 나타나는 난수열을 생성하기 위해 균등 분포 정의.
	std::uniform_int_distribution<int> dis1(0, 9);
	std::uniform_int_distribution<int> dis2(0, 9);


	std::vector<int> input(9);
	std::vector<int> kernel(4);
	std::vector<int> output(4);

	//OneIdx input_one_idx(0, 0, 1, 1);

	//auto in_id = input_one_idx.idx;

	//std::cout << in_id << std::endl << "\n";

	for (auto& e1 : input) {
		e1 = dis1(gen);
	}

	for (auto& e2 : kernel) {
		e2 = dis2(gen);
	}

	
	
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
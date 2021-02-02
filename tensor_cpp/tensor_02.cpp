#include <iostream>
#include <vector>
#include <random>

class Tensor {
public:
	Tensor() {}
	Tensor(std::vector<int>& InTensor) : InTensor(InTensor) {}
	//Tensor(std::vector<int> InTensor) : InTensor(InTensor) {}
	~Tensor() {}

	std::vector<int> operator+ (std::vector<int>& a) 
	{ 
		std::vector<int> re(static_cast<int>(InTensor.size()));

		for (int i = 0; i < static_cast<int>(InTensor.size()); ++i) {
			re[i] = InTensor[i] + a[i];
		}

		return re;
	}

	int shape() { return static_cast<int>(InTensor.size()); }

	//int OneDimIndex() const {}

private:
	std::vector<int> InTensor;
	//int shape;
};

int main()
{
	std::vector<int> a{ 1, 2, 3, 4 };

	std::vector<int> b{ 1, 2, 3, 4 };

	Tensor A(a);

	Tensor B(b);

	auto c = A + b;

	for (auto& e : c) {
		std::cout << e << std::endl;
	}

	//std::cout << A.shape() << std::endl;


	return 0;
}
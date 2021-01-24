# Deep-Learning-with-C-cuDNN-CUDA-IMPLEMENTATION

## Convolution Backpropragation Implementation

[Convolutional Neural Networks: Step by Step](https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html)

**Setup Visual Studio Code for Multi-File C++ Projects**
<br>
➡[Link](https://dev.to/talhabalaj/setup-visual-studio-code-for-multi-file-c-projects-1jpi#setup3)
---

[CUDA programming example](https://cuda.readthedocs.io/ko/latest/)

[cuda-neural-network-implementation](https://luniak.io/cuda-neural-network-implementation-part-1/#programming-model)

```cuda
//CUDA kernel sample code for maxpooling2d
__global__ void maxpooling_kernel(float *output, float *input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, 
	int stride_height, int stride_width, int total_size)
{
	int N = batch;
	int C = channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	//tid : thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= total_size)
		return;

	//q_idx : output w-index
	int q_idx = tid % Q;
	int idx = tid / Q;

	//p_idx : output h-index
	int p_idx = idx % P;
	idx /= P;

	//k_idx : output channel-index
	int k_idx = idx % C;

	//n_idx : output batch-index
	int n_idx = idx / C;

	//output(n_idx, k_idx, p_idx, q_idx)

	float max = -FLT_MAX;
	for (int kh = 0; kh < kH; kh++) {
		int h_idx = p_idx * sH + kh - pH;
		if (h_idx >= 0 && h_idx < H) {
			for (int kw = 0; kw < kW; kw++) {
				int w_idx = q_idx * sW + kw - pW;
				if (w_idx >= 0 && w_idx < W) {
					int input_index = n_idx * C * H * W + k_idx * H * W + h_idx * W + w_idx;
					if (input[input_index] > max) {
						max = input[input_index];
					}
				}
			}
		}
	}
	output[tid] = max;

}


void maxpooling(float *output, float *input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, 
	int stride_height, int stride_width)
{
	int N = batch;
	int C = channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = (H + 2 * pH - kH) / sH + 1;
	int Q = (W + 2 * pW - kW) / sW + 1;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * C * P * Q;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	maxpooling_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (output, input, 
	N, C, H, W, kH, kW, pH, pW, sH, sW, TOTAL_SIZE);

}
```

```cpp
//cpp code for maxpooling2d
void maxpooling(float *output, float *input, int batch, int channel, 
int input_height, int input_width,
int kernel_height, int kernel_width, 
int pad_top, int pad_bottom, int pad_left, int pad_right, 
int stride_height, int stride_width)
{
	int N = batch;
	int C = channel;
	int H = input_height;
	int W = input_width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pT = pad_top;
	int pB = pad_bottom;
	int pL = pad_left;
	int pR = pad_right;
	int sH = stride_height;
	int sW = stride_width;
	int P = ((input_height + pad_top + pad_bottom - kernel_height) / stride_height) + 1;
	int Q = ((input_width + pad_left + pad_right - kernel_width) / stride_width) + 1;

	//maxpooling
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int p = 0; p < P; p++) {
				for (int q = 0; q < Q; q++) {
					float max = -FLT_MAX;
					for (int kh = 0; kh < kH; kh++) {
						int h_idx = p * sH + kh - pT;
						if (h_idx >= 0 && h_idx < H) {
							for (int kw = 0; kw < kW; kw++) {
								int w_idx = q * sW + kw - pL;
								if (w_idx >= 0 && w_idx < W) {
									int index = n * C * H * W + c * H * W + h_idx * W + w_idx;
									if (input[index] > max) {
										max = input[index];
									}
								}
							}
						}
					}
					int output_index = n * C * P * Q + c * P * Q + p * Q + q;
					output[output_index] = max;
				}
			}
		}
	}

}
```

```cpp
//cuDNN code for maxpooling2d

cudnnPoolingDescriptor_t poolingDesc;

cudnnPoolingMode_t pooling_mode = CUDNN_POOLING_MAX;

cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;

cudnnCreatePoolingDescriptor(&poolingDesc);

cudnnSetPooling2dDescriptor(
	poolingDesc,
	pooling_mode,
	maxpoolingNanOpt,
	maxpool_kH,
	maxpool_kW,
	maxpool_pH,
	maxpool_pW,
	maxpool_sH,
	maxpool_sW);

cudnnTensorDescriptor_t maxpoolDesc;

cudnnCreateTensorDescriptor(&maxpoolDesc);

cudnnSetTensor4dDescriptor(
	maxpoolDesc,
	dataformat,
	datatype,
	N,
	K,
	maxpool_H,
	maxpool_W);

const float alpha_pool = 1.0f;
const float beta_pool = 0.0f;

void *d_maxpool;
cudaMalloc((void **)&d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float)); // cuda allocate 

cudnnPoolingForward(
	HANDLE,
	poolingDesc,
	&alpha_pool,
	conv_imageDesc,
	d_conv_fusion,
	&beta_pool,
	maxpoolDesc,
	d_maxpool);

float* h_maxpool = (float*)malloc(N * K * maxpool_H * maxpool_W * sizeof(float));
cudaMemcpy(h_maxpool, d_maxpool, N * K * maxpool_H * maxpool_W * sizeof(float), cudaMemcpyDeviceToHost);
```

```bash
$ export PATH=/usr/local/cuda-10.0/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/TensorRT-7.0.0.11/lib:$LD_LIBRARY_PATH
```

```bash
$ nvcc main.cpp -o main
$ ./main
```

```bash
$ nvcc main.cpp -c
$ nvcc main.o -o main -L/usr/local/cuda-10.0/lib64 -lcudnn -I/usr/local/cuda-10.0/include
$ ./main
```

```makefile
# makefile
NVCC = nvcc
TARGET = mnist_cudnn
OBJECTS = mnist_cudnn.o
OBJECTS_CU = mnist_cudnn.cpp
CUDA_PATH = /usr/local/cuda/include
CUDNN_PATH = /usr/local/cuda/lib64

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(NVCC) -o $@ $^ -I$(CUDA_PATH) -lcudnn

$(OBJECTS) : $(OBJECTS_CU)
	$(NVCC) -c $^

clean :
	rm -rf $(OBJECTS) $(TARGET)
```


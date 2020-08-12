__global__ void relu_kernel(float *output, float *input, int batch, int channel, int height, int width, int total_size)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	if (input[tid] > 0.0f) {
		output[tid] = input[tid];
	}
	else {
		output[tid] = 0.0f;
	}
}

void relu(float *output, float *input, int batch, int channel, int height, int width)
{
	int N = batch;
	int C = channel;
	int H = height;
	int W = width;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * C * H * W;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	relu_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (output, input, N, C, H, W, TOTAL_SIZE);
}

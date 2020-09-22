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
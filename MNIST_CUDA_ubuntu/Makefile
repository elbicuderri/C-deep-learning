NVCC = nvcc
TARGET = mnist
OBJECTS = conv_batchnorm_fusion.o maxpooling.o relu.o dense.o softmax.o main.o
OBJECTS_CU = conv_batchnorm_fusion.cu maxpooling.cu relu.cu dense.cu softmax.cu main.cpp
CUDA_PATH = /usr/local/cuda/include

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(NVCC) -o $@ $^ -I$(CUDA_PATH) 

$(OBJECTS) : $(OBJECTS_CU)
	$(NVCC) -c $^

clean :
	rm -rf $(OBJECTS) $(TARGET)


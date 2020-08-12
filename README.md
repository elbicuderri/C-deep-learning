# Deep-Learning-with-CUDA


(28, 28, 1) -> conv -> (28, 28, 5) -> batchnorm -> maxpool -> (14, 14, 5) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)

mnist_pytorch.c : NCHW convolution


(1, 28, 28) -> conv -> (5, 28, 28) -> batchnorm -> maxpool -> (5, 14, 14) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)

MNIST_tfone.py : you can get weights and every layer outputs of tf-model

MNIST_torch.py : you can get weights and every layer outputs of torch-model

MNIST_C_tf.c : you can run NHWC format CNN with weight files and compare results to original value files.

MNIST_C_torch.c : you can run NCHW format CNN with weight files and compare results to original value files.


MNIST_CUDNN_window : In Visual Studio, CUDNN style CNN code.

MNIST_CUDA_window : In Visual Studio, CUDA style CNN code.

MNIST_CUDA_ubuntu : In ubuntu, make.

As a result,

1. Yon can make and get weight files and values from each layers of mnist_test file by running .py files.

2. You can run C, CUDNN, CUDA code with using weight files and compare them with original value files.




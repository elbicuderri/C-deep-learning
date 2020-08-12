# Deep-Learning-with-CUDA


(28, 28, 1) -> conv -> (28, 28, 5) -> batchnorm -> maxpool -> (14, 14, 5) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)

mnist_pytorch.c : NCHW convolution


(1, 28, 28) -> conv -> (5, 28, 28) -> batchnorm -> maxpool -> (5, 14, 14) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)

mnist_tfone.py : you can get weights and every layer outputs of tf-model

mnist_torch.py : you can get weights and every layer outputs of torch-model


1. Yon can make and get weight files and values from each layers of mnist_test by running .py files.

2. You can run C, CUDNN, CUDA code with using weight files and compare them with original value files.




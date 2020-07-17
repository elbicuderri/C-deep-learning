# Deep-Learning-with-C

convolution2D.c : code for convolution in 2D

mnist_tensorflow.c : NHWC convolution

mnist_pytorch.c : NCHW convolution

(28, 28, 1) -> conv -> (28, 28, 5) -> batchnorm -> maxpool -> (14, 14, 5) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)
(1, 28, 28)            (5, 28, 28)                            (5, 14, 14)    

mnist_tfone.py : you can get weights and every layer outputs of tf-model

mnist_torch.py : you can get weights and every layer outputs of torch-model




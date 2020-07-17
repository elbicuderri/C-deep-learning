# Deep-Learning-with-C

convolution2D.c : code for convolution in 2D

mnist_tensorflow : NHWC convolution

mnist_pytorch : NCHW convolution


(28, 28, 1) -> conv -> (28, 28, 5) -> batchnorm -> maxpool -> (14, 14, 5) -> flatten -> (980, ) -> dense -> (120, ) -> dense -> (10,)
(1, 28, 28)            (5, 28, 28)                            (5, 14, 14)                 


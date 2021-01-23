import sys
import struct
from tensorflow.keras.models import load_model
import numpy as np

"""
Point!!!!

FC직전에 flatten을 한 경우 TF dense weight shape (kH, kW, C, K)

Torch dense weight shape (K, C, kH, kW)

따라서 이 dense weight shape를 바꾸고 싶은 경우에는

w = w.transpose(3, 2, 0, 1) 를 해주면 된다.

보통의 dense layer에서는

 w = w.transpose(1, 0) 만 하면 된다.

"""
model_path = 'model/mnist_model.h5'
# weights_path = 'weights/mnist_float32.wts'

model = load_model(model_path)
# model.summary()

weights = model.get_weights()

weights_list = ['conv1filter', 'conv1bias',
               'ip1filter', 'ip1bias',
               'ip2filter', 'ip2bias']

w = weights[0]
w = w.transpose(3, 2, 0, 1)

print(w)


# for i in range(6):
#     weights_path = f'weights/{weights_list[i]}_1103.wts'
#     with open(weights_path, 'wb') as f:
#         w = weights[i]
#         if w.ndim == 4:
#             w = w.transpose(3, 2, 0, 1)
#         elif w.ndim == 3:
#             w = w.transpose(2, 1, 0)
#         elif w.ndim == 2:
#             w = w.transpose(1, 0)
#         print(weights_list[i], ': ', w.shape)
#         w = w.astype('float32')
#         w.tofile(f, sep=' ')

print('finished')


import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization, Dropout
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshaping X data: (n, 28, 28) => (n, 28, 28, 1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)).astype("float32") / 255.0

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("data ready")

def mnist_model():
    inputs = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False, name="conv")(inputs)
    batchnorm = BatchNormalization(epsilon=0.001, name="batchnorm")(conv)
    maxpool = MaxPooling2D(pool_size=(2, 2), name="maxpool")(batchnorm)
    relu_maxpool = Activation('relu',name="relu_maxpool")(maxpool)
    flatten = Flatten(name="flatten")(relu_maxpool)
    dense1 = Dense(120, name="dense1")(flatten)
    relu_dense1 = Activation('relu', name="relu_dense1")(dense1)
    dense2 = Dense(10, name="dense2")(relu_dense1)
    outputs = Activation('softmax', name="result")(dense2)

    model = Model(inputs, outputs)

    optimizer = optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = mnist_model()

model.summary()

history = model.fit(X_train, y_train, batch_size=100, epochs=5, verbose=2)

# print(model.layers[1])
#
# print(model.layers[1].get_weights())

def save_values(value, name):
    value.tofile(f"weights_tfone/{name}_tfone_2.bin")

layer_name_list = ["conv", "batchnorm", "maxpool", "relu_maxpool", "flatten", "dense1", "relu_dense1", "dense2", "result"]

intermediate_output_list = []
for n in layer_name_list:
    intermidiate_layer_model = Model(inputs=model.input, outputs=model.get_layer(n).output)
    intermidiate_output = intermidiate_layer_model.predict(X_test)
    save_values(intermidiate_output, n)
    print(intermidiate_output.shape)
    print(f"{n} is saved.")

print("=======================================================================")

weights = model.get_weights()

name_list = ["kernel",
             "gamma", "beta",
             "mean", "variance",
             "W2", "b2",
             "W3", "b3"]

def save_weights(weight, name):
    weight.tofile(f"weights_tfone/{name}_tfone_2.bin")

for w, name in zip(weights, name_list):
    print(w.shape)
    save_weights(w, name)
    print(f"{name} is saved.")

print("Finished!")

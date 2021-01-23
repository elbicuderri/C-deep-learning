from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)).astype("float32") / 255.0

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("data ready")

def cifar10_model():
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False)(inputs)
    batchnorm1 = BatchNormalization(epsilon=0.001)(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(batchnorm1)
    relu_maxpool1 = Activation('relu')(maxpool1)
    conv2 = Conv2D(filters=40, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False)(relu_maxpool1)
    batchnorm2 = BatchNormalization(epsilon=0.001)(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(batchnorm2)
    relu_maxpool2 = Activation('relu')(maxpool2)
    conv3 = Conv2D(filters=80, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False)(relu_maxpool2)
    batchnorm3 = BatchNormalization(epsilon=0.001)(conv3)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(batchnorm3)
    relu_maxpool3 = Activation('relu')(maxpool3)
    conv4 = Conv2D(filters=160, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False, name="conv")(relu_maxpool3)
    batchnorm4 = BatchNormalization(epsilon=0.001, name="batchnorm")(conv4)
    maxpool4 = MaxPooling2D(pool_size=(2, 2), name="maxpool")(batchnorm4)
    relu_maxpool4 = Activation('relu',name="relu_maxpool")(maxpool4)
    flatten = Flatten(name="flatten")(relu_maxpool4)
    dense1 = Dense(320, name="dense1")(flatten)
    relu_dense1 = Activation('relu', name="relu_dense1")(dense1)
    dense2 = Dense(320)(relu_dense1)
    relu_dense2 = Activation('relu')(dense2)
    dense3 = Dense(10, name="dense2")(relu_dense2)
    outputs = Activation('softmax', name="result")(dense3)
    model = Model(inputs, outputs)

    return model

model = cifar10_model()

model.summary()

optimizer = Adam(lr=0.001)

checkpoint_path = 'cifar10_checkpoint.ckpt'

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',
                             save_weights_only=True,
                             save_best_only=True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.8,
                              patience=5,
                              verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128,
                    validation_data=(X_test, y_test), epochs=100, callbacks=[checkpoint, reduce_lr, earlystopping], verbose=2)

model.load_weights(checkpoint_path)

model.save("cifar10_model.h5")

# def save_values(value, name):
#     value.tofile(f"value/{name}_tf.bin")
#
# layer_name_list = ["conv", "batchnorm", "maxpool", "relu_maxpool",
#                    "flatten", "dense1", "relu_dense1",
#                    "dense2", "result"]
#
# intermediate_output_list = []
# for n in layer_name_list:
#     intermidiate_layer_model = Model(inputs=model.input, outputs=model.get_layer(n).output)
#     intermidiate_output = intermidiate_layer_model.predict(X_test)
#     save_values(intermidiate_output, n)
#     print(n, ": ", intermidiate_output.shape)
#
# print("=======================================================================")
#
# weights = model.get_weights()
#
# name_list = ["kernel",
#              "gamma", "beta",
#              "mean", "variance",
#              "W1", "b1",
#              "W2", "b2"]
#
# def save_weights(weight, name):
#     weight.tofile(f"weight/{name}_tf.bin")
#
# for w, name in zip(weights, name_list):
#     save_weights(w, name)
#     print(name, ": ", w.shape)

print("Finished!")

import pandas as pd
import numpy as np

df = pd.read_csv("data/mnist_test.csv")

print(df.head())

label = df.iloc[:, 0]
image = df.iloc[:, 1:]

label = np.array(label).astype("int32")
image = np.array(image).astype("float32")
image = image.flatten()

print(type(label[0]))
print(type(image[0]))

print(label[:30])

image.tofile("data/mnist_test_images_float32.bin")
label.tofile("data/mnist_test_labels_int32.bin")


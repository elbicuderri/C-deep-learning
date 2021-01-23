import numpy as np
import pandas as pd

df = pd.read_csv("mnist_test.csv", delimiter=",", header=None)

print(df.head(5), type(df), df.shape, df.dtypes)

df = df.astype("float32")

print(df.dtypes)

df = np.array(df)

df.tofile("mnist_test_float.bin")
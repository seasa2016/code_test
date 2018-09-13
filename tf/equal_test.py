import tensorflow as tf
import os
import numpy as np

predictions = np.array([[[[0],[0]],[[1],[1]]],[[[1],[1]],[[1],[1]]]])
targets = np.array([[[[0],[0]],[[0],[1]]],[[[1],[0]],[[0],[1]]]])
expected = np.mean(np.prod((predictions == targets).astype(float), axis=(1,2)))


print(np.prod((predictions == targets).astype(float)))
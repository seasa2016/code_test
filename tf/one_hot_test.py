import tensorflow as tf
import numpy as np
import os,
a = tf.one_hot(np.array([1,2,3]),depth = 4)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
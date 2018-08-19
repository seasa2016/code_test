import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

t = tf.constant([[[1],
                  [2],
                  [3]]])
# 'constant_values' is 0.
# rank of 't' is 2.
t1 = tf.pad(t, [[0, 0],
        [0, 5],
        [0, 0]])  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t1))
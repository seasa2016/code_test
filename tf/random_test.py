import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


random_indx = tf.random_uniform([1], minval=1, maxval=5, dtype=tf.int32)
qq =tf.Print(random_indx,[random_indx])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(qq)
                
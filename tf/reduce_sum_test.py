import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

a = tf.constant([1,2,3])
b = tf.reduce_sum(a,axis = -1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

x = tf.constant(1)
y = tf.constant(2)

result1 = tf.cond(x < y, lambda: tf.add(x, y), lambda: tf.square(y))

result2 = tf.cond(x > y, lambda: tf.add(x, y), lambda: tf.square(y))

with tf.Session() as sess:
    print(sess.run([x,y,result1,result2]))
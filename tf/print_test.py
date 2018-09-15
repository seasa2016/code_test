import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

a = tf.constant(1)
n = tf.constant(10)
a = a + n

def cond(a,n):
    return a<n

def body(a,n):
    a = a + 1
    a = tf.Print(a,[a])
    return a,n 

a,n = tf.while_loop(cond,body,[a,n])
a = tf.Print(a,[a])


with tf.Session() as sess:
    sess.run([a,n])
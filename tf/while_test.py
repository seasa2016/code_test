import tensorflow as tf 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

a = tf.get_variable("ii", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
n = tf.constant(10)

def cond(a, n):
    return  a< n
def body(a, n):
    a = a + 1
    return a, n

a, n = tf.while_loop(cond, body, [a, n])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run([a, n])
    print(res)
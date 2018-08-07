import tensorflow as tf 
import numpy as np 

indices = np.array([[0, 2], [4, 1]])

print(indices.shape)

depth = 4
a=tf.one_hot(indices, depth,on_value=5.0, off_value=0.0,axis=-1)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(a).shape)
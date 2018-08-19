import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


temp = [ _ for _ in range(256)]

a = tf.train.shuffle_batch([temp],batch_size=4,num_threads=4,capacity=500,min_after_dequeue=10)


#arr = tf.sparse_tensor_to_dense([[0,1],[1,1]])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
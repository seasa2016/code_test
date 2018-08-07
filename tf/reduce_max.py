import tensorflow as tf

arr = tf.random_normal([2,5,1,4])
q1 = tf.reduce_max(arr, reduction_indices=[1], keep_dims=True)
q2 = tf.squeeze(q1, [1, 2])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(arr))
    print('-'*10)
    print(sess.run(q1))
    print('-'*10)
    print(sess.run(q2))
    




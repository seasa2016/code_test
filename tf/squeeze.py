import tensorflow as tf

t = tf.constant([[[[1,2,3]],[[4,5,6]]]])
a = tf.concat([t,t],1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t).shape)
    print(sess.run(t))
    print(sess.run(a).shape)
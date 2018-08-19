import tensorflow as tf 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

char_size=3
embedding=2

init = tf.contrib.layers.xavier_initializer(uniform=False)
embedding_matrix = tf.get_variable('char_embedding', [char_size, embedding], initializer=init)
cnn_x = tf.nn.embedding_lookup(embedding_matrix, [[1,2]])
cnn_x1 = tf.expand_dims(cnn_x, -1)

cnn_x2 = tf.one_hot([1,2], depth=char_size)
cnn_x2 = tf.expand_dims(cnn_x2, -1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embedding_matrix))
    print()
    print(sess.run(cnn_x))
    print('*'*10)
    print(sess.run(cnn_x1))
    print('*'*10)
    print(sess.run(cnn_x2))
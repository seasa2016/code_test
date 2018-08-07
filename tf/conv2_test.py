import tensorflow as tf

cnn_x = tf.ones([2,25,50,1], tf.float32)

conv_layers = [[5,3,-1],[5,5,-1]]

arr=[]
check_point = 0
for i, conv_info in enumerate(conv_layers):
    # conv_info = [# of feature, kernel height, pool height]
    check_point += 1

    with tf.name_scope("Conv-Layer-" + str(i)):
        filter_width = cnn_x.get_shape()[2].value
        filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]  # [각 filter 크기, emb 크기, 1, kernel 크기]

        init = tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.get_variable("Conv_W" + str(i), filter_shape, initializer=init)
        b = tf.get_variable("Conv_b" + str(i), [conv_info[0]], initializer=init)
        
        with tf.device("/gpu:1"):  #GPU1
            conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="conv")

    with tf.name_scope("Non-Linear"):
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
    if conv_info[-1] != -1:
        with tf.name_scope("Max-Polling"):
            pool_shape = [1, conv_info[-1], 1, 1]
            conv = tf.nn.max_pool(conv, ksize=pool_shape, strides=pool_shape, padding="VALID")
    with tf.name_scope("One-Max-Pooling"):
        conv = tf.reduce_max(conv, reduction_indices=[1], keep_dims=True)  # 1-max pooling
        conv = tf.squeeze(conv, [1, 2])
        if i == 0:
            cnn_output = conv
        else:
            cnn_output = tf.concat([cnn_output, conv], 1)
    arr.append(tf.shape(cnn_output))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(arr))
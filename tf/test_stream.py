import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_one_sequence(file_queue):
    """ read one sequence from .tfrecords"""

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)

    feature = tf.parse_single_example(serialized_example, features={
        'encoder_input': tf.VarLenFeature(tf.int64),
        'encoder_input_len': tf.FixedLenFeature([1], tf.int64),
        'decoder_input': tf.VarLenFeature(tf.int64),
        'decoder_input_len': tf.FixedLenFeature([1], tf.int64),
        'detector_la_target': tf.VarLenFeature(tf.int64),
        'detector_ge_target': tf.VarLenFeature(tf.int64),
        'detector_co_target': tf.VarLenFeature(tf.int64)
        #'detector_la_target': tf.FixedLenFeature([1], tf.int64),
        #'detector_ge_target': tf.FixedLenFeature([1], tf.int64),
        #'detector_co_target': tf.FixedLenFeature([1], tf.int64)
    })

    return feature['encoder_input'], feature['encoder_input_len'], \
        feature['decoder_input'], feature['decoder_input_len'], \
        feature['detector_la_target'], feature['detector_ge_target'], \
        feature['detector_co_target']
##
batch_size = 2
capacity = 4

##
def read_batch_sequences():
    file_queue = tf.train.string_input_producer(['./../data/train.tfrecords'])

    ei, ei_len, di, di_len, la, ge, co = read_one_sequence(file_queue)

    min_after_dequeue = 3000
    capacity = min_after_dequeue + 3 * batch_size

    encoder_inputs, encoder_inputs_len, decoder_inputs, decoder_inputs_len, \
    detector_la_target,  detector_ge_target,  detector_co_target = \
        tf.train.shuffle_batch(
            [ei, ei_len, di, di_len, la, ge, co],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

    encoder_inputs = tf.sparse_tensor_to_dense(tf.to_int64(encoder_inputs))
    decoder_inputs = tf.sparse_tensor_to_dense(tf.to_int64(decoder_inputs))
    detector_la_target = tf.sparse_tensor_to_dense(tf.to_int64(detector_la_target))
    detector_ge_target = tf.sparse_tensor_to_dense(tf.to_int64(detector_ge_target))
    detector_co_target = tf.sparse_tensor_to_dense(tf.to_int64(detector_co_target))
        
    encoder_inputs_len = tf.reshape(encoder_inputs_len,
                                    [batch_size])
    decoder_inputs_len = tf.reshape(decoder_inputs_len,
                                    [batch_size])
    
    return encoder_inputs, tf.to_int32(encoder_inputs_len), \
               decoder_inputs, tf.to_int32(decoder_inputs_len), \
               detector_la_target, \
               detector_ge_target, \
               detector_co_target

raw_encoder_inputs, raw_encoder_inputs_len, \
raw_decoder_inputs, raw_decoder_inputs_len, \
raw_detector_la_targets, raw_detector_ge_targets, \
raw_detector_co_targets = read_batch_sequences()


# self.encoder_inputs: [batch_size, max_len]
# self.encoder_inputs = self.raw_encoder_inputs[:, 1:]
# self.encdoer_inputs_len: [batch_size]
encoder_inputs_len = raw_encoder_inputs_len
# Randomly change a word by <UNK>
encoder_inputs = []
for b in range(batch_size):
    an_en_in_ln = encoder_inputs_len[b]
    random_indx = tf.random_uniform([1], minval=1, maxval=an_en_in_ln-1, dtype=tf.int32)[0]
    a_en_in = raw_encoder_inputs[b,1:] # No <GO>
    a_en_in = tf.Print(a_en_in,[a_en_in])
    a_en_in = tf.cond(an_en_in_ln > tf.constant(3), 
                        lambda: tf.concat(axis=0, values=[a_en_in[:random_indx], [3], a_en_in[random_indx+1:]]), 
                        lambda: a_en_in)
    encoder_inputs.append(a_en_in)
"""
encoder_inputs = tf.stack(encoder_inputs)
"""

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    print(sess.run([raw_encoder_inputs,raw_detector_la_targets]))
    coord.request_stop()


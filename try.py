import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from TFCommon.RNNCell import GRUCell
from hjk_tools.Layers import cbhg
import math
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper


input = tf.get_variable(name='input', shape=(2, 3, 4), dtype=tf.float32, initializer=tf.constant_initializer(1))

def self_rnn(input, units=128, layer_num = 2, parallel_iterations=64, name='gru', reuse=False):
    with tf.variable_scope(name_or_scope=name):
        with tf.variable_scope('enc'):
            encoder_rnn = MultiRNNCell([GRUCell(units) for _ in range(layer_num)])
        with tf.variable_scope('dec'):
            decoder_rnn = MultiRNNCell([ResidualWrapper(GRUCell(units)) for _ in range(layer_num)])

        rnn_tot = input.shape[1]
        batch = input.shape[0]

        cond = lambda x, *_: tf.less(x, rnn_tot)

        with tf.variable_scope('pre'):
            cnt = tf.zeros((), dtype=tf.int32)
            encoder_init_state = encoder_rnn.zero_state(batch, dtype=tf.float32)
            decoder_init_state = decoder_rnn.zero_state(batch, dtype=tf.float32)
            res_ta = tf.TensorArray(dtype=tf.float32, size=rnn_tot)
            input_time_major = tf.transpose(input, (1, 0, 2))

        def body(cnt, encoder_pre, decoder_pre, res_ta):
            input = input_time_major[cnt]
            with tf.variable_scope('enc'):
                output_enc, new_enc_state = encoder_rnn(input, encoder_pre)
            with tf.variable_scope('dec'):
                output_dec, new_dec_state = decoder_rnn(output_enc, decoder_pre)
            res_ta = res_ta.write(cnt, output_dec)
            cnt = tf.add(cnt, 1)
            return cnt, new_enc_state, new_dec_state, res_ta


        res_cnt, encoder_res, decoder_res, final_res_ta = tf.while_loop(cond, body, loop_vars=[cnt, encoder_init_state, decoder_init_state, res_ta], parallel_iterations=parallel_iterations)
        # final_res_ta = tf.stack(final_res_ta)
        final_res = final_res_ta.stack()

        return final_res


final_res = self_rnn(input, units=4, layer_num=2)



with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    out1 = sess.run(final_res)
    print(out1)
    print('adfas')

    def test():
        return 1, 2, 3, 4, 5
    a, b, *_ = test()
    print(b)





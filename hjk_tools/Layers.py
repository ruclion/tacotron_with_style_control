import tensorflow as tf
import math
from tensorflow.python.ops import array_ops
from TFCommon.RNNCell import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper


def Reference_embedding(inputs, input_lengths, training=True, channels=[32, 64, 128], gru_unit=128, name='reference_embedding', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        Mel_dim_size = 80
        batch_size = tf.shape(inputs)[0]
        input_time_steps = tf.shape(inputs)[1]
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        mask = tf.expand_dims(
                tf.expand_dims(array_ops.sequence_mask(input_lengths, tf.shape(inputs)[1], tf.float32), axis=-1), axis=-1)
        loop_conv2d = inputs * mask
        for idk, channel in enumerate(channels):
            loop_conv2d = tf.layers.conv2d(loop_conv2d, filters=channel, kernel_size=(3, 3), strides=(2, 2),
                             padding='same', name='conv2d_{}'.format(idk), activation=tf.nn.relu)
            input_lengths = tf.ceil(input_lengths / 2)
            mask = tf.expand_dims(
                tf.expand_dims(array_ops.sequence_mask(input_lengths, tf.shape(loop_conv2d)[1], tf.float32), axis=-1), axis=-1)
            loop_conv2d = loop_conv2d * mask
            loop_conv2d = tf.layers.batch_normalization(loop_conv2d, training=training)
            loop_conv2d = loop_conv2d * mask
            Mel_dim_size = math.ceil(Mel_dim_size / 2)
        loop_conv2d = tf.reshape(loop_conv2d, shape=(batch_size, -1, Mel_dim_size*channels[-1]))
        gru_output = gru(loop_conv2d, gru_unit, sequence_length=input_lengths)
        gru_output = tf.transpose(gru_output, [1, 0, 2])
        output = gru_output[-1]
        output = tf.layers.dense(output, units=gru_unit, activation=tf.nn.tanh, name='after_dense_style_emb')
    return output








#conv1d_bank, include: conv1d+batch_normalization without pooling(watch out: trainning and train_dependency)
def conv1d_bank(inputs, training=True, k=16, bank_filters=128, name='conv1d_bank', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        stacked_conv = tf.concat([tf.layers.conv1d(inputs, filters=bank_filters, kernel_size=idk, strides=1,
                                                   padding='same', name='inner_conv_{}'.format(idk))
                                  for idk in range(1, k + 1)], axis=-1)
        normed_conv = tf.nn.relu(tf.layers.batch_normalization(stacked_conv, training=training))
    return normed_conv

#conv1d_projections, two conv1d, if need mask, then after first conv need to do either.
def conv1d_projections(inputs, training=True, projection_filters=(128, 128), name='conv1d_projections', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv_0 = tf.layers.conv1d(inputs, filters=projection_filters[0], kernel_size=3, strides=1,
                                  padding='same', name='inner_conv_0')
        norm_0 = tf.nn.relu(tf.layers.batch_normalization(conv_0, training=training))
        conv_1 = tf.layers.conv1d(norm_0, filters=projection_filters[1], kernel_size=3, strides=1,
                                  padding='same', name='inner_conv_1')
        norm_1 = tf.layers.batch_normalization(conv_1, training=training)
    return norm_1

#highway_net, use more complex than dese, likely with residual
def highway_net(inputs, layers=4, activation=tf.nn.relu, name='highway_net', reuse=False):
    assert layers >= 1, '[E] "layers" must be a positive integer.'
    with tf.variable_scope(name, reuse=reuse):
        units = inputs.shape[-1].value
        x = inputs
        for layer_id in range(layers):
            with tf.variable_scope('inner_fc_{}'.format(layer_id)):
                h = tf.layers.dense(name='H', inputs=x, units=units, activation=activation)
                t = tf.layers.dense(name='T', inputs=x, units=units, activation=tf.nn.sigmoid)
                y = h * t + x * (1 - t)
                x = y
    return y

#bi_gru, no need to write loop, use gru in tf.common
def bi_gru(inputs, units=128, sequence_length=None, parallel_iterations=64, name='bidirectional_gru', reuse=False):
    print('use his local')
    with tf.variable_scope(name, reuse=reuse):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRUCell(units), cell_bw=GRUCell(units),
                                                     inputs=inputs, sequence_length=sequence_length,
                                                     dtype=inputs.dtype, parallel_iterations=parallel_iterations)
        outputs = tf.concat(outputs, axis=-1)
    return outputs

#gru, no need to write loop, use gru in tf.common
def gru(inputs, units=128, sequence_length=None, parallel_iterations=64, name='gru', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        outputs, _ = tf.nn.dynamic_rnn(cell=GRUCell(units), inputs=inputs, sequence_length=sequence_length,
                                       dtype=inputs.dtype, parallel_iterations=parallel_iterations)
    return outputs

#cbhg, use cov, rnn, highway, residul layers
def cbhg(inputs, training=True, k=16, bank_filters=128, projection_filters=(128, 128),
         highway_layers=4, highway_units=128, bi_gru_units=128, sequence_length=None,
         name='cbhg', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # for correctness.
        if sequence_length is not None:
            mask = tf.expand_dims(array_ops.sequence_mask(sequence_length, tf.shape(inputs)[1], tf.float32), -1)
            inputs = inputs * mask
        conv_bank_out = conv1d_bank(inputs, training=training, k=k, bank_filters=bank_filters, reuse=reuse)

        # for correctness.
        if sequence_length is not None:
            conv_bank_out = conv_bank_out * mask
        pooled_conv = tf.layers.max_pooling1d(conv_bank_out, pool_size=2, strides=1, padding='same')

        # for correctness.
        if sequence_length is not None:
            pooled_conv = pooled_conv * mask
        conv_proj_out = conv1d_projections(pooled_conv, training=training, projection_filters=projection_filters, reuse=reuse)

        highway_inputs = conv_proj_out + inputs
        if projection_filters[-1] != highway_units:
            # linear transform for highway.
            highway_inputs = tf.layers.dense(highway_inputs, highway_units)
        # for correctness.
        if sequence_length is not None:
            highway_inputs = highway_inputs * mask
        highway_outputs = highway_net(highway_inputs, layers=highway_layers, reuse=reuse)

        # for correctness.
        if sequence_length is not None:
            highway_outputs = highway_outputs * mask
        bi_gru_out = bi_gru(highway_outputs, units=bi_gru_units, sequence_length=sequence_length, reuse=reuse)
    return bi_gru_out

#residul rnn
def multi_residul_rnn(inputs, units=128, layer_num = 2, sequence_length=None, parallel_iterations=64, name='gru', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        outputs, _ = tf.nn.dynamic_rnn(cell=MultiRNNCell([ResidualWrapper(GRUCell(units)) for _ in range(layer_num)]),
                                       inputs=inputs, sequence_length=sequence_length,
                                       dtype=inputs.dtype, parallel_iterations=parallel_iterations)
    return outputs

#self loop rnn, no mask
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


#EmbeddingLayer
class EmbeddingLayer(object):
    """Embedding Layer
    """

    def __init__(self, classes, size, initializer=None, dtype=tf.float32, reuse=None):
        """
        Args:
            classes[int]: embedding classes.
            size[int]: embedding units(size).
            initializer:
            reuse:
        """
        self.__classes = classes
        self.__size = size
        self.__initializer = initializer
        self.__dtype = dtype
        self.__reuse = reuse

    @property
    def classes(self):
        return self.__classes

    @property
    def dtype(self):
        return self.__dtype

    @property
    def size(self):
        return self.__size

    def __call__(self, input_ts, scope=None):
        print(type(self).__name__)
        with tf.variable_scope(scope or type(self).__name__, reuse=self.__reuse):
            if self.__initializer:
                initializer = self.__initializer
            else:
                # Default initializer for embeddings should have variance=1.
                sqrt3 = math.sqrt(3)    # Uniform(-sqrt(3), sqrt(3)) has variance=1.
                initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            embedding = tf.get_variable(name="embedding", shape=(self.classes, self.size),
                                        initializer=initializer, dtype=self.dtype)
            embedded = tf.nn.embedding_lookup(embedding, input_ts)
        return embedded


class BahdanauAttentionModule(object):
    """Attention Module
    Args:
        attention_units:    The attention module's capacity (should be proportional to query_units)
        memory:             A tensor, whose shape should be (None, Time, Unit)
        time_major:
    """

    def __init__(self, attention_units, memory, sequence_length=None, time_major=True, mode=0):
        self.attention_units = attention_units
        self.enc_units = memory.get_shape()[-1].value

        if time_major:
            memory = tf.transpose(memory, perm=(1, 0, 2))

        self.enc_length = tf.shape(memory)[1]
        self.batch_size = tf.shape(memory)[0]
        self.mode = mode
        self.mask = array_ops.sequence_mask(sequence_length, self.enc_length,
                                            tf.float32) if sequence_length is not None else None

        self.memory = tf.reshape(memory, (tf.shape(memory)[0], self.enc_length, 1, self.enc_units))

        # pre-compute Uahj to minimize the computational cost
        with tf.variable_scope('attention'):
            Ua = tf.get_variable(name='Ua', shape=(1, 1, self.enc_units, self.attention_units))
        self.hidden_feats = tf.nn.conv2d(self.memory, Ua, [1, 1, 1, 1], "SAME")

    def __call__(self, query):

        with tf.variable_scope('attention'):
            # Check if the m emory's batch_size is consistent with query's batch_size

            query_units = query.get_shape()[-1].value

            Wa = tf.get_variable(name='Wa', shape=(query_units, self.attention_units))
            Va = tf.get_variable(name='Va', shape=(self.attention_units,),
                                 initializer=tf.constant_initializer(
                                     0.0) if self.mode == 0 else tf.constant_initializer(1e-2))
            b = tf.get_variable(name='b', shape=(self.attention_units,),
                                initializer=tf.constant_initializer(0.0) if self.mode == 0 else tf.constant_initializer(
                                    0.5))

            # 1st. compute query_feat (query's repsentation in attention module)
            query_feat = tf.reshape(tf.matmul(query, Wa), (-1, 1, 1, self.attention_units))

            # 2nd. compute the energy for all time steps in encoder (element-wise mul then reduce)
            e = tf.reduce_sum(Va * tf.nn.tanh(self.hidden_feats + query_feat + b), axis=(2, 3))

            # 3rd. compute the score
            if self.mask is not None:
                exp_e = tf.exp(e)
                exp_e = exp_e * self.mask
                alpha = tf.divide(exp_e, tf.reduce_sum(exp_e, axis=-1, keep_dims=True))
            else:
                alpha = tf.nn.softmax(e)

            # 4th. get the weighted context from memory (element-wise mul then reduce)
            context = tf.reshape(alpha, (tf.shape(query)[0], self.enc_length, 1, 1)) * self.memory
            context = tf.reduce_sum(context, axis=(1, 2))

            return context, alpha


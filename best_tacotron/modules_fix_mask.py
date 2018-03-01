import tensorflow as tf
from tensorflow.python.ops import array_ops
from TFCommon.RNNCell import GRUCell

###
def conv1d_bank(inputs, training=True, k=16, bank_filters=128, name='conv1d_bank', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        stacked_conv = tf.concat([tf.layers.conv1d(inputs, filters=bank_filters, kernel_size=idk, strides=1,
                                                   padding='same', name='inner_conv_{}'.format(idk))
                                  for idk in range(1, k + 1)], axis=-1)
        normed_conv = tf.nn.relu(tf.layers.batch_normalization(stacked_conv, training=training))
    return normed_conv


def conv1d_projections(inputs, training=True, projection_filters=(128, 128), name='conv1d_projections', reuse=False, mask=None):
    with tf.variable_scope(name, reuse=reuse):
        conv_0 = tf.layers.conv1d(inputs, filters=projection_filters[0], kernel_size=3, strides=1,
                                  padding='same', name='inner_conv_0')
        norm_0 = tf.nn.relu(tf.layers.batch_normalization(conv_0, training=training))
        if mask is not None:
            norm_0 = norm_0 * mask
        conv_1 = tf.layers.conv1d(norm_0, filters=projection_filters[1], kernel_size=3, strides=1,
                                  padding='same', name='inner_conv_1')
        norm_1 = tf.layers.batch_normalization(conv_1, training=training)
        #no need to mask, next has mask, but add mask for good look
        if mask is not None:
            norm_1 = norm_1 * mask
    return norm_1


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


def bi_gru(inputs, units=128, sequence_length=None, parallel_iterations=64, name='bidirectional_gru', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRUCell(units), cell_bw=GRUCell(units),
                                                     inputs=inputs, sequence_length=sequence_length,
                                                     dtype=inputs.dtype, parallel_iterations=parallel_iterations)
        outputs = tf.concat(outputs, axis=-1)
    return outputs


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
        if sequence_length is not None:
            conv_proj_out = conv1d_projections(pooled_conv, training=training, projection_filters=projection_filters, reuse=reuse, mask=mask)
        else:
            conv_proj_out = conv1d_projections(pooled_conv, training=training, projection_filters=projection_filters,
                                               reuse=reuse)


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

def cbhg_fused(inputs, training=True, k=16, bank_filters=128, projection_filters=(128, 128),
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
        with tf.variable_scope("biGRU", reuse=reuse):
            fw_cell = tf.contrib.rnn.FusedRNNCellAdaptor(GRUCell(bi_gru_units))
            bw_cell = tf.contrib.rnn.TimeReversedFusedRNN(fw_cell)
            fw_out, _ = fw_cell(highway_outputs, scope="forward")
            bw_out, _ = bw_cell(highway_outputs, scope="backward")
            bi_gru_out = tf.concat([fw_out, bw_out], axis=-1)
    return bi_gru_out
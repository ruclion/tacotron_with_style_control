import tensorflow as tf
from Speech.tensorflow.TFCommon.Model import Model
from Speech.tensorflow.TFCommon.RNNCell import GRUCell
from Speech.tensorflow.TFCommon.Layers import EmbeddingLayer
from Speech.tensorflow.TFCommon.Attention import BahdanauAttentionModule as AttentionModule
from Speech.tensorflow.models.Tacotron import modules
from Speech.tensorflow.models.Tacotron.hyperparameter import HyperParams
from Speech.tensorflow.models.Tacotron.rnn_cells import ZoneoutWrapper, GRUCell as sGRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops


class Tacotron(Model):
    def __init__(self, inp, inp_mask, decode_time_steps, hyper_params=None, name='Tacotron'):
        """
        Build the computational graph.
        :param inp:
        :param inp_mask:
        :param decode_time_steps:
        :param hyper_params:
        :param name:
        """
        super(Tacotron, self).__init__(name)
        self.hyper_params = HyperParams() if hyper_params is None else hyper_params

        with tf.variable_scope(name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            batch_size = tf.shape(inp)[0]
            input_time_steps = tf.shape(inp)[1]
            reduc = self.hyper_params.reduction_rate
            output_time_steps = decode_time_steps * reduc

            ### Encoder [begin]
            with tf.variable_scope('character_embedding'):
                embed_inp = EmbeddingLayer(self.hyper_params.embed_class, self.hyper_params.embed_dim)(inp)
            with tf.variable_scope('encoder_pre_net'):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=False)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=False)
            encoder_output = modules.cbhg(pre_ed_inp, training=False, k=16, bank_filters=128,
                                          projection_filters=(128, 128), highway_layers=4, highway_units=128,
                                          bi_gru_units=128, sequence_length=inp_mask,
                                          name='encoder_cbhg', reuse=False)
            ### Encoder [end]

            ### Attention Module
            with tf.variable_scope('attention'):
                att_module = AttentionModule(256, encoder_output, sequence_length=inp_mask, time_major=False)

            ### Decoder [begin]
            att_cell = ZoneoutWrapper(sGRUCell(256), 0.1, False)
            dec_cell = MultiRNNCell([ResidualWrapper(GRUCell(256)) for _ in range(2)])
            # prepare output alpha TensorArray
            with tf.variable_scope('prepare_decode'):
                # prepare output alpha TensorArray
                reduced_time_steps = tf.div(output_time_steps, reduc)
                init_att_cell_state = att_cell.zero_state(batch_size, tf.float32)
                init_dec_cell_state = dec_cell.zero_state(batch_size, tf.float32)
                init_state_tup = tuple([init_att_cell_state, init_dec_cell_state])
                init_output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                init_alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                go_array = tf.zeros([batch_size, self.hyper_params.seq2seq_dim], dtype=tf.float32)
                init_context = tf.zeros([batch_size, 256], dtype=tf.float32)
                init_time = tf.constant(0, dtype=tf.int32)
            cond = lambda x, *_: tf.less(x, reduced_time_steps)
            def body(this_time, old_output_ta, old_alpha_ta, old_state_tup, last_context, last_output):
                with tf.variable_scope('decoder_pre_net'):
                    dec_pre_ed_inp = last_output
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 256, tf.nn.relu), training=True)
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 128, tf.nn.relu), training=True)
                with tf.variable_scope('attention_rnn'):
                    att_cell_inp = tf.concat([last_context, dec_pre_ed_inp], axis=-1)
                    att_cell_out, att_cell_state = att_cell(att_cell_inp, old_state_tup[0])
                with tf.variable_scope('attention'):
                    query = att_cell_state
                    context, alpha = att_module(query)
                    new_alpha_ta = old_alpha_ta.write(this_time, alpha)
                with tf.variable_scope('decoder_rnn'):
                    dec_input = tf.layers.dense(tf.concat([att_cell_out, context], axis=-1), 256)
                    dec_cell_out, dec_cell_state = dec_cell(dec_input, old_state_tup[1])
                    dense_out = tf.layers.dense(dec_cell_out, self.hyper_params.seq2seq_dim * reduc)
                    new_output_ta = old_output_ta.write(this_time, dense_out)
                    new_output = dense_out[:, -self.hyper_params.seq2seq_dim:]
                new_state_tup = tuple([att_cell_state, dec_cell_state])
                return tf.add(this_time, 1), new_output_ta, new_alpha_ta, new_state_tup, context, new_output

            # run loop
            _, seq2seq_output_ta, alpha_ta, *_ = tf.while_loop(cond, body, [init_time, init_output_ta,
                                                                            init_alpha_ta, init_state_tup,
                                                                            init_context, go_array])
            with tf.variable_scope('reshape_decode'):
                seq2seq_output = tf.reshape(seq2seq_output_ta.stack(),
                                            shape=(reduced_time_steps, batch_size, self.hyper_params.seq2seq_dim * reduc))
                seq2seq_output = tf.reshape(tf.transpose(seq2seq_output, perm=(1, 0, 2)),
                                            shape=(batch_size, output_time_steps, self.hyper_params.seq2seq_dim))
                self.seq2seq_output = seq2seq_output

                alpha_output = tf.reshape(alpha_ta.stack(),
                                          shape=(reduced_time_steps, batch_size, input_time_steps))
                alpha_output = tf.expand_dims(tf.transpose(alpha_output, perm=(1, 0, 2)), -1)
                self.alpha_output = alpha_output
            ### Decoder [end]

            ### PostNet [begin]
            post_output = modules.cbhg(seq2seq_output, training=False, k=8, bank_filters=128,
                                       projection_filters=(256, self.hyper_params.seq2seq_dim),
                                       highway_layers=4, highway_units=128,
                                       bi_gru_units=128, sequence_length=None,
                                       name='decoder_cbhg', reuse=False)
            post_output = tf.layers.dense(post_output, self.hyper_params.post_dim, name='post_linear_transform')
            self.post_output = post_output
            ### PostNet [end]

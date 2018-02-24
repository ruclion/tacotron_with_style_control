import tensorflow as tf
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from TFCommon.Layers import EmbeddingLayer
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
import best_tacotron.modules as modules
from best_tacotron.hyperparameter_style import HyperParams
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops

import os


def l1_loss(gtruth, predict):
    return tf.reduce_mean(tf.abs(gtruth - predict))


class Tacotron(Model):
    def __init__(self, inp, inp_mask, seq2seq_gtruth, post_gtruth, hyper_params=None, training=True, name='Tacotron', reuse=False):
        """
        Build the computational graph.
        :param inp:
        :param inp_mask:
        :param seq2seq_gtruth:
        :param post_gtruth:
        :param hyper_params:
        :param training:
        :param name:
        """
        super(Tacotron, self).__init__(name)
        self.hyper_params = HyperParams() if hyper_params is None else hyper_params
        with tf.variable_scope(name, reuse=reuse):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.Variable(self.hyper_params.learning_rate[0],
                                             name='learning_rate', trainable=False, dtype=tf.float32)

            batch_size = tf.shape(inp)[0]
            input_time_steps = tf.shape(inp)[1]
            output_time_steps = tf.shape(seq2seq_gtruth)[1]

            ### Encoder [begin]
            with tf.variable_scope('character_embedding'):
                embed_inp = EmbeddingLayer(self.hyper_params.embed_class, self.hyper_params.embed_dim)(inp)
            with tf.variable_scope("changeToVarible"):
                self.single_style_token = tf.get_variable('style_token', (1, self.hyper_params.styles_kind, self.hyper_params.style_dim), dtype=tf.float32)
                self.style_token = tf.tile(self.single_style_token, (batch_size, 1, 1))
            with tf.variable_scope('encoder_pre_net'):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=training)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=training)
            encoder_output = modules.cbhg(pre_ed_inp, training=training, k=16, bank_filters=128,
                                          projection_filters=(128, 128), highway_layers=4, highway_units=128,
                                          bi_gru_units=128, sequence_length=inp_mask,
                                          name='encoder_cbhg', reuse=False)
            ### Encoder [end]

            ### Attention Module
            with tf.variable_scope('attention'):
                att_module = AttentionModule(256, encoder_output, sequence_length=inp_mask, time_major=False)
            with tf.variable_scope("attention_style"):
                att_module_style = AttentionModule(256, self.style_token, time_major=False)

            ### Decoder [begin]
            att_cell = GRUCell(256)
            att_cell_style = GRUCell(256)
            dec_cell = MultiRNNCell([ResidualWrapper(GRUCell(256)) for _ in range(2)])
            # prepare output alpha TensorArray
            with tf.variable_scope('prepare_decode'):
                reduc = self.hyper_params.reduction_rate
                reduced_time_steps = tf.div(output_time_steps, reduc)
                init_att_cell_state = att_cell.zero_state(batch_size, tf.float32)
                init_att_cell_state_style = att_cell_style.zero_state(batch_size, tf.float32)
                init_dec_cell_state = dec_cell.zero_state(batch_size, tf.float32)
                init_state_tup = tuple([init_att_cell_state, init_dec_cell_state, init_att_cell_state_style])
                init_output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                init_alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                init_weight_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                init_weight_per_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                init_alpha_style_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                time_major_seq2seq_gtruth = tf.transpose(seq2seq_gtruth, perm=(1, 0, 2))
                indic_array = tf.concat([tf.zeros([reduc, batch_size, self.hyper_params.seq2seq_dim]),
                                         time_major_seq2seq_gtruth], axis=0)
                init_context = tf.zeros([batch_size, 256], dtype=tf.float32)
                init_context_style = tf.zeros([batch_size, 256], dtype=tf.float32)
                init_time = tf.constant(0, dtype=tf.int32)
            cond = lambda x, *_: tf.less(x, reduced_time_steps)
            def body(this_time, old_context, old_context_style, old_output_ta, old_alpha_ta, old_alpha_style_ta, old_weight_ta, old_weight_per_ta, old_state_tup):
                with tf.variable_scope('decoder_pre_net'):
                    dec_pre_ed_inp = indic_array[reduc*this_time + reduc - 1]
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 256, tf.nn.relu), training=training)
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 128, tf.nn.relu), training=training)
                with tf.variable_scope('attention_rnn'):
                    att_cell_inp = tf.concat([old_context, dec_pre_ed_inp], axis=-1)
                    att_cell_out, att_cell_state = att_cell(att_cell_inp, old_state_tup[0])
                with tf.variable_scope('attention_rnn_style'):
                    att_cell_inp_style = tf.concat([old_context_style, dec_pre_ed_inp], axis=-1)
                    att_cell_out_style, att_cell_state_style = att_cell_style(att_cell_inp_style, old_state_tup[2])
                with tf.variable_scope('attention'):
                    query = att_cell_state[0]
                    context, alpha = att_module(query)
                    new_alpha_ta = old_alpha_ta.write(this_time, alpha)
                with tf.variable_scope("attention_style"):
                    query_style = att_cell_state_style[0]
                    context_style, alpha_style = att_module_style(query_style)
                    new_alpha_style_ta = old_alpha_style_ta.write(this_time, alpha_style)
                with tf.variable_scope("weighting"):
                    weighting = tf.layers.dense(dec_pre_ed_inp, 1, tf.nn.sigmoid)
                    # weighting = tf.nn.softmax(weighting)
                    new_weight_ta = old_weight_ta.write(this_time, weighting)
                with tf.variable_scope('decoder_rnn'):
                    weighting_context = weighting * context + (1 - weighting) * context_style
                    weight_per = tf.reduce_mean(tf.abs((1 - weighting) * context_style) / (tf.abs(weighting * context) + tf.abs((1 - weighting) * context_style)))
                    new_weight_per_ta = old_weight_per_ta.write(this_time, weight_per)
                    dec_input = tf.layers.dense(tf.concat([att_cell_out, weighting_context], axis=-1), 256)
                    dec_cell_out, dec_cell_state = dec_cell(dec_input, old_state_tup[1])
                    dense_out = tf.layers.dense(dec_cell_out, self.hyper_params.seq2seq_dim * reduc)
                    new_output_ta = old_output_ta.write(this_time, dense_out)
                new_state_tup = tuple([att_cell_state, dec_cell_state, att_cell_state_style])
                return tf.add(this_time, 1), context, context_style, new_output_ta, new_alpha_ta, new_alpha_style_ta, new_weight_ta, new_weight_per_ta, new_state_tup

            # run loop
            _, _, _, seq2seq_output_ta, alpha_ta, alpha_style_ta, weight_ta, weight_per_ta, *_ = tf.while_loop(cond, body, [init_time, init_context, init_context_style, init_output_ta,
                                                                               init_alpha_ta, init_alpha_style_ta, init_weight_ta, init_weight_per_ta, init_state_tup],
                                                                  parallel_iterations=32)

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

                alpha_output_style = tf.reshape(alpha_style_ta.stack(),
                                               shape=(reduced_time_steps, batch_size, self.hyper_params.styles_kind))
                alpha_output_style = tf.expand_dims(tf.transpose(alpha_output_style, perm=(1, 0, 2)), -1)  # batch major
                self.alpha_output_style = alpha_output_style

                weight_ta = tf.reshape(weight_ta.stack(), shape=(reduced_time_steps, batch_size, 1))
                weight_ta = tf.transpose(weight_ta, perm=(1, 0, 2))
                self.weight_ta = weight_ta

                weight_per_ta = tf.reshape(weight_per_ta.stack(), shape=(reduced_time_steps, 1))
                self.weight_per_ta = weight_per_ta
            ### Decoder [end]

            ### PostNet [begin]
            post_output = modules.cbhg(seq2seq_output, training=training, k=8, bank_filters=128,
                                       projection_filters=(256, self.hyper_params.seq2seq_dim),
                                       highway_layers=4, highway_units=128,
                                       bi_gru_units=128, sequence_length=None,
                                       name='decoder_cbhg', reuse=False)
            post_output = tf.layers.dense(post_output, self.hyper_params.post_dim, name='post_linear_transform')
            self.post_output = post_output
            ### PostNet [end]

        ### Loss
        with tf.variable_scope('loss'):
            self.seq2seq_loss = l1_loss(seq2seq_gtruth, seq2seq_output)
            self.post_loss = l1_loss(post_gtruth, post_output)
            self.loss = self.seq2seq_loss + self.post_loss

    def get_scalar_summary(self, suffix):
        with tf.variable_scope('summary'):
            sums = [tf.summary.scalar('{}/loss'.format(suffix), self.loss),
                    tf.summary.scalar('{}/seq2seq_loss'.format(suffix), self.seq2seq_loss),
                    tf.summary.scalar('{}/post_loss'.format(suffix), self.post_loss),
                    tf.summary.histogram('{}/weight_0'.format(suffix), self.weight_ta[0]),
                    tf.summary.histogram('{}/weight_1'.format(suffix), self.weight_ta[1]),
                    tf.summary.histogram('{}/weight_per'.format(suffix), self.weight_per_ta),
                    tf.summary.scalar("train/style_0_0", self.single_style_token[0][0][0]),
                    tf.summary.scalar("train/style_0_100", self.single_style_token[0][0][100]),
                    tf.summary.scalar("train/style_5_100", self.single_style_token[0][5][100])]
            return tf.summary.merge(sums)

    def get_alpha_summary(self, suffix, num_img=2):
        with tf.variable_scope('summary'):
            sums = [tf.summary.image('{}/alpha'.format(suffix), self.alpha_output[:num_img]),
                    tf.summary.image('{}/alpha_style'.format(suffix), self.alpha_output_style[:num_img])]
            return tf.summary.merge(sums)

    def get_learning_rate_upd_op(self, global_step_eval):
        for lr, lr_decay_step in zip(self.hyper_params.learning_rate[1:], self.hyper_params.learning_rate_decay_step):
            if global_step_eval == lr_decay_step:
                return self.learning_rate.assign(lr)
        return None

    def exp_learning_rate_decay(self, init_lr=0.001, stair=8000):
        return tf.train.exponential_decay(init_lr, self.global_step, stair, 0.96, staircase=False)

import tensorflow as tf
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from TFCommon.Layers import EmbeddingLayer
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
import best_tacotron.modules as modules
from best_tacotron.hyperparameter_style import HyperParams
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops

unkonwn_parallel_iterations = 128
#
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
            with tf.variable_scope("changeToVarible"):
                self.single_style_token = tf.get_variable('style_token', (1, self.hyper_params.styles_kind, self.hyper_params.style_dim), dtype=tf.float32)
                self.style_token = tf.tile(self.single_style_token, (batch_size, 1, 1))
            with tf.variable_scope("inp_att_prepare"):
                inp_att = tf.get_variable('inp_att', (32, 10), tf.float32)
                self.inp_att = inp_att
            with tf.variable_scope('encoder_pre_net'):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=False)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=False)
            encoder_output = modules.cbhg(pre_ed_inp, training=False, k=16, bank_filters=128,
                                          projection_filters=(128, 128), highway_layers=4, highway_units=128,
                                          bi_gru_units=128, sequence_length=inp_mask,
                                          name='encoder_cbhg', reuse=False)
            inp_att = tf.Print(inp_att, [inp_att], message='inp_att', summarize=10)
            sentence_style = tf.reduce_sum(tf.expand_dims(inp_att, axis=-1) * self.style_token, axis=1)
            sentence_style = tf.Print(sentence_style, [sentence_style], message='style', summarize=10)

            # with tf.variable_scope('post_text'):
            #     all_outputs, _ = tf.nn.dynamic_rnn(cell=GRUCell(256), inputs=encoder_output, sequence_length=inp_mask,
            #                                    dtype=encoder_output.dtype, parallel_iterations=unkonwn_parallel_iterations)
            #     all_outputs = tf.transpose(all_outputs, [1, 0, 2])
            #     static_encoder_output = all_outputs[-1]
            # ### Encoder [end]
            #
            # sentence_style_att = tf.layers.dense(static_encoder_output, 256, tf.nn.relu)
            # sentence_style_att = tf.layers.dense(sentence_style_att, 64, tf.nn.relu)
            # sentence_style = tf.layers.dense(sentence_style_att, 10, tf.nn.softmax)
            #
            # sentence_style = tf.cond(tf.equal(ctr_flag, 1), lambda: ctr_attention, lambda: sentence_style)
            # sentence_style = tf.Print(sentence_style, [sentence_style], message='att', summarize=10)
            # sentence_style = tf.reduce_sum(tf.expand_dims(sentence_style, axis=-1) * self.style_token, axis=1)
            # sentence_style = tf.Print(sentence_style, [sentence_style], message='style', summarize=10)
            # sentence_style = tf.cond(tf.equal(ctr_flag, 1),
            #                         lambda: tf.reduce_sum(tf.expand_dims(sentence_style, axis=-1) * self.style_token,
            #                                               axis=1),
            #                         lambda: sentence_style)


            ### Attention Module
            with tf.variable_scope('attention'):
                att_module = AttentionModule(256, encoder_output, sequence_length=inp_mask, time_major=False)


            ### Decoder [begin]
            att_cell = GRUCell(256)
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
                init_weight_per_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                go_array = tf.zeros([batch_size, self.hyper_params.seq2seq_dim], dtype=tf.float32)
                init_context = tf.zeros([batch_size, 256], dtype=tf.float32)
                init_time = tf.constant(0, dtype=tf.int32)
            cond = lambda x, *_: tf.less(x, reduced_time_steps)
            def body(this_time, old_output_ta, old_alpha_ta, old_weight_per_ta,
                     old_state_tup, last_context, last_output):
                with tf.variable_scope('decoder_pre_net'):
                    dec_pre_ed_inp = last_output
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 256, tf.nn.relu), training=False)
                    dec_pre_ed_inp = tf.layers.dropout(tf.layers.dense(dec_pre_ed_inp, 128, tf.nn.relu), training=False)
                with tf.variable_scope('attention_rnn'):
                    # dec_pre_ed_inp = tf.Print(dec_pre_ed_inp, [dec_pre_ed_inp[0]], message='dec', summarize=10)
                    att_cell_inp = tf.concat([last_context, dec_pre_ed_inp], axis=-1)
                    att_cell_out, att_cell_state = att_cell(att_cell_inp, old_state_tup[0])
                with tf.variable_scope('attention'):
                    query = att_cell_state[0]
                    context, alpha = att_module(query)
                    new_alpha_ta = old_alpha_ta.write(this_time, alpha)
                with tf.variable_scope('decoder_rnn'):
                    weighting_context = context + sentence_style
                    weight_per = tf.reduce_mean(tf.abs(sentence_style) / (tf.abs(context) + tf.abs(sentence_style)))
                    new_weight_per_ta = old_weight_per_ta.write(this_time, weight_per)
                    dec_input = tf.layers.dense(tf.concat([att_cell_out, weighting_context], axis=-1), 256)
                    # dec_input = tf.layers.dense(tf.concat([att_cell_out, context], axis=-1), 256)
                    dec_cell_out, dec_cell_state = dec_cell(dec_input, old_state_tup[1])
                    dense_out = tf.layers.dense(dec_cell_out, self.hyper_params.seq2seq_dim * reduc)
                    new_output_ta = old_output_ta.write(this_time, dense_out)
                    new_output = dense_out[:, -self.hyper_params.seq2seq_dim:]
                new_state_tup = tuple([att_cell_state, dec_cell_state])
                return tf.add(this_time, 1), new_output_ta, new_alpha_ta, \
                       new_weight_per_ta, new_state_tup, context, new_output


            # run loop
            _, seq2seq_output_ta, alpha_ta, weight_per_ta, *_ = tf.while_loop(cond, body, [init_time,
                                                                                                                      init_output_ta,
                                                                                                                      init_alpha_ta,
                                                                                                                      init_weight_per_ta,
                                                                                                                      init_state_tup,
                                                                                                                      init_context,
                                                                                                                      go_array
                                                                                                                      ])
            with tf.variable_scope('reshape_decode'):
                seq2seq_output = tf.reshape(seq2seq_output_ta.stack(),
                                            shape=(reduced_time_steps, batch_size, self.hyper_params.seq2seq_dim * reduc))
                seq2seq_output = tf.reshape(tf.transpose(seq2seq_output, perm=(1, 0, 2)),
                                            shape=(batch_size, output_time_steps, self.hyper_params.seq2seq_dim))
                self.seq2seq_output = seq2seq_output

                # alpha_output = tf.reshape(alpha_ta.stack(),
                #                           shape=(reduced_time_steps, batch_size, input_time_steps))
                # alpha_output = tf.expand_dims(tf.transpose(alpha_output, perm=(1, 0, 2)), -1)
                # self.alpha_output = alpha_output
                #
                # alpha_output_style = tf.reshape(alpha_style_ta.stack(),
                #                                 shape=(reduced_time_steps, batch_size, self.hyper_params.styles_kind))
                # alpha_output_style = tf.expand_dims(tf.transpose(alpha_output_style, perm=(1, 0, 2)), -1)  # batch major
                # self.alpha_output_style = alpha_output_style
                #
                # weight_ta = tf.reshape(weight_ta.stack(), shape=(reduced_time_steps, batch_size, 1))
                # weight_ta = tf.transpose(weight_ta, perm=(1, 0, 2))
                # self.weight_ta = weight_ta
                #
                # weight_per_ta = tf.reshape(weight_per_ta.stack(), shape=(reduced_time_steps, 1))
                # self.weight_per_ta = weight_per_ta
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

    # def get_scalar_summary(self, suffix):
    #     with tf.variable_scope('summary'):
    #         sums = [tf.summary.histogram('{}/weight_0'.format(suffix), self.weight_ta[0]),
    #                 tf.summary.histogram('{}/weight_1'.format(suffix), self.weight_ta[1]),
    #                 tf.summary.histogram('{}/weight_per'.format(suffix), self.weight_per_ta),
    #                 tf.summary.scalar("train/style_0_0", self.single_style_token[0][0][0]),
    #                 tf.summary.scalar("train/style_0_100", self.single_style_token[0][0][100]),
    #                 tf.summary.scalar("train/style_5_100", self.single_style_token[0][5][100])]
    #         return tf.summary.merge(sums)
    #
    # def get_alpha_summary(self, suffix, num_img=2):
    #     with tf.variable_scope('summary'):
    #         sums = [tf.summary.image('{}/alpha'.format(suffix), self.alpha_output[:num_img]),
    #                 tf.summary.image('{}/alpha_style'.format(suffix), self.alpha_output_style[:num_img])]
    #         return tf.summary.merge(sums)
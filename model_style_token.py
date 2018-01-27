import tensorflow as tf, pickle as pkl, os
from Tacotron.Modules.CBHG import CBHG
from TFCommon.Model import Model
from TFCommon.RNNCell import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
from tensorflow.python.ops import array_ops
from TFCommon.Attention import BahdanauAttentionModule as AttentionModule
from TFCommon.Layers import EmbeddingLayer
import numpy as np
import random
import sys
import gc
import time
import audio

bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sr = 16000

EPOCHS = 100000	# 7142 -> 2M
EMBED_CLASS = 100
EMBED_DIM = 256
SPC_EMBED_CLASS = 5
SPC_EMBED_DIM = 32
ATT_RNN_SIZE = 256
STYLE_ATT_RNN_SIZE = 256
DEC_RNN_SIZE = 256
OUTPUT_MEL_DIM = 80	# 80
OUTPUT_SPEC_DIM = 513 # 513
LR_RATE = 0.001
styles_kind = 10
style_dim = 256
train_r = 5



save_path = os.path.join('model', 'weighting_style_div_200')
model_name = "TTS"
tfrecord_path = './data/th-coss_female_wav_mel_stftm.tfrecords'
pkl_path = './data/th-coss_female_stats.pkl'


slice_data_size = 256
BATCH_SIZE = 32
slice_num = 23
batch_id = 0
batch_no = 0
batch_idx = None
global_data = None



def add_layer(inputs, in_size, out_size, w_name, b_name, activation_function=None):
    Weights = tf.get_variable(w_name, [in_size, out_size],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))

    biases = tf.get_variable(b_name, out_size,
                             initializer=tf.constant_initializer(0))

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


class TTS(Model):

    def __init__(self, r=5, is_training=True, name="TTS"):
        super(TTS, self).__init__(name)
        self.__r = r
        self.training = is_training

    @property
    def r(self):
        return self.__r

    def build(self, inp, inp_mask, mel_gtruth, spec_gtruth):
        batch_size = tf.shape(inp)[0]
        input_time_steps = tf.shape(inp)[1]
        output_time_steps = tf.shape(mel_gtruth)[1]

        ### Encoder [ begin
        with tf.variable_scope("encoder"):

            with tf.variable_scope("embedding"):
                embed_inp = EmbeddingLayer(EMBED_CLASS, EMBED_DIM)(inp)

            with tf.variable_scope("changeToVarible"):

                self.single_style_token = tf.get_variable('style_token', (1, styles_kind, style_dim), dtype=tf.float32)
                self.style_token = tf.tile(self.single_style_token, (batch_size, 1, 1))



            with tf.variable_scope("pre-net"):
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(embed_inp, 256, tf.nn.relu), training=self.training)
                pre_ed_inp = tf.layers.dropout(tf.layers.dense(pre_ed_inp, 128, tf.nn.relu), training=self.training)




            with tf.variable_scope("CBHG"):
                # batch major
                encoder_output = CBHG(16, (128, 128))(pre_ed_inp, sequence_length=inp_mask, is_training=self.training, time_major=False)

        with tf.variable_scope("attention"):
            att_module = AttentionModule(ATT_RNN_SIZE, encoder_output, sequence_length=inp_mask, time_major=False)
        with tf.variable_scope("attention_style"):
            att_module_style = AttentionModule(STYLE_ATT_RNN_SIZE, self.style_token, time_major=False)

        with tf.variable_scope("decoder"):
            with tf.variable_scope("attentionRnn"):
                att_cell = GRUCell(ATT_RNN_SIZE)
            with tf.variable_scope("acoustic_module"):
                aco_cell = MultiRNNCell([ResidualWrapper(GRUCell(DEC_RNN_SIZE)) for _ in range(2)])

            ### prepare output alpha TensorArray
            reduced_time_steps = tf.div(output_time_steps, self.r)
            att_cell_state = att_cell.init_state(batch_size, tf.float32)
            aco_cell_state = aco_cell.zero_state(batch_size, tf.float32)
            state_tup = tuple([att_cell_state, aco_cell_state])
            output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            weight_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            alpha_style_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
            indic_ta = tf.TensorArray(size=self.r + output_time_steps, dtype=tf.float32)
            time_major_mel_gtruth = tf.transpose(mel_gtruth, perm=(1, 0, 2))
            indic_array = tf.concat([tf.zeros([self.r, batch_size, OUTPUT_MEL_DIM]), time_major_mel_gtruth], axis=0)
            indic_ta = indic_ta.unstack(indic_array)
            #init_context = tf.zeros((batch_size, 256))

            time = tf.constant(0, dtype=tf.int32)
            cond = lambda time, *_: tf.less(time, reduced_time_steps)
            def body(time, output_ta, alpha_ta, alpha_style_ta, weight_ta, state_tup):
                with tf.variable_scope("att-rnn"):
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(indic_ta.read(self.r*time + self.r - 1), 256, tf.nn.relu), training=self.training)
                    pre_ed_indic = tf.layers.dropout(tf.layers.dense(pre_ed_indic, 128, tf.nn.relu), training=self.training)
                    att_cell_out, att_cell_state = att_cell(tf.concat([pre_ed_indic], axis=-1), state_tup[0])
                with tf.variable_scope("attention"):
                    query = att_cell_state[0]    # att_cell_out
                    context, alpha = att_module(query)
                    alpha_ta = alpha_ta.write(time, alpha)
                with tf.variable_scope("attention_style"):
                    context_style, alpha_style = att_module_style(query)
                    alpha_style_ta = alpha_style_ta.write(time, alpha_style)
                with tf.variable_scope("weighting"):
                    print(query)
                    weighting = add_layer(query, query.shape[-1], 1, 'weighting_w', 'weighting_b', activation_function=tf.nn.sigmoid)
                    # weighting = tf.nn.softmax(weighting)
                    weight_ta = weight_ta.write(time, weighting)

                with tf.variable_scope("acoustic_module"):
                    # weighting0 = tf.reshape(weighting[:, 0], (BATCH_SIZE, 1))
                    # weighting1 = tf.reshape(weighting[:, 1], (BATCH_SIZE, 1))
                    weighting_context = context + weighting * tf.nn.tanh(context_style)
                    # print(weighting_context)
                    aco_input = tf.layers.dense(tf.concat([att_cell_out, weighting_context], axis=-1), DEC_RNN_SIZE)
                    aco_cell_out, aco_cell_state = aco_cell(aco_input, state_tup[1])
                    dense_out = tf.layers.dense(aco_cell_out, OUTPUT_MEL_DIM * self.r)
                    output_ta = output_ta.write(time, dense_out)
                state_tup = tuple([att_cell_state, aco_cell_state])

                return tf.add(time, 1), output_ta, alpha_ta, alpha_style_ta, weight_ta, state_tup

            ### run loop
            _, output_mel_ta, final_alpha_ta, final_alpha_style_ta, final_weight_ta, *_ = tf.while_loop(cond, body, [time, output_ta, alpha_ta, alpha_style_ta, weight_ta, state_tup])
        # print('hjhhhh', reduced_time_steps, batch_size, OUTPUT_MEL_DIM * self.r, batch_size, output_time_steps,
        #       OUTPUT_MEL_DIM)
        # sys.stdout.flush()
        ### time major
        with tf.variable_scope("output"):
            # print('hjhhhh', reduced_time_steps, batch_size, OUTPUT_MEL_DIM * self.r, batch_size, output_time_steps, OUTPUT_MEL_DIM)
            # sys.stdout.flush()
            output_mel = tf.reshape(output_mel_ta.stack(), shape=(reduced_time_steps, batch_size, OUTPUT_MEL_DIM * self.r))
            output_mel = tf.reshape(tf.transpose(output_mel, perm=(1, 0, 2)), shape=(batch_size, output_time_steps, OUTPUT_MEL_DIM))
            self.out_mel = output_mel

            with tf.variable_scope("post-net"):
                output_post = CBHG(8, (256, OUTPUT_MEL_DIM))(output_mel, sequence_length=None, is_training=self.training, time_major=False)
                output_spec = tf.layers.dense(output_post, OUTPUT_SPEC_DIM)
                self.out_stftm = output_spec

            final_alpha = tf.reshape(final_alpha_ta.stack(), shape=(reduced_time_steps, batch_size, input_time_steps))
            final_alpha = tf.transpose(final_alpha, perm=(1,0,2))    # batch major

            final_alpha_style = tf.reshape(final_alpha_style_ta.stack(), shape=(reduced_time_steps, batch_size, styles_kind))
            final_alpha_style = tf.transpose(final_alpha_style, perm=(1, 0, 2))  # batch major

            final_weight_ta = tf.reshape(final_weight_ta.stack(), shape=(reduced_time_steps, batch_size, 1))
            final_weight_ta = tf.transpose(final_weight_ta, perm=(1, 0, 2))  # batch major
            self.weighting = final_weight_ta

            # self.alpha_style_hjk_img = tf.reshape(final_alpha_style, shape=(batch_size, reduced_time_steps, styles_kind))

        with tf.variable_scope("loss_and_metric"):
            self.loss_mel = tf.reduce_mean(tf.abs(mel_gtruth - output_mel))
            self.loss_spec = tf.reduce_mean(tf.abs(spec_gtruth - output_spec))
            self.loss = self.loss_mel + self.loss_spec
            self.alpha_img = tf.expand_dims(final_alpha, -1)
            self.alpha_style_img = tf.expand_dims(final_alpha_style, -1)
            self.weight_img = tf.expand_dims(final_weight_ta, -1)

            self.sums = []
            self.sums.append(tf.summary.image("train/alpha", self.alpha_img[:2]))
            self.sums.append(tf.summary.image("train/alpha_style", self.alpha_style_img[:2]))
            self.sums.append(tf.summary.image("train/weight", self.weight_img[:2]))
            self.sums.append(tf.summary.scalar("train/loss", self.loss))
            self.sums.append(tf.summary.scalar("train/style_0_0", self.single_style_token[0][0][0]))
            self.sums.append(tf.summary.scalar("train/style_0_100", self.single_style_token[0][0][100]))
            self.sums.append(tf.summary.scalar("train/style_5_100", self.single_style_token[0][5][100]))
            self.sums.append(tf.summary.histogram("train/style_vec", self.single_style_token))

            self.pred_audio_holder = tf.placeholder(shape=(None, None), dtype=tf.float32, name='pred_audio')
            self.pred_audio_summary = tf.summary.audio('pred_audio_summary', self.pred_audio_holder,
                                                  sample_rate=sr, max_outputs=12)


    def summary(self, suffix, num_img=2):
        sums = []
        sums.append(tf.summary.scalar("%s/loss" % suffix, self.loss))
        sums.append(tf.summary.scalar("%s/loss_mel" % suffix, self.loss_mel))
        sums.append(tf.summary.scalar("%s/loss_spec" % suffix, self.loss_spec))
        sums.append(tf.summary.image("%s/alpha" % suffix, self.alpha_img[:num_img]))
        sums.append(tf.summary.image("%s/alpha" % suffix, self.alpha_style_img[:num_img]))

        return tf.summary.merge(sums)



with tf.variable_scope("data"):
    inp = tf.placeholder(name="input", shape=(None, None), dtype=tf.int32)
    inp_mask = tf.placeholder(name="inp_mask", shape=(None,), dtype=tf.int32)
    mel_gtruth = tf.placeholder(name="output_mel", shape=(None, None, OUTPUT_MEL_DIM), dtype=tf.float32)
    spec_gtruth = tf.placeholder(name="output_spec", shape=(None, None, OUTPUT_SPEC_DIM), dtype=tf.float32)
    # single_style_token = tf.placeholder(name="single_style_token", shape=(1, styles_kind, style_dim), dtype=tf.float32)


with tf.variable_scope("model"):
    train_model = TTS(r=train_r)
    train_model.build(inp, inp_mask, mel_gtruth, spec_gtruth)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_var = tf.trainable_variables()
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(LR_RATE)
        grads_and_vars = opt.compute_gradients(train_model.loss)
        for i, (grad, var) in enumerate(grads_and_vars):
            # print(var.name)
            if var.name.find('style_token:0') != -1:
                grads_and_vars[i] = (grad / 200.0, var)
                print(var.name)
                print('hhhh time')
                break
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(grads_and_vars, global_step=global_step)
        train_model.saver = tf.train.Saver(max_to_keep=10)



def parse_single_example(example_proto):
    features = {
                "frames": tf.FixedLenFeature([], tf.int64),

                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "norm_mel_raw": tf.FixedLenFeature([], tf.string),
                "norm_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)

    frames = tf.cast(parsed["frames"], tf.int32)

    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    norm_mel = tf.reshape(tf.decode_raw(parsed["norm_mel_raw"], tf.float32), (frames, OUTPUT_MEL_DIM))
    norm_stftm = tf.reshape(tf.decode_raw(parsed["norm_stftm_raw"], tf.float32), (frames, OUTPUT_SPEC_DIM))
    return {"frames": frames, "txt":txt, "txt_len":txt_len, "norm_mel": norm_mel, "norm_stftm": norm_stftm}


def get_dataset(tfrecord_path):
    with open(pkl_path, "rb") as f:
        stats = pkl.load(f)
    pad_mel = np.ones((1,80), dtype=np.float32) * -5
    norm_pad_mel = (pad_mel - stats["log_mel_mean"]) / stats["log_mel_std"]
    norm_pad_mel_min = np.asscalar(np.float32(np.min(norm_pad_mel)))
    print(norm_pad_mel_min)

    pad_stftm = np.ones((1, OUTPUT_SPEC_DIM), dtype=np.float32) * -5
    norm_pad_stftm = (pad_stftm - stats["log_stftm_mean"]) / stats["log_stftm_std"]
    norm_pad_stftm_min = np.asscalar(np.float32(np.min(norm_pad_stftm)))
    print(norm_pad_stftm_min)

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    print('???????????????????')
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(100000)
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={
                                                     "frames": (),

                                                     "txt": [None],
                                                     "txt_len": (),
                                                     "norm_mel": [None, 80],
                                                     "norm_stftm": [None, OUTPUT_SPEC_DIM]}, padding_values={

                                                     "frames": 0,

                                                     "txt": 0,
                                                     "txt_len": 0,
                                                     "norm_mel": norm_pad_mel_min,
                                                     "norm_stftm": norm_pad_stftm_min})
    return dataset

def init_next_batch():
    data_set = get_dataset(tfrecord_path)
    iterator = data_set.make_one_shot_iterator()
    next_item = iterator.get_next()
    return next_item

def get_next_batch(sess, next_item):
    t = sess.run(next_item)
    return t['txt'], t['txt_len'], t['norm_mel'], t['norm_stftm']

def draw_weighting_spec(raw_weighting, train_r, stftm, name):
    weighting = raw_weighting.repeat(train_r)
    print('--------------- :', weighting.shape, stftm.shape)
    ans = {'weighting':weighting, 'stftm':stftm}
    with open(name + '.pkl', "wb") as f:
        pkl.dump(ans, f)






if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_model.sess = sess
        writer = tf.summary.FileWriter("logs/")

        # train_summary = train_model.summary("train", 2)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)
        ass_style_token = tf.placeholder(name="ass_style_token", shape=(1, styles_kind, style_dim), dtype=tf.float32)
        ass_opt = train_model.single_style_token.assign(ass_style_token)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            train_model.saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
        else:
            print('no restor, init all include style:')
            np.random.seed(1)
            init_style_token = np.random.normal(0, 0.3, (1, styles_kind, style_dim))
            print('look random:', np.max(init_style_token), np.min(init_style_token))
            sess.run(ass_opt, feed_dict={ass_style_token:init_style_token})
        # linear_style_vec = np.zeros((1, styles_kind, style_dim), np.float32)
        # for i in range(styles_kind):
        #     for j in range(i * 26, min((i + 1) * 26, style_dim)):
        #         linear_style_vec[0][i][j] = 1
        # print(linear_style_vec)
        next_item = init_next_batch()

        total_loss = 0.
        total_batch_cnt = 0
        with open(pkl_path, "rb") as f:
            stats = pkl.load(f)
        try:
            for cnt in range(EPOCHS):
                pre_time = time.time()
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess, next_item)
                init_time_stamp = batch_mel_gtruth.shape[1]
                fix_time_stamp = init_time_stamp // train_r * train_r
                batch_mel_gtruth = batch_mel_gtruth[:, 0 : fix_time_stamp]
                batch_spec_gtruth = batch_spec_gtruth[:, 0: fix_time_stamp]
                print(batch_inp.shape, batch_inp_mask.shape, batch_mel_gtruth.shape, batch_spec_gtruth.shape,
                      )
                print(np.max(batch_inp), np.max(batch_inp_mask), np.max(batch_mel_gtruth), np.max(batch_spec_gtruth),
                      )
                print(np.min(batch_inp), np.min(batch_inp_mask), np.min(batch_mel_gtruth), np.min(batch_spec_gtruth),
                      )



                # mean_loss_holder = tf.placeholder(shape=(), dtype=tf.float32, name='mean_loss')
                # train_epoch_summary = tf.summary.scalar('epoch/train/loss', mean_loss_holder)




                print('start:', cnt, EPOCHS)
                train_time = time.time()
                print('pre time:', train_time - pre_time)


                if cnt % 100 == 0 and cnt > 0:
                    merged_summary_op = tf.summary.merge(train_model.sums)
                    raw_weighting, trained_style_token, summary_str, _, pred_out, loss_eval, global_step_eval = sess.run(
                        [train_model.weighting, train_model.single_style_token, merged_summary_op, train_upd, train_model.out_stftm, train_model.loss, global_step],
                        feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                   mel_gtruth: batch_mel_gtruth,
                                   spec_gtruth: batch_spec_gtruth})
                    writer.add_summary(summary_str, global_step_eval)


                    #draw img of weighting
                    #draw_weighting_spec(raw_weighting[0], train_r, pred_out[0], 'im' + str(global_step_eval))


                    all_pred_out = []
                    #generate general voice
                    pred_out = pred_out * stats["log_stftm_std"] + stats["log_stftm_mean"]
                    pred_audio, exp_spec = audio.invert_spectrogram(pred_out[0], 1.2)
                    pred_audio = np.reshape(pred_audio, (1, pred_audio.shape[-1]))
                    all_pred_out.append(pred_audio)
                    # pred_audio_summary_str = sess.run(train_model.pred_audio_summary, feed_dict={train_model.pred_audio_holder:pred_audio})
                    # writer.add_summary(pred_audio_summary_str, global_step_eval)
                    # generate unique style voice
                    for i in range(0, styles_kind, 3):
                        for tim in range(1, 4):
                            unique_style_token = np.copy(trained_style_token)
                            unique_style_token[0] += tim * trained_style_token[0][i]
                            sess.run(ass_opt, feed_dict={ass_style_token: unique_style_token})
                            pred_out_unique = sess.run(train_model.out_stftm,
                                                        feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                                                   mel_gtruth: batch_mel_gtruth,
                                                                   spec_gtruth: batch_spec_gtruth})
                            pred_out_unique = pred_out_unique * stats["log_stftm_std"] + stats["log_stftm_mean"]
                            pred_audio_unique, exp_spec_unique = audio.invert_spectrogram(pred_out_unique[0], 1.2)
                            pred_audio_unique = np.reshape(pred_audio_unique, (1, pred_audio_unique.shape[-1]))
                            all_pred_out.append(pred_audio_unique)
                            print('----', i, tim)

                    # unique_style_token = np.copy(trained_style_token)
                    # unique_style_token[0] += trained_style_token[0][0]



                    sess.run(ass_opt, feed_dict={ass_style_token: trained_style_token})

                    # pred_out_unique0 = pred_out_unique0 * stats["log_stftm_std"] + stats["log_stftm_mean"]
                    # pred_audio_unique0, exp_spec_unique = audio.invert_spectrogram(pred_out_unique0[0], 1.2)
                    # pred_audio_unique0 = np.reshape(pred_audio_unique0, (1, pred_audio_unique0.shape[-1]))

                    all_pred_out = np.concatenate(all_pred_out, axis=0)

                    pred_audio_summary_str = sess.run(train_model.pred_audio_summary,
                                                      feed_dict={train_model.pred_audio_holder: all_pred_out})
                    writer.add_summary(pred_audio_summary_str, global_step_eval)

                else:
                     _, loss_eval, global_step_eval = sess.run(
                        [train_upd, train_model.loss, global_step],
                        feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                   mel_gtruth: batch_mel_gtruth,
                                   spec_gtruth: batch_spec_gtruth})
                # summary_str = sess.run(merged_summary_op)





                total_loss += loss_eval
                total_batch_cnt += 1



                if cnt % 213 == 0 and cnt > 0:
                    t = total_loss / total_batch_cnt

                    with open('train_totol_loss.txt', 'a') as f:
                        print('\nglobal_step_eval---', global_step_eval, '\n', file=f)
                        print(t, file=f)
                        sys.stdout.flush()
                    total_loss = 0.
                    total_batch_cnt = 0


                post_time = time.time()



                print('train time:', post_time - train_time)
                gc.collect()

                # if global_step_eval % 50 == 0:
                #     train_sum_eval = sess.run(train_summary)
                #     writer.add_summary(train_sum_eval, global_step_eval)
                if global_step_eval % 200 == 0:
                    train_model.save(save_path, global_step_eval)
                    # np.savez(os.path.join(save_path, 'style_token_' + str(global_step_eval) + '.npz'), all_style = new_style)
                if global_step_eval == 100000:
                    break


                mean_loss = total_loss / BATCH_SIZE
                with open('train_loss.txt', 'a') as f:
                    f.write('{:f}\n'.format(loss_eval))
                # with open('train_style.txt', 'a') as f:
                #     # f = open('train_style.txt', 'a')
                #     print('\nglobal_step_eval---', global_step_eval, '\n', file=f)
                #     print(new_style, file=f)
                #     sys.stdout.flush()



                # train_epoch_summary_eval = sess.run(train_epoch_summary, feed_dict={mean_loss_holder: loss_eval})
                # writer.add_summary(train_epoch_summary_eval, cnt)
                print('post time:', time.time() - post_time)






        except Exception as e:
            print('Training stopped', str(e))


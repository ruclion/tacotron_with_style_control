import tensorflow as tf, pickle as pkl, os
import numpy as np
import random
import tqdm
import sys
import time
import hjk_tools.audio as audio
import math
import codecs
from best_tacotron.hyperparameter_style_debug import HyperParams
from best_tacotron.train_feedcontext2att_old_style_debug import Tacotron
# from tensorflow.python import debug as tf_debug


os.environ["CUDA_VISIBLE_DEVICES"] = ""
hp = HyperParams()


data_name = 'sr16_aB_2_style_debug'
save_path = os.path.join('model', data_name)
model_name = "TTS"
# tfrecord_train_path = './data/sr16_aB_sorted_train.tfrecords'
tfrecord_train_path = './data/sr16_aB_sorted_dev.tfrecords'
# pkl_train_path = './data/sr16_aB_sorted_train.pkl'
pkl_train_path = './data/sr16_aB_sorted_dev.pkl'
tb_logs_path = os.path.join('logs', data_name) + '/'
dev_txt_path = os.path.join('logs', data_name) + '/dev_loss.txt'



def parse_single_example(example_proto):
    features = {
                "frames": tf.FixedLenFeature([], tf.int64),
                "char_txt": tf.FixedLenFeature([], tf.string),
                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "log_mel_raw": tf.FixedLenFeature([], tf.string),
                "log_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    frames = tf.cast(parsed["frames"], tf.int32)
    char_txt = parsed["char_txt"]
    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    log_mel = tf.reshape(tf.decode_raw(parsed["log_mel_raw"], tf.float32), (frames, hp.seq2seq_dim))
    log_stftm = tf.reshape(tf.decode_raw(parsed["log_stftm_raw"], tf.float32), (frames, hp.post_dim))
    return {"frames": frames, "char_txt": char_txt, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}
    # return {"frames": frames, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}


def get_dataset(tfrecord_path, shuffle_buf, repeat_times):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.padded_batch(hp.batch_size, padded_shapes={
        "frames": (),
        "char_txt": (),
        "txt": [None],
        "txt_len": (),
        "log_mel": [None, hp.seq2seq_dim],
        "log_stftm": [None, hp.post_dim]}, padding_values={
        "frames": 0,
        "char_txt": "",
        "txt": np.int32(0),
        "txt_len": 0,
        "log_mel": np.float32(np.log(0.01)),
        "log_stftm": np.float32(np.log(0.01))})
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.repeat(repeat_times)

    # dataset = dataset.batch(hp.batch_size)


    return dataset

def init_next_batch(tfrecord_path, shuffle_buf, repeat_times):
    data_set = get_dataset(tfrecord_path, shuffle_buf, repeat_times)
    iterator = data_set.make_one_shot_iterator()
    next_item = iterator.get_next()
    return next_item

def get_next_batch(sess, next_item):
    t = sess.run(next_item)
    # print('frames:', t['frames'])
    return t['txt'], t['txt_len'], t['log_mel'], t['log_stftm']
def post_next_batch(batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, stats_meta):
    init_time_stamp = batch_mel_gtruth.shape[1]
    fix_time_stamp = init_time_stamp // hp.reduction_rate * hp.reduction_rate
    batch_mel_gtruth = batch_mel_gtruth[:, 0: fix_time_stamp]
    batch_spec_gtruth = batch_spec_gtruth[:, 0: fix_time_stamp]
    batch_mel_gtruth = (batch_mel_gtruth - stats_meta["log_mel_mean"]) / stats_meta["log_mel_std"]
    batch_spec_gtruth = (batch_spec_gtruth - stats_meta["log_stftm_mean"]) / stats_meta["log_stftm_std"]

    # print(batch_inp.shape, batch_inp_mask.shape, batch_mel_gtruth.shape, batch_spec_gtruth.shape,
    #       )
    # print(np.max(batch_inp), np.max(batch_inp_mask), np.max(batch_mel_gtruth), np.max(batch_spec_gtruth),
    #       )
    # print(np.min(batch_inp), np.min(batch_inp_mask), np.min(batch_mel_gtruth), np.min(batch_spec_gtruth),
    #       )
    return batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth

def get_style_token(trained_style_token, style_no):
    t = [0, 5, 9]
    unique_style_token = np.copy(trained_style_token)
    if style_no == 0:
        pass
    elif style_no == 1:
        unique_style_token = unique_style_token * 0
    else:
        tag = t[(style_no - 2) // 3]
        cat = (style_no - 2) % 3
        if cat == 0:
            unique_style_token += trained_style_token[0][tag]
        elif cat == 1:
            unique_style_token += trained_style_token[0][tag] * 2
        elif cat == 2:
            unique_style_token = unique_style_token * 0 + trained_style_token[0][tag]
    return unique_style_token





def main():

    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
        seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
        post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)

    train_meta_path = pkl_train_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)

    # dev_meta_path = pkl_dev_path
    # assert os.path.exists(dev_meta_path), \
    #     '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)

    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)
        train_meta['reduction_rate'] = hp.reduction_rate

    # with open(dev_meta_path, 'rb') as f:
    #     dev_meta = pkl.load(f)
    #     dev_meta['reduction_rate'] = hp.reduction_rate

    train_model = Tacotron(inp=inp, inp_mask=inp_mask, seq2seq_gtruth=seq2seq_gtruth, post_gtruth=post_gtruth,
                           hyper_params=hp, training=True, reuse=False)

    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(train_model.exp_learning_rate_decay(0.001))
        # grad, var = zip(*opt.compute_gradients(train_model.loss))
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     train_upd = opt.apply_gradients(zip(grad, var), global_step=train_model.global_step)

        grads_and_vars = opt.compute_gradients(train_model.loss)
        for i, (grad, var) in enumerate(grads_and_vars):
            # print(var.name)
            if var.name.find('style_token:0') != -1:
                grads_and_vars[i] = (grad / 200, var)
                print(var.name)
                print('hhhh time')
                break
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(grads_and_vars, global_step=train_model.global_step)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_model.sess = sess
        writer = tf.summary.FileWriter(tb_logs_path, filename_suffix='train')
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)
        saver = tf.train.Saver(max_to_keep=20)
        train_model.saver = saver
        ass_style_token = tf.placeholder(name="ass_style_token", shape=(1, hp.styles_kind, hp.style_dim), dtype=tf.float32)
        ass_opt = train_model.single_style_token.assign(ass_style_token)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
        else:
            print('no restor, init all include style:')
            np.random.seed(1)
            init_style_token = np.random.uniform(low=-1, high=1, size=(1, hp.styles_kind, hp.style_dim))
            print('look random:', np.max(init_style_token), np.min(init_style_token))
            sess.run(ass_opt, feed_dict={ass_style_token: init_style_token})

        train_next_item = init_next_batch(tfrecord_train_path, 1, 1)
        # dev_next_item = init_next_batch(tfrecord_dev_path, 1, 2000)


        t = {'style_token':sess.run(train_model.style_token)}
        with open('tmp.pkl', 'wb') as f:
            pkl.dump(t, f)
        train_scalar_summary = train_model.get_scalar_summary('train')
        train_alpha_summary = train_model.get_alpha_summary('train', 2)
        dev_loss_holder = tf.placeholder(shape=(), dtype=tf.float32, name='dev_loss')
        dev_loss_summary = tf.summary.scalar('dev_loss_summary', dev_loss_holder)
        pred_audio_holder = tf.placeholder(shape=(None, None), dtype=tf.float32, name='pred_audio')
        pred_audio_summary = tf.summary.audio('pred_audio_summary', pred_audio_holder,
                                                   sample_rate=hp.sample_rate, max_outputs=30)

        already_step_eval = sess.run(train_model.global_step)
        try:
            for cnt in range(1):
                # print('now is', cnt)
                # pre_time = time.time()
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess, train_next_item)

                # print('bug', batch_inp[0], 'len', batch_inp_mask[0], 'actual', batch_inp[0].shape)
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, train_meta)
                batch_inp = np.array([[1, 2, 0, 0], [2, 3, 4, 5]])
                batch_inp_mask = np.array([2, 4])
                # print(batch_inp_mask)
                # print('look', batch_mel_gtruth[0], batch_spec_gtruth[0])
                # train_time = time.time()
                # print('pre time:', train_time - pre_time)

                _, loss_eval, global_step_eval = sess.run(
                    [train_upd, train_model.loss, train_model.global_step],
                    feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                               seq2seq_gtruth: batch_mel_gtruth,
                               post_gtruth: batch_spec_gtruth})
                # print('step:', global_step_eval)



                # post_time = time.time()

                # print('train time:', post_time - train_time)


        except Exception as e:
            print('Training stopped', str(e))


if __name__ == '__main__':
    main()
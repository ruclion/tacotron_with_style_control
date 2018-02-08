import tensorflow as tf, pickle as pkl, os
import numpy as np
import random
import tqdm
import sys
import time
import hjk_tools.audio
import math
import codecs
from best_tacotron.hyperparameter import HyperParams
from best_tacotron.train_feedcontext2att_old import Tacotron

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
hp = HyperParams()


data_name = 'sr16_aB_1'
save_path = os.path.join('model', data_name)
model_name = "TTS"
tfrecord_train_path = './data/sr16_aB_train.tfrecords'
tfrecord_dev_path = './data/sr16_aB_dev.tfrecords'
pkl_train_path = './data/sr16_aB_train.pkl'
pkl_dev_path = './data/sr16_aB_dev.pkl'
tb_logs_path = os.path.join('logs', data_name) + '/'
dev_txt_path = os.path.join('logs', data_name) + '/dev_loss.txt'



def parse_single_example(example_proto):
    features = {
                "frames": tf.FixedLenFeature([], tf.int64),
                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "log_mel_raw": tf.FixedLenFeature([], tf.string),
                "log_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    frames = tf.cast(parsed["frames"], tf.int32)
    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    log_mel = tf.reshape(tf.decode_raw(parsed["log_mel_raw"], tf.float32), (frames, hp.seq2seq_dim))
    log_stftm = tf.reshape(tf.decode_raw(parsed["log_stftm_raw"], tf.float32), (frames, hp.post_dim))
    return {"frames": frames, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}


def get_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.repeat(1000)
    dataset = dataset.shuffle(10000)
    # dataset = dataset.batch(hp.batch_size)

    dataset = dataset.padded_batch(hp.batch_size, padded_shapes={
                                                     "frames": (),
                                                     "txt": [None],
                                                     "txt_len": (),
                                                     "log_mel": [None, hp.seq2seq_dim],
                                                     "log_stftm": [None, hp.post_dim]}, padding_values={
                                                     "frames": 0,
                                                     "txt": np.int32(0),
                                                     "txt_len": 0,
                                                     "log_mel": np.float32(np.log(0.01)),
                                                     "log_stftm": np.float32(np.log(0.01))})
    return dataset

def init_next_batch(tfrecord_path):
    data_set = get_dataset(tfrecord_path)
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






def main():

    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
        seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
        post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)

    train_meta_path = pkl_train_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)

    dev_meta_path = pkl_dev_path
    assert os.path.exists(dev_meta_path), \
        '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)

    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)
        train_meta['reduction_rate'] = hp.reduction_rate

    with open(dev_meta_path, 'rb') as f:
        dev_meta = pkl.load(f)
        dev_meta['reduction_rate'] = hp.reduction_rate

    train_model = Tacotron(inp=inp, inp_mask=inp_mask, seq2seq_gtruth=seq2seq_gtruth, post_gtruth=post_gtruth,
                           hyper_params=hp, training=True, reuse=False)

    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(train_model.exp_learning_rate_decay(0.001))
        grad, var = zip(*opt.compute_gradients(train_model.loss))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(zip(grad, var), global_step=train_model.global_step)

    train_batches_per_epoch = math.ceil(len(train_meta['key_lst']) / hp.batch_size)
    dev_batches_per_epoch = math.ceil(len(dev_meta['key_lst']) / hp.batch_size)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_model.sess = sess
        writer = tf.summary.FileWriter(tb_logs_path, filename_suffix='train')
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)
        saver = tf.train.Saver()
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
        else:
            print('no restor, init all')

        train_next_item = init_next_batch(tfrecord_train_path)
        dev_next_item = init_next_batch(tfrecord_dev_path)
        with open(pkl_train_path, "rb") as f:
            train_stats = pkl.load(f)
        with open(pkl_dev_path, "rb") as f:
            dev_stats = pkl.load(f)

        train_scalar_summary = train_model.get_scalar_summary('train')
        train_alpha_summary = train_model.get_alpha_summary('train', 1)
        pred_audio_holder = tf.placeholder(shape=(None, None), dtype=tf.float32, name='pred_audio')
        pred_audio_summary = tf.summary.audio('pred_audio_summary', pred_audio_holder,
                                                   sample_rate=hp.sample_rate, max_outputs=8)

        already_step_eval = sess.run(train_model.global_step)
        try:
            for cnt in tqdm.tqdm(range(1, hp.max_global_steps + 10 - already_step_eval)):
                # print('now is', cnt)
                # pre_time = time.time()
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess, train_next_item)
                # print('bug', batch_inp[0], 'len', batch_inp_mask[0], 'actual', batch_inp[0].shape)
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, train_meta)
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

                if cnt % 50 == 0:
                    summary_str = sess.run(train_scalar_summary,
                        feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                   seq2seq_gtruth: batch_mel_gtruth,
                                   post_gtruth: batch_spec_gtruth})
                    writer.add_summary(summary_str, global_step_eval)
                if cnt % 200 == 0:#about one epoch
                    summary_str = sess.run(train_alpha_summary,
                                           feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                                      seq2seq_gtruth: batch_mel_gtruth,
                                                      post_gtruth: batch_spec_gtruth})
                    writer.add_summary(summary_str, global_step_eval)
                    dev_loss = 0
                    for dev_i in range(dev_batches_per_epoch):
                        batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess,
                                                                                                        dev_next_item)
                        batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp,
                                                                                                         batch_inp_mask,
                                                                                                         batch_mel_gtruth,
                                                                                                         batch_spec_gtruth,
                                                                                                         dev_meta)
                        _loss = sess.run(train_model.loss,
                        feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                   seq2seq_gtruth: batch_mel_gtruth,
                                   post_gtruth: batch_spec_gtruth})
                        dev_loss += _loss
                    dev_loss /= dev_batches_per_epoch
                    with codecs.open(dev_txt_path, 'a+') as f:
                        f.write('{:.6f}\n'.format(dev_loss))

                if cnt % 2000 == 0:
                    train_model.save(save_path, global_step_eval)
                    pred_out = sess.run(train_model.post_output, feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                                      seq2seq_gtruth: batch_mel_gtruth,
                                                      post_gtruth: batch_spec_gtruth})
                    all_pred_out = []
                    # generate general voice
                    pred_out = pred_out * train_stats["log_stftm_std"] + train_stats["log_stftm_mean"]
                    for audio_i in range(8):
                        pred_audio, exp_spec = audio.invert_spectrogram(pred_out[audio_i], 1.2)
                        pred_audio = np.reshape(pred_audio, (1, pred_audio.shape[-1]))
                        all_pred_out.append(pred_audio)

                    all_pred_out = np.concatenate(all_pred_out, axis=0)

                    pred_audio_summary_str = sess.run(pred_audio_summary,
                                                      feed_dict={pred_audio_holder: all_pred_out})
                    writer.add_summary(pred_audio_summary_str, global_step_eval)

                # post_time = time.time()

                # print('train time:', post_time - train_time)


        except Exception as e:
            print('Training stopped', str(e))


if __name__ == '__main__':
    main()

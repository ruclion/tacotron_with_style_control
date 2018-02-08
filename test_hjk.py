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
from best_tacotron.generate_feedcontext2att_old import Tacotron
import scipy.io.wavfile as siowav

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
generate_path = os.path.join('logs', data_name) + '/generate/'



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
        decode_time_steps = tf.placeholder(name='decode_time_steps', shape=(), dtype=tf.int32)



    dev_meta_path = pkl_dev_path
    assert os.path.exists(dev_meta_path), \
        '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)



    with open(dev_meta_path, 'rb') as f:
        dev_meta = pkl.load(f)
        dev_meta['reduction_rate'] = hp.reduction_rate

    model = Tacotron(inp, inp_mask, decode_time_steps, hyper_params=hp)


    dev_batches_per_epoch = math.ceil(len(dev_meta['key_lst']) / hp.batch_size)
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.sess = sess
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)
        saver = tf.train.Saver()
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
        else:
            print('no restor, init all')

        # train_next_item = init_next_batch(tfrecord_train_path)
        dev_next_item = init_next_batch(tfrecord_dev_path)

        with open(pkl_dev_path, "rb") as f:
            dev_stats = pkl.load(f)

        for dev_i in range(dev_batches_per_epoch):
            batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess,
                                                                                            dev_next_item)
            batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp,
                                                                                             batch_inp_mask,
                                                                                             batch_mel_gtruth,
                                                                                             batch_spec_gtruth,
                                                                                             dev_meta)
            pred_out, alpha_out = sess.run([model.post_output, model.alpha_output],
                             feed_dict={inp: batch_inp, inp_mask: batch_inp_mask, decode_time_steps: 100})
            all_pred_out = []
            # generate general voice
            pred_out = pred_out * dev_stats["log_stftm_std"] + dev_stats["log_stftm_mean"]
            for audio_i in range(8):
                pred_audio, exp_spec = audio.invert_spectrogram(pred_out[audio_i], 1.2)
                siowav.write(os.path.join(generate_path, "random%d.wav" % (audio_i)), hp.sample_rate, pred_audio)
                # pred_audio = np.reshape(pred_audio, (1, pred_audio.shape[-1]))


                # all_pred_out.append(pred_audio)

            # all_pred_out = np.concatenate(all_pred_out, axis=0)
            break


if __name__ == '__main__':
    main()

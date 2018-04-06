import tensorflow as tf, pickle as pkl, os
import numpy as np
import random
import tqdm
import sys
import time
import hjk_tools.audio as audio
import math
import codecs
import copy
from best_tacotron.hyperparameter_style import HyperParams
from best_tacotron.train_model_sentence_style3 import Tacotron
import scipy.io.wavfile as siowav

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hp = HyperParams()


data_name = 'sr16_aB_sentence_style3'
save_path = os.path.join('model', data_name)
model_name = "TTS"
tfrecord_train_path = './data/sr16_aB_sorted_train.tfrecords'
tfrecord_dev_path = './data/sr16_aB_sorted_dev.tfrecords'
pkl_train_id_path = './data/sr16_aB_sorted_train_id.pkl'
pkl_train_path = './data/sr16_aB_sorted_train.pkl'
pkl_dev_path = './data/sr16_aB_sorted_dev.pkl'
tb_logs_path = os.path.join('logs', data_name) + '/generate_log/'
dev_txt_path = os.path.join('logs', data_name) + '/dev_loss.txt'
generate_path = os.path.join('logs', data_name) + '/generate_select/'



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
    return t['txt'], t['txt_len'], t['log_mel'], t['log_stftm'], t['char_txt']

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
    unique_style_token = np.copy(trained_style_token)
    if style_no < 70:
        tag = style_no // 7
        cat = style_no % 7
        if cat == 0:
            unique_style_token += trained_style_token[0][tag] * 0.3
        elif cat == 1:
            unique_style_token += trained_style_token[0][tag] * 0.4
        elif cat == 2:
            unique_style_token += trained_style_token[0][tag] * 0.5
        elif cat == 3:
            unique_style_token += trained_style_token[0][tag] * 0.6
        elif cat == 4:
            unique_style_token += trained_style_token[0][tag] * 0.1
        elif cat == 5:
            unique_style_token += trained_style_token[0][tag] * 0.2
        elif cat == 6:
            unique_style_token = unique_style_token * 0 + trained_style_token[0][tag]
    else:
        if style_no == 70:
            pass
        elif style_no == 71:
            unique_style_token = unique_style_token * 0


    return unique_style_token

def get_style_attention(style_no):
    unique_style_token = np.zeros([32, 10], dtype=np.float32)
    for i in range(32):
        if style_no < 70:
            tag = style_no // 7
            cat = style_no % 7
            if cat == 0:
                unique_style_token[i][tag] += 0.1
            elif cat == 1:
                unique_style_token[i][tag] += 0.2
            elif cat == 2:
                unique_style_token[i][tag] += 0.3
            elif cat == 3:
                unique_style_token[i][tag] += 0.4
            elif cat == 4:
                unique_style_token[i][tag] += 0.5
            elif cat == 5:
                unique_style_token[i][tag] += 0.7
            elif cat == 6:
                unique_style_token[i][tag] += 1.0
        else:
            if style_no == 70:
                pass
            elif style_no == 71:
                pass

    return unique_style_token




def main():

    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
        inp_id = tf.placeholder(name='inp_id', shape=(None,), dtype=tf.int32)
        seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
        post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)



    dev_meta_path = pkl_dev_path
    assert os.path.exists(dev_meta_path), \
        '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)



    with open(dev_meta_path, 'rb') as f:
        dev_meta = pkl.load(f)
        dev_meta['reduction_rate'] = hp.reduction_rate
    print(dev_meta.keys())
    dev_char_map = dev_meta['char_map']

    # txt = ["She glanced at his newspaper, then stopped and stared.",
    #        "I think you'll have to marry Count Paris.",
    #        "My house is the best of all!"]

    txt = ["She glanced at his newspaper, then stopped and stared.",
           "I think you'll have to marry Count Paris.",
           "My house is the best of all!"]

    # print('**', txt[0][0])
    max_txt_len = 0
    for i in range(len(txt)):
        max_txt_len = max(max_txt_len, len(txt[i]))
    txt_inp = []
    for i in range(len(txt)):
        txt_inp_a = []
        for j in range(len(txt[i])):
            # print('---:', txt[i][j])
            txt_inp_a.append(dev_char_map[txt[i][j]])
        for j in range(len(txt[i]), max_txt_len):
            txt_inp_a.append(0)
        txt_inp.append(txt_inp_a)
    txt_inp = np.asarray(txt_inp)

    # print(txt_inp)
    txt_mask = []
    for i in range(len(txt)):
        txt_mask.append(len(txt[i]))
    txt_mask = np.asarray(txt_mask)
    # print(txt_mask)

    model = Tacotron(inp=inp, inp_mask=inp_mask, inp_id=inp_id, seq2seq_gtruth=seq2seq_gtruth,
                           post_gtruth=post_gtruth,
                           hyper_params=hp, training=False, reuse=False)


    dev_batches_per_epoch = math.ceil(len(dev_meta['key_lst']) / hp.batch_size)
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.sess = sess
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)

        saver = tf.train.Saver(max_to_keep=20)
        model.saver = saver
        # ass_style_token = tf.placeholder(name="ass_style_token", shape=(1, hp.styles_kind, hp.style_dim),
        #                                  dtype=tf.float32)
        # ass_opt = model.single_style_token.assign(ass_style_token)
        # ass_inp_att = tf.placeholder(name="ass_inp_att", shape=(None, hp.styles_kind),
        #                              dtype=tf.float32)
        # att_ass_opt = model.inp_att.assign(ass_inp_att)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
        else:
            print('no restor, init all')

        wav_folder = os.path.join(generate_path, data_name)


        with open('selected_data.pkl', 'rb') as f:
            selected_data = pkl.load(f)

        print('no ctr')
        for id_no in range(5):
            now_id_no = selected_data['id'][id_no]
            for i in range(5):
                pred_out = sess.run(model.post_output,
                            feed_dict={inp: [selected_data['batch_inp'][i]],
                                       inp_mask: [selected_data['batch_inp_mask'][i]],
                                       inp_id: [now_id_no],
                                       seq2seq_gtruth: [selected_data['batch_mel_gtruth'][i]],
                                       post_gtruth: [selected_data['batch_spec_gtruth'][i]]})
                pred_out = pred_out * dev_meta["log_stftm_std"] + dev_meta["log_stftm_mean"]
                for j in range(1):
                    pred_audio, exp_spec = audio.invert_spectrogram(pred_out[j], 1.2)
                    # wav_folder = os.path.join(generate_path, data_name)
                    if not os.path.exists(wav_folder):
                        os.makedirs(wav_folder)
                    siowav.write(os.path.join(wav_folder, "audio%d_style_%d.wav" % (i, now_id_no)), hp.sample_rate,
                                 pred_audio)

        '''
        #ctr, no style
        print('no style')
        unique_style_attention = np.zeros([len(txt_inp), 10], dtype=np.float32)
        # sess.run(att_ass_opt, feed_dict={ass_inp_att: unique_style_attention})
        pred_out = sess.run(model.post_output, feed_dict={inp: txt_inp, inp_mask: txt_mask, inp_att: unique_style_attention,
                                                          decode_time_steps: 60,
                                                          })
        pred_out = pred_out * dev_meta["log_stftm_std"] + dev_meta["log_stftm_mean"]
        for j in range(len(txt)):
            pred_audio, exp_spec = audio.invert_spectrogram(pred_out[j], 1.2)
            # wav_folder = os.path.join(generate_path, data_name)
            if not os.path.exists(wav_folder):
                os.makedirs(wav_folder)
            siowav.write(os.path.join(wav_folder, "audio%d_style_%d.wav" % (j, 200)), hp.sample_rate,
                         pred_audio)

        #ctr, spec style
        for i in range(10):
            print('style', i)
            unique_style_attention = np.zeros([len(txt_inp), 10], dtype=np.float32)
            for j in range(len(txt_inp)):
                unique_style_attention[j][i] = 0.5
            # sess.run(att_ass_opt, feed_dict={ass_inp_att: unique_style_attention})
            pred_out = sess.run(model.post_output, feed_dict={inp: txt_inp, inp_mask: txt_mask, inp_att: unique_style_attention,
                                                              decode_time_steps: 60,
                                                              })
            pred_out = pred_out * dev_meta["log_stftm_std"] + dev_meta["log_stftm_mean"]
            for j in range(len(txt)):
                pred_audio, exp_spec = audio.invert_spectrogram(pred_out[j], 1.2)
                # wav_folder = os.path.join(generate_path, data_name)
                if not os.path.exists(wav_folder):
                    os.makedirs(wav_folder)
                siowav.write(os.path.join(wav_folder, "audio%d_style_%d.wav" % (j, i)), hp.sample_rate,
                             pred_audio)

        '''








if __name__ == '__main__':
    main()

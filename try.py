import numpy as np

a = np.array([1, 2, 3])
a = np.minimum(a, 2)
print(a)

'''
import tensorflow as tf, pickle as pkl, os
import numpy as np
import random
import tqdm
import sys
import time
import hjk_tools.audio as audio
import math
import codecs
from best_tacotron.hyperparameter_style import HyperParams
from best_tacotron.train_model_sentence_style3_label import Tacotron

#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hp = HyperParams()

data_name = '111iemocap_1_sentence_style3_label'
save_path = os.path.join('model', data_name)
model_name = "TTS"
tfrecord_train_path = './data/iemocap_1.tfrecords'
pkl_train_path = './data/iemocap_1.pkl'
tb_logs_path = os.path.join('logs', data_name) + '/'



def parse_single_example(example_proto):
    features = {"key": tf.FixedLenFeature([], tf.string),
                "frames": tf.FixedLenFeature([], tf.int64),
                "char_txt": tf.FixedLenFeature([], tf.string),
                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "style_label":tf.FixedLenFeature([], tf.int64),
                "log_mel_raw": tf.FixedLenFeature([], tf.string),
                "log_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    key = parsed["key"]
    frames = tf.cast(parsed["frames"], tf.int32)
    char_txt = parsed["char_txt"]
    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    style_label = tf.cast(parsed["style_label"], tf.int32)
    log_mel = tf.reshape(tf.decode_raw(parsed["log_mel_raw"], tf.float32), (frames, hp.seq2seq_dim))
    log_stftm = tf.reshape(tf.decode_raw(parsed["log_stftm_raw"], tf.float32), (frames, hp.post_dim))
    return {"key":key, "frames": frames, "char_txt": char_txt, "txt":txt, "txt_len":txt_len, "style_label":style_label, "log_mel": log_mel, "log_stftm": log_stftm}
    # return {"frames": frames, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}


def get_dataset(tfrecord_path, shuffle_buf, repeat_times):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.padded_batch(hp.batch_size, padded_shapes={
        "key": (),
        "frames": (),
        "char_txt": (),
        "txt": [None],
        "txt_len": (),
        "style_label": (),
        "log_mel": [None, hp.seq2seq_dim],
        "log_stftm": [None, hp.post_dim]}, padding_values={
        "key": "",
        "frames": 0,
        "char_txt": "",
        "txt": np.int32(0),
        "txt_len": 0,
        "style_label": 0,
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
    # while t['txt'].shape[0] != 32:
    #     t = sess.run(next_item)
    #     print('not 32 batch happen')
    # print('frames:', t['frames'])
    return t['txt'], t['txt_len'], t['style_label'], t['log_mel'], t['log_stftm'], t['key']
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
            unique_style_token += trained_style_token[0][tag] * 0.5
        elif cat == 1:
            unique_style_token += trained_style_token[0][tag] * 1
        elif cat == 2:
            unique_style_token = unique_style_token * 0 + trained_style_token[0][tag]
    return unique_style_token

# def linear_change_to_one(vec):
#     vec = np.maximum(0, vec)
#     sum = np.sum(vec)
#     vec = vec / sum
#     return vec
def get_id_from_key(key_list, data_dict):
    ans = []
    for var in key_list:
        key = var.decode()
        ans.append(data_dict[key])
    ans = np.asarray(ans)
    return ans

# def get_att_from_key(key_list, data_dict):
#     ans = []
#     for var in key_list:
#         key = var.decode()
#         # print('here???')
#         # print(key)
#         if key in data_dict:
#             pass
#         else:
#             # print('this')
#             data_dict[key] = linear_change_to_one(np.random.uniform(low=0, high=1, size=(hp.styles_kind)))
#         ans.append(data_dict[key])
#     ans = np.asarray(ans)
#     return ans
# def change_dict_from_return(key_list, return_att, data_dict):
#     for i, var in enumerate(key_list):
#         var = var.decode()
#         data_dict[var] = linear_change_to_one(return_att[i])

def main():
    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
        inp_id = tf.placeholder(name='inp_id', shape=(None,), dtype=tf.int32)
        seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
        post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)

    train_meta_path = pkl_train_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)


    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)
        train_meta['reduction_rate'] = hp.reduction_rate


    train_model = Tacotron(inp=inp, inp_mask=inp_mask, seq2seq_gtruth=seq2seq_gtruth, post_gtruth=post_gtruth,
                           hyper_params=hp, training=True, reuse=False)

    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(train_model.exp_learning_rate_decay(0.001))
        # grad, var = zip(*opt.compute_gradients(train_model.loss))
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     train_upd = opt.apply_gradients(zip(grad, var), global_step=train_model.global_step)

        grads_and_vars = opt.compute_gradients(train_model.loss)
        # for i, (grad, var) in enumerate(grads_and_vars):
        #     # print(var.name)
        #     if var.name.find('style_token:0') != -1:
        #         grads_and_vars[i] = (grad / 200.0, var)
        #         print(var.name)
        #         print('hhhh time')
        #         break
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(grads_and_vars, global_step=train_model.global_step)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(tb_logs_path):
        os.makedirs(tb_logs_path)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    # sess = tf.Session(config=config)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=config) as sess:
        train_model.sess = sess
        writer = tf.summary.FileWriter(tb_logs_path, filename_suffix='train')
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(save_path)
        saver = tf.train.Saver(max_to_keep=20)
        train_model.saver = saver
        # ass_style_token = tf.placeholder(name="ass_style_token", shape=(1, hp.styles_kind, hp.style_dim), dtype=tf.float32)
        # ass_opt = train_model.single_style_token.assign(ass_style_token)
        # ass_inp_att = tf.placeholder(name="ass_inp_att", shape=(None, hp.styles_kind),
        #                                  dtype=tf.float32)
        # att_ass_opt = train_model.inp_att.assign(ass_inp_att)

        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(save_path, ckpt_name))
            print('restore path:', ckpt_name)
            # with open(save_data_att_path, "rb") as f:
            #     data_style_att_dict = pkl.load(f)
            # print('load att dict')
        else:
            print('no restor, init all include style:')
            # np.random.seed(1)
            init_style_token = np.random.uniform(low=-0.01, high=0.01, size=(1, hp.styles_kind, hp.style_dim))
            print('look random:', np.max(init_style_token), np.min(init_style_token))
            # sess.run(ass_opt, feed_dict={ass_style_token: init_style_token})
            # data_style_att_dict = dict()



        train_next_item = init_next_batch(tfrecord_train_path, 501, 2000)
        # dev_next_item = init_next_batch(tfrecord_dev_path, 1, 2000)


        # train_scalar_summary = train_model.get_scalar_summary('train')
        # train_alpha_summary = train_model.get_alpha_summary('train', 2)
        dev_loss_holder = tf.placeholder(shape=(), dtype=tf.float32, name='dev_loss')
        dev_loss_summary = tf.summary.scalar('dev_loss_summary', dev_loss_holder)
        pred_audio_holder = tf.placeholder(shape=(None, None), dtype=tf.float32, name='pred_audio')
        pred_audio_summary = tf.summary.audio('pred_audio_summary', pred_audio_holder,
                                                   sample_rate=hp.sample_rate, max_outputs=30)

        already_step_eval = sess.run(train_model.global_step)
        try:
            for cnt in tqdm.tqdm(range(already_step_eval + 1, hp.max_global_steps + 10)):
                # print('now is', cnt)
                pre_time = time.time()
                batch_inp, batch_inp_mask, batch_style_label, batch_mel_gtruth, batch_spec_gtruth, batch_key = get_next_batch(sess, train_next_item)
                # print('bug', batch_inp[0], 'len', batch_inp_mask[0], 'actual', batch_inp[0].shape)
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, train_meta)
                # batch_id = get_id_from_key(batch_key, data_id_dict)
                # print(batch_id)
                # print(batch_key)
                # for var in batch_key:
                #     print(var.decode('utf-8'))
                # # print(batch_key.decode('utf-8'))
                #
                # return
                # print(batch_mel_gtruth.shape[1], batch_inp[0][0])
                # print(batch_inp_mask)
                # print('look', batch_mel_gtruth[0].shape, batch_spec_gtruth[0].shape)
                train_time = time.time()
                # print('pre time:', train_time - pre_time)

                # sess.run(att_ass_opt, feed_dict={ass_inp_att:batch_att})
                _, loss_eval, global_step_eval = sess.run(
                    [train_upd, train_model.loss, train_model.global_step],
                    feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                               seq2seq_gtruth: batch_mel_gtruth,
                               post_gtruth: batch_spec_gtruth})
                # return_att = sess.run(train_model.inp_att)
                # change_dict_from_return(batch_key, return_att, data_style_att_dict)
                # print('step:', global_step_eval)

                # if cnt % 50 == 0:
                # # if cnt % 5 == 0:
                #     summary_str = sess.run(train_scalar_summary,
                #         feed_dict={inp: batch_inp, inp_mask: batch_inp_mask, inp_id: batch_style_label,
                #                    seq2seq_gtruth: batch_mel_gtruth,
                #                    post_gtruth: batch_spec_gtruth})
                #     writer.add_summary(summary_str, global_step_eval)
                # if cnt % 200 == 0:#about one epoch
                #     summary_str = sess.run(train_alpha_summary,
                #                            feed_dict={inp: batch_inp, inp_mask: batch_inp_mask, inp_id: batch_style_label,
                #                                       seq2seq_gtruth: batch_mel_gtruth,
                #                                       post_gtruth: batch_spec_gtruth})
                #     writer.add_summary(summary_str, global_step_eval)
                    # dev_loss = 0
                    # dev_batches_per_epoch = 0
                    # dev_next_item = init_next_batch(tfrecord_dev_path, 1000, 1)#use the last batch to listen feel
                    # while True:
                    #     try:
                    #         batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = get_next_batch(sess,
                    #                                                                                         dev_next_item)
                    #         batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp,
                    #                                                                                          batch_inp_mask,
                    #                                                                                          batch_mel_gtruth,
                    #                                                                                          batch_spec_gtruth,
                    #                                                                                          dev_meta)
                    #         _loss = sess.run(train_model.loss,
                    #                          feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                    #                                     seq2seq_gtruth: batch_mel_gtruth,
                    #                                     post_gtruth: batch_spec_gtruth})
                    #         dev_loss += _loss
                    #         dev_batches_per_epoch += 1
                    #     except:
                    #         dev_loss /= dev_batches_per_epoch
                    #         dev_loss_summary_str = sess.run(dev_loss_summary,
                    #                                           feed_dict={dev_loss_holder: dev_loss})
                    #         writer.add_summary(dev_loss_summary_str, global_step_eval)
                    #         break
                if cnt % 1000 == 0:
                    train_model.save(save_path, global_step_eval)
                    all_pred_out = []
                    # sess.run(ass_opt, feed_dict={ass_style_token: unique_style_token})
                    pred_out = sess.run(train_model.post_output, feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                                                                            inp_id: batch_style_label,
                                                                            seq2seq_gtruth: batch_mel_gtruth,
                                                                            post_gtruth: batch_spec_gtruth})
                    pred_out = pred_out * train_meta["log_stftm_std"] + train_meta["log_stftm_mean"]
                    for audio_i in range(3):
                        pred_audio, exp_spec = audio.invert_spectrogram(pred_out[audio_i], 1.2)
                        pred_audio = np.reshape(pred_audio, (1, pred_audio.shape[-1]))
                        all_pred_out.append(pred_audio)
                    all_pred_out = np.concatenate(all_pred_out, axis=0)

                    pred_audio_summary_str = sess.run(pred_audio_summary,
                                                      feed_dict={pred_audio_holder: all_pred_out})
                    writer.add_summary(pred_audio_summary_str, global_step_eval)


                    # trained_style_token = sess.run(train_model.single_style_token)
                    # for style_no in range(11):
                    #     unique_style_token = get_style_token(trained_style_token, style_no)
                    #     sess.run(ass_opt, feed_dict={ass_style_token: unique_style_token})
                    #     pred_out = sess.run(train_model.post_output, feed_dict={inp: batch_inp, inp_mask: batch_inp_mask,
                    #                                                             inp_id: batch_style_label,
                    #                                       seq2seq_gtruth: batch_mel_gtruth,
                    #                                       post_gtruth: batch_spec_gtruth})
                    #     pred_out = pred_out * train_meta["log_stftm_std"] + train_meta["log_stftm_mean"]
                    #     for audio_i in range(3):
                    #         pred_audio, exp_spec = audio.invert_spectrogram(pred_out[audio_i], 1.2)
                    #         pred_audio = np.reshape(pred_audio, (1, pred_audio.shape[-1]))
                    #         all_pred_out.append(pred_audio)
                    # inp_all_pred_out = []
                    # for m in range(3):
                    #     for x in range(30):
                    #         if x % 3 == m:
                    #             inp_all_pred_out.append(all_pred_out[x])
                    #
                    # all_pred_out = np.concatenate(inp_all_pred_out, axis=0)
                    #
                    #
                    # pred_audio_summary_str = sess.run(pred_audio_summary,
                    #                                   feed_dict={pred_audio_holder: all_pred_out})
                    # writer.add_summary(pred_audio_summary_str, global_step_eval)
                    # sess.run(ass_opt, feed_dict={ass_style_token: trained_style_token})

                post_time = time.time()

                # print('train time:', post_time - train_time)


        except Exception as e:
            print('Training stopped', str(e))


if __name__ == '__main__':
    main()

'''



'''
import tensorflow as tf, pickle as pkl, os
import numpy as np
from best_tacotron.hyperparameter_style import HyperParams

hp = HyperParams()

tfrecord_train_path = './data/sr16_aB_sorted_train.tfrecords'
tfrecord_dev_path = './data/sr16_aB_sorted_dev.tfrecords'
pkl_train_path = './data/sr16_aB_sorted_train.pkl'
pkl_dev_path = './data/sr16_aB_sorted_dev.pkl'

def parse_single_example(example_proto):
    features = {"key": tf.FixedLenFeature([], tf.string),
                "frames": tf.FixedLenFeature([], tf.int64),
                "char_txt": tf.FixedLenFeature([], tf.string),
                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "log_mel_raw": tf.FixedLenFeature([], tf.string),
                "log_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    key = parsed["key"]
    frames = tf.cast(parsed["frames"], tf.int32)
    char_txt = parsed["char_txt"]
    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    log_mel = tf.reshape(tf.decode_raw(parsed["log_mel_raw"], tf.float32), (frames, hp.seq2seq_dim))
    log_stftm = tf.reshape(tf.decode_raw(parsed["log_stftm_raw"], tf.float32), (frames, hp.post_dim))
    return {"key":key, "frames": frames, "char_txt": char_txt, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}
    # return {"frames": frames, "txt":txt, "txt_len":txt_len, "log_mel": log_mel, "log_stftm": log_stftm}


def get_dataset(tfrecord_path, shuffle_buf, repeat_times):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.padded_batch(hp.batch_size, padded_shapes={
        "key": (),
        "frames": (),
        "char_txt": (),
        "txt": [None],
        "txt_len": (),
        "log_mel": [None, hp.seq2seq_dim],
        "log_stftm": [None, hp.post_dim]}, padding_values={
        "key": "",
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
    # while t['txt'].shape[0] != 32:
    #     t = sess.run(next_item)
    #     print('not 32 batch happen')
    # print('frames:', t['frames'])
    return t['txt'], t['txt_len'], t['log_mel'], t['log_stftm'], t['key']
sess = tf.Session()
train_next_item = init_next_batch(tfrecord_train_path, 1, 1)
data_id_dict = dict()
cnt = 0
try:
    while True:
        batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, batch_key = get_next_batch(sess, train_next_item)
        for var in batch_key:
            key = var.decode()
            data_id_dict[key] = cnt
            cnt += 1
except:
    print(data_id_dict)

with open('sr16_aB_sorted_train_id.pkl', "wb") as f:
    pkl.dump(data_id_dict, f)
'''
'''
import numpy as np
import os
d = dict()
d['a'] = 1
d['b'] = 2
x = np.array(['a', 'b'])
t = list(d.keys())
print(t[0], type(t))
ans = 0
for i in range(1, 6):
    ans += (6 - i) * (10 - i)
print(ans)

x = [1, 2, 3, 4, 5]
y = list(map(lambda s:s, x))
print(y)
print(os.path.abspath('.'))
s = '../tacotron_with_style_control/data/audioBook/All_Slices_wav_24k/TheRunawayPancake/19_sr16k.wav'
s_p = s.split('/')
print(s.split('/'))
src = 'D:/hjk/tacotron_with_style_control/data/All_Slices_wav_16k'
src = src + '/' + s_p[-2] + '/' + s_p[-1]
dst = 'D:/hjk/tacotron_with_style_control/document/tot_unsp'
class_num = 0
dst = dst + '/class' + str(class_num) + '/' + s_p[-2] + '_' + s_p[-1]
print(dst)
import csv
with open("test.csv", "w") as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow({"index", "a_name", "b_name"})
    # 写入多行用writerows
    writer.writerows([[0, 1, 3], [1, 2, 3], [2, 3, 4]])
import pandas as pd

#任意的多组列表
a = [1,2,3]
b = [4,5,6]

#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name':a,'b_name':b})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("test22.csv",index=False,sep='/t')
'''
'''
import numpy as np
import pickle as pkl
import os
import numpy as np
import scipy.io.wavfile as siowav
import pickle as pkl
import librosa
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import tqdm
from matplotlib import pyplot



ans = 0
for i in range(1, 10000):
    t = str(i)
    for j in t:
        if j == '0':
            ans += 1
print(ans)
def drawHist(heights):
    #创建直方图
    #第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    #第二个参数为划分的区间个数
    pyplot.hist(heights, 100)
    pyplot.xlabel('Attention weights')
    pyplot.ylabel('Frequency')
    pyplot.title('Max weight for Each Sentence')
    pyplot.show()



with open('style_dict_2000.pkl', "rb") as f:
    data_style_att_dict = pkl.load(f)
# print(data_style_att_dict)

t = []
cnt = 0
for i, var in enumerate(data_style_att_dict):
    print(var)
    print('first:', data_style_att_dict[var])
    break
# print('first:', data_style_att_dict[0])
for var in data_style_att_dict:
    # print('key:', var)
    t.append(np.max(data_style_att_dict[var]))
    if np.max(data_style_att_dict[var]) > 0.3:
        print(data_style_att_dict[var])
    cnt += 1
t = np.asarray(t)
drawHist(t)
print('cnt is:', cnt)
'''



# import tensorflow as tf
# from tensorflow.python.ops import array_ops
# from TFCommon.RNNCell import GRUCell
# from hjk_tools.Layers import cbhg
# import math
# from tensorflow.contrib.rnn import MultiRNNCell, ResidualWrapper
#
#
# input = tf.get_variable(name='input', shape=(2, 3), dtype=tf.float32, initializer=tf.constant_initializer([[1, 2, 3], [3, 4, 400]]))
# mean, variance = tf.nn.moments(input, axes=[0], keep_dims=True)
# default_epsilon=0.001
# input = tf.nn.batch_normalization(input, mean, variance, None, None, default_epsilon)
# '''
# def self_rnn(input, units=128, layer_num = 2, parallel_iterations=64, name='gru', reuse=False):
#     with tf.variable_scope(name_or_scope=name):
#         with tf.variable_scope('enc'):
#             encoder_rnn = MultiRNNCell([GRUCell(units) for _ in range(layer_num)])
#         with tf.variable_scope('dec'):
#             decoder_rnn = MultiRNNCell([ResidualWrapper(GRUCell(units)) for _ in range(layer_num)])
#
#         rnn_tot = input.shape[1]
#         batch = input.shape[0]
#
#         cond = lambda x, *_: tf.less(x, rnn_tot)
#
#         with tf.variable_scope('pre'):
#             cnt = tf.zeros((), dtype=tf.int32)
#             encoder_init_state = encoder_rnn.zero_state(batch, dtype=tf.float32)
#             decoder_init_state = decoder_rnn.zero_state(batch, dtype=tf.float32)
#             res_ta = tf.TensorArray(dtype=tf.float32, size=rnn_tot)
#             input_time_major = tf.transpose(input, (1, 0, 2))
#
#         def body(cnt, encoder_pre, decoder_pre, res_ta):
#             input = input_time_major[cnt]
#             with tf.variable_scope('enc'):
#                 output_enc, new_enc_state = encoder_rnn(input, encoder_pre)
#             with tf.variable_scope('dec'):
#                 output_dec, new_dec_state = decoder_rnn(output_enc, decoder_pre)
#             res_ta = res_ta.write(cnt, output_dec)
#             cnt = tf.add(cnt, 1)
#             return cnt, new_enc_state, new_dec_state, res_ta
#
#
#         res_cnt, encoder_res, decoder_res, final_res_ta = tf.while_loop(cond, body, loop_vars=[cnt, encoder_init_state, decoder_init_state, res_ta], parallel_iterations=parallel_iterations)
#         # final_res_ta = tf.stack(final_res_ta)
#         final_res = final_res_ta.stack()
#
#         return final_res
#
#
# final_res = self_rnn(input, units=4, layer_num=2)
#
#
# '''
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     out1, a, b = sess.run([input, mean, variance])
#     print(out1)
#     print('adfas')
#     print(a, b)
#
#
#
#
#

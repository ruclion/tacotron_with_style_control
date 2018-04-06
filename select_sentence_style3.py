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
from best_tacotron.train_model_sentence_style3 import Tacotron
from matplotlib import pyplot
import shutil
import csv
# import pandas as pd



train_data_num = 6148
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hp = HyperParams()

data_name = 'sr16_aB_sentence_style3'
save_path = os.path.join('model', data_name)
save_data_att_path = os.path.join(save_path, 'style_dict.pkl')
model_name = "TTS"
tfrecord_train_path = './data/sr16_aB_sorted_train.tfrecords'
tfrecord_dev_path = './data/sr16_aB_sorted_dev.tfrecords'
pkl_train_id_path = './data/sr16_aB_sorted_train_id.pkl'
pkl_train_path = './data/sr16_aB_sorted_train.pkl'
pkl_dev_path = './data/sr16_aB_sorted_dev.pkl'
tb_logs_path = os.path.join('logs', data_name) + '/'
dev_txt_path = os.path.join('logs', data_name) + '/dev_loss.txt'



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
    return t['txt'], t['txt_len'], t['log_mel'], t['log_stftm'], t['key'], t['char_txt']
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
    # with tf.variable_scope('data'):
    #     inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
    #     inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
    #     inp_id = tf.placeholder(name='inp_id', shape=(None,), dtype=tf.int32)
    #     seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
    #     post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)
    #
    # train_meta_path = pkl_train_path
    # assert os.path.exists(train_meta_path),\
    #     '[!] Train meta not exists! PATH: {}'.format(train_meta_path)
    #
    # dev_meta_path = pkl_dev_path
    # assert os.path.exists(dev_meta_path), \
    #     '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)
    #
    # with open(train_meta_path, 'rb') as f:
    #     train_meta = pkl.load(f)
    #     train_meta['reduction_rate'] = hp.reduction_rate
    #
    # with open(dev_meta_path, 'rb') as f:
    #     dev_meta = pkl.load(f)
    #     dev_meta['reduction_rate'] = hp.reduction_rate
    #
    # train_model = Tacotron(inp=inp, inp_mask=inp_mask, inp_id = inp_id, seq2seq_gtruth=seq2seq_gtruth, post_gtruth=post_gtruth,
    #                        hyper_params=hp, training=True, reuse=False)
    #
    # with tf.variable_scope('optimizer'):
    #     opt = tf.train.AdamOptimizer(train_model.exp_learning_rate_decay(0.001))
    #     # grad, var = zip(*opt.compute_gradients(train_model.loss))
    #     # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #     #     train_upd = opt.apply_gradients(zip(grad, var), global_step=train_model.global_step)
    #
    #     grads_and_vars = opt.compute_gradients(train_model.loss)
    #     # for i, (grad, var) in enumerate(grads_and_vars):
    #     #     # print(var.name)
    #     #     if var.name.find('style_token:0') != -1:
    #     #         grads_and_vars[i] = (grad / 200.0, var)
    #     #         print(var.name)
    #     #         print('hhhh time')
    #     #         break
    #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #         train_upd = opt.apply_gradients(grads_and_vars, global_step=train_model.global_step)
    #
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if not os.path.exists(tb_logs_path):
    #     os.makedirs(tb_logs_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # train_model.sess = sess
        # writer = tf.summary.FileWriter(tb_logs_path, filename_suffix='train', graph=sess.graph)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # ckpt = tf.train.get_checkpoint_state(save_path)
        # saver = tf.train.Saver(max_to_keep=20)
        # train_model.saver = saver
        # ass_style_token = tf.placeholder(name="ass_style_token", shape=(1, hp.styles_kind, hp.style_dim), dtype=tf.float32)
        # ass_opt = train_model.single_style_token.assign(ass_style_token)
        # # ass_inp_att = tf.placeholder(name="ass_inp_att", shape=(None, hp.styles_kind),
        # #                                  dtype=tf.float32)
        # # att_ass_opt = train_model.inp_att.assign(ass_inp_att)
        select_key = ['LittleRedRidingHood/37_sr16k.wav',
                        'AMidsummerNightsDream/204_sr16k.wav',
                        'TheReluctantDragon/79_sr16k.wav',
                        'SorcerersApprentice/165_sr16k.wav',
                        'TheMousesWedding/66_1_sr16k.wav',
                      'AMidsummerNightsDream/102_0_sr16k.wav',
                      'PhantomOfTheOpera/171_sr16k.wav',
                      'PhantomOfTheOpera/174_sr16k.wav',
                      'RomeoAndJuliet/186_2_sr16k.wav'
                      # 'AMidsummerNightsDream/77_sr16k.wav'
                      ]

        train_next_item = init_next_batch(tfrecord_train_path, 1, 1)
        with open(pkl_train_path, 'rb') as f:
            train_meta = pkl.load(f)
            train_meta['reduction_rate'] = hp.reduction_rate
        with open(pkl_train_id_path, "rb") as f:
            data_id_dict = pkl.load(f)


        select_lst = dict()
        select_lst['batch_key'] = []
        select_lst['batch_inp'] = []
        select_lst['batch_inp_mask'] = []
        select_lst['batch_mel_gtruth'] = []
        select_lst['batch_spec_gtruth'] = []
        select_lst['batch_char_txt'] = []
        select_lst['batch_id'] = []
        while True:
            try:
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth, batch_key, batch_char_txt = get_next_batch(sess,
                                                                                                       train_next_item)
                batch_inp, batch_inp_mask, batch_mel_gtruth, batch_spec_gtruth = post_next_batch(batch_inp, batch_inp_mask,
                                                                                             batch_mel_gtruth,
                                                                                             batch_spec_gtruth, train_meta)
                for i, var in enumerate(batch_key):
                    var = var.decode()
                    for j in select_key:
                        # print(var)
                        if var.find(j) != -1:
                            print(var)
                            select_lst['batch_key'].append(var)
                            select_lst['batch_inp'].append(batch_inp[i])

                            select_lst['batch_inp_mask'].append(batch_inp_mask[i])
                            select_lst['batch_mel_gtruth'].append(batch_mel_gtruth[i])
                            select_lst['batch_spec_gtruth'].append(batch_spec_gtruth[i])
                            select_lst['batch_char_txt'].append(batch_char_txt[i].decode())

                            # print(batch_char_txt[i].decode())
                            # print(batch_char_txt[i])
                            # print('in:', select_lst['batch_char_txt'][0])
                            select_lst['batch_id'].append(data_id_dict[var])
            except Exception as e:
                print('stopped')
                break

        with open('selected_data_10.pkl', 'wb') as f:
            pkl.dump(select_lst, f)
        for i in range(len(select_key)):
            print(select_lst['batch_key'][i], select_lst['batch_id'][i], select_lst['batch_char_txt'][i])
        # print(select_lst['batch_key'], select_lst['batch_id'], select_lst['batch_char_txt'])

        # if ckpt:
        #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        #     saver.restore(sess, os.path.join(save_path, 'Tacotron-200000'))
        #     # print('restore path:', ckpt_name)
        #     # with open(save_data_att_path, "rb") as f:
        #     #     data_style_att_dict = pkl.load(f)
        #     # print('load att dict')
        # else:
        #     print('no restor, init all include style:')
        #     # np.random.seed(1)
        #     init_style_token = np.random.uniform(low=-1, high=1, size=(1, hp.styles_kind, hp.style_dim))
        #     print('look random:', np.max(init_style_token), np.min(init_style_token))
        #     sess.run(ass_opt, feed_dict={ass_style_token: init_style_token})
            # data_style_att_dict = dict()
        # print(data_id_dict)
        # data_id_list = np.arange(0, train_data_num, dtype=np.int32)
        # print(data_id_list)
        # data_att = sess.run(train_model.inp_att, feed_dict={inp_id: data_id_list})
        # print(data_att)
        # key_list = list(data_id_dict.keys())
        # ans = []
        # cnt = [[]for _ in range(10)]
        # for i in range(train_data_num):
        #     itm = info(i, key_list[i], data_att[i], np.argmax(data_att[i]), np.max(data_att[i]))
        #     ans.append(itm)
        #     cnt[itm.c].append(itm)
        # for i in range(10):
        #     print(i, ':', len(cnt[i]))
        #     cnt[i] = sorted(cnt[i], key=lambda s:s.c_val, reverse=True)
        #     for j in range(10):
        #         print('class:', cnt[i][j].c, 'val:', cnt[i][j].c_val, cnt[i][j].key)
        #         select_by_class(cnt[i][j].key, cnt[i][j].c)
        #
        #     # val_cnt = np.asarray(cnt[i][:].c_val)
        #     val_cnt = np.asarray(list(map(lambda s:s.c_val, cnt[i])))
        #     # print(val_cnt)
        #     # print(type(val_cnt))
        #     drawHist(val_cnt)
        #
        # #get data style vector
        # data_style_vec = sess.run(train_model.single_style_token)
        # data_style_vec = data_style_vec[0]
        # ans = np.zeros((10, 10), dtype=np.float32)
        # for i in range(10):
        #     for j in range(10):
        #         x = data_style_vec[i]
        #         y = data_style_vec[j]
        #         angle = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
        #         ans[i][j] = angle
        #         print(i, '----', j, ':', angle)
        #
        # with open("test.csv", "w") as csvfile:
        #     writer = csv.writer(csvfile)
        #
        #     # 先写入columns_name
        #     # writer.writerow({"index", "a_name", "b_name"})
        #     # 写入多行用writerows
        #     writer.writerows(ans)










if __name__ == '__main__':
    main()

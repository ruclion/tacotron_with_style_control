import tensorflow as tf
import scipy.io.wavfile as siowav
import librosa
import numpy as np
import pickle as pkl
import os
import argparse
import tqdm
import random
import copy
from scipy import interpolate

char_map = {}
cnt = 1



label_to_pos = {'neu':0, 'ang':1, 'sad':2, 'hap':3}

def get_path_lst(root, str='.txt', cur_list=[]):
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            get_path_lst(item_path, str, cur_list)
        if os.path.isfile(item_path):
            if item_path.endswith(str):
                cur_list.append(item_path)
    return cur_list



label_path_raw = 'D:/hjk/raw_data/iemocap_label'
label_path_lst = get_path_lst(label_path_raw, '')
print(label_path_lst)
label = dict()
for label_path in label_path_lst:
    with open(label_path, 'r') as f:
        txt = f.read()
        txt = txt.split('\n')

    for var in txt:
        t = var.split(' ')
        # print(t)
        if len(t) == 2:
            print(t[0], ':', t[1])
            label[t[0]] = label_to_pos[t[1]]
        # print(t)
    print(len(label))



txt_label = dict()
txt_path = 'D:/hjk/raw_data/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release'
txt_path_list = get_path_lst(txt_path)
print(txt_path_list)
for var in txt_path_list:
    with open(var, 'r') as f:
        txt = f.read()
        txt = txt.split('\n')
    for sentence in txt:
        if len(sentence) > 0 and sentence[0] == 'S':
            # print(sentence, 'hhhh')
            pos1 = sentence.find(' [')
            # print(pos1)
            key = sentence[0:pos1]
            pos2 = sentence.find(': ') + 2
            val = sentence[pos2:]
            txt_label[key] = val
print(len(txt_label))
for var in txt_label:
    print(var, txt_label[var])

for var in label:
    print(txt_label[var])


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert wav file to TFRecords file.")

    # parser.add_argument("--wav_root", "-s", type=str, default="../tacotron_with_style_control/data/audioBook", help="")
    parser.add_argument("--tfrecords_train_path", type=str, default="./iemocap_all.tfrecords", help="")
    # parser.add_argument("--tfrecords_dev_path", type=str, default="./sr16_aB_sorted_dev.tfrecords", help="")
    parser.add_argument("--pkl_train_path", type=str, default="./iemocap_all.pkl", help="where to store the norm coefs.")
    # parser.add_argument("--pkl_dev_path", type=str, default="./sr16_aB_sorted_dev.pkl", help="where to store the norm coefs.")
    parser.add_argument("--sr", type=int, default=16000, help="")
    parser.add_argument("--frame_shift", type=float, default=0.0125, help="")
    parser.add_argument("--frame_size", type=float, default=0.050, help="")
    parser.add_argument("--n_fft", type=int, default=1024, help="")
    parser.add_argument("--n_mels", type=int, default=80, help="")
    parser.add_argument("--window", type=str, default="hann", help="")
    parser.add_argument("--floor_gate", type=float, default=0.01, help="")
    parser.add_argument("--mel_fmin", type=float, default=125., help="")
    parser.add_argument("--mel_fmax", type=float, default=7600., help="")
    parser.add_argument("--random_num", type=int, default=30, help="")
    parser.add_argument("--train_percent", type=float, default=0.9, help="")

    return parser.parse_args()


def get_stftm(wav, sr, frame_shift, frame_size, n_fft, window):
    tmp = np.abs(librosa.core.stft(y=wav, n_fft=n_fft, hop_length=int(frame_shift * sr),
                                   win_length=int(frame_size * sr), window=window))
    return tmp.T


def get_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    tmp = librosa.filters.mel(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return tmp.T


def get_mel(stftm, mel_filterbank):
    return np.matmul(stftm, mel_filterbank)


def log_compress(spec, floor_gate):
    return np.log(np.maximum(spec, floor_gate))


def get_norm(path_list, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    log_stftm_list, log_mel_list = [], []
    cnt = 0
    for wav_path in path_list:
        cnt += 1
        print('norm:', cnt)
        this_sr, this_wav = siowav.read(wav_path)
        assert this_sr == sr, "[E] {}'sr is {}, NOT {}".format(wav_path, this_sr, sr)
        stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
        mel = get_mel(stftm, mel_filterbank)
        log_stftm_list.append(log_compress(stftm, floor_gate))
        log_mel_list.append(log_compress(mel, floor_gate))
        # print(mel.shape, stftm.shape)

    log_stftm_arr, log_mel_arr = np.concatenate(log_stftm_list, 0), np.concatenate(log_mel_list, 0)
    log_stftm_mean, log_stftm_std = np.mean(log_stftm_arr, 0), np.std(log_stftm_arr, 0)
    log_mel_mean, log_mel_std = np.mean(log_mel_arr, 0), np.std(log_mel_arr, 0)

    return {"log_stftm_mean": log_stftm_mean, "log_stftm_std": log_stftm_std,
            "log_mel_mean": log_mel_mean, "log_mel_std": log_mel_std}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_char_no(ch):
    global char_map, cnt
    if ch in char_map:
        return char_map[ch]
    else:
        char_map[ch] = cnt
        cnt += 1
        return char_map[ch]


# def change_sample_rate(old_samplerate, old_audio, NEW_SAMPLERATE):
#     if old_samplerate != NEW_SAMPLERATE:
#         duration = old_audio.shape[0] // old_samplerate
#
#         time_old = np.linspace(0, duration, old_audio.shape[0])
#         time_new = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))
#
#         interpolator = interpolate.interp1d(time_old, old_audio.T)
#         new_audio = interpolator(time_new).T
#         return NEW_SAMPLERATE, new_audio


def extract_and_sort(path, stats, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    this_sr, this_wav = siowav.read(path)
    assert this_sr == sr, "[E] {}'sr is {}, NOT {}".format(path, this_sr, sr)
    wav_name = path.split('/')[-1][:-4]

    txt = txt_label[wav_name]
    # print('txt:', txt)
    char_txt = np.copy(np.asarray(txt))
    txt = np.int32(np.asarray(list(map(get_char_no, txt))))
    txt_len = txt.shape[0]

    style_label = label[wav_name]
    if wav_name[-4] == 'F':
        style_label += 4
    # print('style:', style_label)
    if len(txt) == 0:
        print('happen')
        return False, False
    key = path.encode("utf-8")
    stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
    mel = get_mel(stftm, mel_filterbank)
    log_stftm, log_mel = log_compress(stftm, floor_gate), log_compress(mel, floor_gate)



    frames = log_mel.shape[0]
    # print(frames, '---', txt_len)
    if frames >= 2000:
        print('happen frame', frames)
        return False, False

    return {"sr": sr,
        "key": key,
        "frames": frames,
        "this_wav": this_wav,
        "char_txt": char_txt,
        "txt": txt,
        "txt_len": txt_len,
        "style_label": style_label,
        "log_stftm": log_stftm,
        "log_mel": log_mel}, True

def dict_to_example(info):
    sr = info.get('sr')
    key = info.get('key')
    frames = info.get('frames')
    this_wav = info.get('this_wav')
    char_txt = info.get('char_txt')
    txt = info.get('txt')
    txt_len = info.get('txt_len')
    style_label = info.get('style_label')
    log_stftm = info. get('log_stftm')
    log_mel = info.get('log_mel')

    txt_raw = txt.tostring()
    char_txt = char_txt.tostring()

    wav_raw = this_wav.tostring()
    log_stftm_raw = log_stftm.astype(np.float32).tostring()
    log_mel_raw = log_mel.astype(np.float32).tostring()

    # create tf example feature
    example = tf.train.Example(features=tf.train.Features(feature={
        "sr": _int64_feature(int(sr)),
        "key": _bytes_feature(key),
        "char_txt": _bytes_feature(char_txt),
        "frames": _int64_feature(int(frames)),
        "wav_raw": _bytes_feature(wav_raw),
        "txt_raw": _bytes_feature(txt_raw),
        "txt_len": _int64_feature(int(txt_len)),
        "style_label": _int64_feature(int(style_label)),
        "log_stftm_raw": _bytes_feature(log_stftm_raw),
        "log_mel_raw": _bytes_feature(log_mel_raw)}))
    return example.SerializeToString()



def read_to_bytes(path, stats, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    global cnt_len_0, cnt_len_450
    this_sr, this_wav = siowav.read(path)
    assert this_sr == sr, "[E] {}'sr is {}, NOT {}".format(path, this_sr, sr)

    txt_path = path.replace('_sr16k.wav', '.lab').replace('_wav_24k', '_lab')
    with open(txt_path, 'r') as f:
        txt = f.read()
    txt = np.int32(np.asarray(list(map(get_char_no, txt))))
    txt_len = txt.shape[0]


    if len(txt) == 0:
        cnt_len_0 += 1
        print('happen')
        return False, False

    # print('check:', txt_len)

    txt_raw = txt.tostring()

    ## txt_len shi 00000

    stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
    mel = get_mel(stftm, mel_filterbank)
    log_stftm, log_mel = log_compress(stftm, floor_gate), log_compress(mel, floor_gate)
    # norm_stftm = (log_stftm - stats["log_stftm_mean"]) / stats["log_stftm_std"]
    # norm_mel = (log_mel - stats["log_mel_mean"]) / stats["log_mel_std"]

    key = path.encode("utf-8")
    wav_raw = this_wav.tostring()
    frames = log_mel.shape[0]
    if frames > 450:
        cnt_len_450 += 1
        return False, False
    log_stftm_raw = log_stftm.astype(np.float32).tostring()
    log_mel_raw = log_mel.astype(np.float32).tostring()
    # create tf example feature
    example = tf.train.Example(features=tf.train.Features(feature={
        "sr": _int64_feature(int(sr)),
        "key": _bytes_feature(key),
        "frames": _int64_feature(int(frames)),
        "wav_raw": _bytes_feature(wav_raw),
        "txt_raw": _bytes_feature(txt_raw),
        "txt_len": _int64_feature(int(txt_len)),
        "log_stftm_raw": _bytes_feature(log_stftm_raw),
        "log_mel_raw": _bytes_feature(log_mel_raw)}))
    return example.SerializeToString(), True





def main():
    print(__file__)
    args = get_arguments()

    # 1st. get path_lst, which contains all the paths to wav files.
    path_lst = []
    for var in label:
        path_lst_var = 'D:/hjk/raw_data/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session%d/sentences/wav/' \
                       % (int(var[4]))\
                       + var[0:-5] + '/' + var + '.wav'
        path_lst.append(path_lst_var)


    # 2st. get mel-filterbank
    mel_filterbank = get_mel_filterbank(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels,
                                        fmin=args.mel_fmin, fmax=args.mel_fmax)



    # 4st. get train norm for all data
    norm_dict = get_norm(path_list=path_lst,  sr=args.sr,
                      frame_shift=args.frame_shift, frame_size=args.frame_size,
                      n_fft=args.n_fft, window=args.window,
                      mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)


    # # 3nd. get stats, which is used to normalize log compressed stftm and mel specs.
    # stats = get_stats(wav_path_list=path_lst, random_num=args.random_num, sr=args.sr,
    #                   frame_shift=args.frame_shift, frame_size=args.frame_size,
    #                   n_fft=args.n_fft, window=args.window,
    #                   mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)

    # # 4th. store normalization coefs.
    # with open(args.stats_path, "wb") as f:
    #     pkl.dump(stats, f)

    # 5th. extract features, sort it in list
    print('start extract features...')
    info_lst = []
    # for train
    for path in tqdm.tqdm(path_lst):
        try:
            info, tag = extract_and_sort(path=path, stats=norm_dict, sr=args.sr,
                                                   frame_shift=args.frame_shift, frame_size=args.frame_size,
                                                   n_fft=args.n_fft, window=args.window,
                                                   mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)
            if tag == True:
                info_lst.append(info)
        except ValueError as e:
            print(e, path)

    info_lst.sort(key=lambda k: (k.get('frames', 0)))
    # 5th. extract features, write them back to disk.
    print('start to write...')
    #for train
    with tf.python_io.TFRecordWriter(args.tfrecords_train_path) as writer:
        for itm in tqdm.tqdm(info_lst):
            try:
                example_str = dict_to_example(itm)
                writer.write(example_str)
            except ValueError as e:
                print(e)
                print('write wrong')


    #store the pkl
    train_stats = {"log_stftm_mean": norm_dict['log_stftm_mean'], "log_stftm_std": norm_dict['log_stftm_std'],
            "log_mel_mean": norm_dict['log_mel_mean'], "log_mel_std": norm_dict['log_mel_std'],
            "char_map":char_map, "key_lst": path_lst}

    with open(args.pkl_train_path, "wb") as f:
        pkl.dump(train_stats, f)



    print("Congratulations!")


if __name__ == "__main__":
    main()

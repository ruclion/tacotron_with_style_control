import tensorflow as tf
import scipy.io.wavfile as siowav
import librosa
import numpy as np
import pickle as pkl
import os
import argparse
import tqdm
import random


cnt = 0
tot_time = []

def get_arguments():
    parser = argparse.ArgumentParser(description="Convert wav file to TFRecords file.")

    parser.add_argument("--wav_root", "-s", type=str, default="../raw_data/THCoSS.TTS.0512/TH-CoSS_8/data/03FR00", help="")
    parser.add_argument("--target_path", "-d", type=str, default="./th-coss_female_wav_mel_stftm.tfrecords", help="")
    parser.add_argument("--stats_path", type=str, default="./th-coss_female_stats.pkl", help="where to store the norm coefs.")
    parser.add_argument("--sr", type=int, default=16000, help="")
    parser.add_argument("--frame_shift", type=float, default=0.0125, help="")
    parser.add_argument("--frame_size", type=float, default=0.050, help="")
    parser.add_argument("--n_fft", type=int, default=1024, help="")
    parser.add_argument("--n_mels", type=int, default=80, help="")
    parser.add_argument("--window", type=str, default="hann", help="")
    parser.add_argument("--floor_gate", type=float, default=0.01, help="")
    parser.add_argument("--mel_fmin", type=float, default=125., help="")
    parser.add_argument("--mel_fmax", type=float, default=7600., help="")
    parser.add_argument("--random_num", type=int, default=2400, help="")

    return parser.parse_args()

with open('./data/train_meta.pkl', "rb") as f:
    tmp_stats = pkl.load(f)
    dic = tmp_stats['char2id_dic']

def get_txt_lab(wav_path):
    xml_path = wav_path[:-4] + '.lab'
    dom =  open(xml_path).read()
    ans = []
    l = len(dom)
    for i in range(l):
        if dom[i] == '"' and dom[i - 7 : i] == 'pinyin=':
            j = i + 1
            while dom[j] != '"':
                ans.append(dic[dom[j]])
                j += 1

            ans.append(dic[' '])
    ans.pop()
    return ans



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


def get_stats(wav_path_list, random_num, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    random.shuffle(wav_path_list)
    random_sel_wav_list = wav_path_list[:random_num]

    log_stftm_list, log_mel_list = [], []
    for wav_path in random_sel_wav_list:
        try:
            print(wav_path)
            this_sr, this_wav = siowav.read(wav_path)
            assert this_sr == sr, "[E] {}'sr is {}, NOT {}".format(wav_path, this_sr, sr)
        except:
            print('just jump')
            continue


        stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
        mel = get_mel(stftm, mel_filterbank)
        log_stftm_list.append(log_compress(stftm, floor_gate))
        log_mel_list.append(log_compress(mel, floor_gate))

    log_stftm_arr, log_mel_arr = np.concatenate(log_stftm_list, 0), np.concatenate(log_mel_list, 0)
    log_stftm_mean, log_stftm_std = np.mean(log_stftm_arr, 0), np.std(log_stftm_arr, 0)
    log_mel_mean, log_mel_std = np.mean(log_mel_arr, 0), np.std(log_mel_arr, 0)


    return {"log_stftm_mean": log_stftm_mean, "log_stftm_std": log_stftm_std,
            "log_mel_mean": log_mel_mean, "log_mel_std": log_mel_std,
            "char_map":dic}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def read_to_bytes(path, stats, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    try:
        this_sr, this_wav = siowav.read(path)
        assert this_sr == sr, "[E] {}'sr is {}, NOT {}".format(path, this_sr, sr)
    except:
        print('just jump return')
        return False, False

    txt = np.asarray(get_txt_lab(path))

    txt_len = txt.shape[0]
    print(path)
    print(txt)
    if txt_len == 0:
        return False, False

    # print('check:', txt_len)
    txt_raw = txt.tostring()


    ## txt_len shi 00000

    stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
    mel = get_mel(stftm, mel_filterbank)
    log_stftm, log_mel = log_compress(stftm, floor_gate), log_compress(mel, floor_gate)
    norm_stftm = (log_stftm - stats["log_stftm_mean"]) / stats["log_stftm_std"]
    norm_mel = (log_mel - stats["log_mel_mean"]) / stats["log_mel_std"]

    key = path.encode("utf-8")
    wav_raw = this_wav.tostring()
    frames = norm_mel.shape[0]
    norm_stftm_raw = norm_stftm.astype(np.float32).tostring()
    norm_mel_raw = norm_mel.astype(np.float32).tostring()
    global cnt, tot_time
    cnt += 1
    tot_time.append(stftm.shape[0])
    # create tf example feature
    example = tf.train.Example(features=tf.train.Features(feature={
        "sr": _int64_feature(int(sr)),
        "key": _bytes_feature(key),
        "frames": _int64_feature(int(frames)),
        "wav_raw": _bytes_feature(wav_raw),
        "txt_raw": _bytes_feature(txt_raw),
        "txt_len": _int64_feature(int(txt_len)),
        "norm_stftm_raw": _bytes_feature(norm_stftm_raw),
        "norm_mel_raw": _bytes_feature(norm_mel_raw)}))
    return example.SerializeToString(), True


def get_path_lst(root, cur_list=[]):
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            get_path_lst(item_path, cur_list)
        if os.path.isfile(item_path):
            if item_path.endswith("wav"):
                cur_list.append(item_path)
    return cur_list


def main():
    print(__file__)
    args = get_arguments()

    # 1st. get path_lst, which contains all the paths to wav files.
    path_lst = get_path_lst(args.wav_root)
    assert path_lst, "[E] Path list is empty!"

    # 2st. get mel-filterbank
    mel_filterbank = get_mel_filterbank(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels,
                                        fmin=args.mel_fmin, fmax=args.mel_fmax)

    # 3nd. get stats, which is used to normalize log compressed stftm and mel specs.
    stats = get_stats(wav_path_list=path_lst, random_num=args.random_num, sr=args.sr,
                      frame_shift=args.frame_shift, frame_size=args.frame_size,
                      n_fft=args.n_fft, window=args.window,
                      mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)

    # 4th. store normalization coefs.
    with open(args.stats_path, "wb") as f:
        pkl.dump(stats, f)

    # 5th. extract features, normalize them and write them back to disk.
    with tf.python_io.TFRecordWriter(args.target_path) as writer:
        for path in tqdm.tqdm(path_lst):

            try:
                example_str, value_tag = read_to_bytes(path=path, stats=stats, sr=args.sr,
                                            frame_shift=args.frame_shift, frame_size=args.frame_size,
                                            n_fft=args.n_fft, window=args.window,
                                            mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)
                if value_tag == True:
                    # print('have ***')
                    writer.write(example_str)
            except ValueError as e:
                print(e, path)

    print("Congratulations!")


if __name__ == "__main__":
    main()
    with open('tot_cnt.pkl', "wb") as f:
        pkl.dump({'cnt':cnt, 'tot_time':tot_time}, f)
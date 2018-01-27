import tensorflow as tf
import os
import argparse
import scipy.io.wavfile as siowav
import numpy as np
import tqdm
import pickle as pkl


def get_arguments():
    parser = argparse.ArgumentParser(description="Extract wav from TFRecords file and save.")
    parser.add_argument("--tfrecord_path", "-s", type=str, default="./aB_wav_mel_stftm.tfrecords", help="")
    return parser.parse_args()


def parse_single_example(example_proto):
    features = {"sr": tf.FixedLenFeature([], tf.int64),
                "key": tf.FixedLenFeature([], tf.string),
                "frames": tf.FixedLenFeature([], tf.int64),
                "wav_raw": tf.FixedLenFeature([], tf.string),
                "txt_raw": tf.FixedLenFeature([], tf.string),
                "txt_len": tf.FixedLenFeature([], tf.int64),
                "norm_mel_raw": tf.FixedLenFeature([], tf.string),
                "norm_stftm_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    sr = tf.cast(parsed["sr"], tf.int32)
    key = parsed["key"]
    frames = tf.cast(parsed["frames"], tf.int32)
    wav = tf.decode_raw(parsed["wav_raw"], tf.int32)
    txt = tf.decode_raw(parsed["txt_raw"], tf.int32)
    txt_len = tf.cast(parsed["txt_len"], tf.int32)
    norm_mel = tf.reshape(tf.decode_raw(parsed["norm_mel_raw"], tf.float32), (frames, 80))
    norm_stftm = tf.reshape(tf.decode_raw(parsed["norm_stftm_raw"], tf.float32), (frames, 1025))
    return {"sr": sr, "key": key, "frames": frames, "wav": wav, "txt":txt, "txt_len":txt_len, "norm_mel": norm_mel, "norm_stftm": norm_stftm}


def get_dataset(tfrecord_path):
    with open('./aB_stats.pkl', "rb") as f:
        stats = pkl.load(f)
    pad_mel = np.ones((1,80), dtype=np.float32) * -3
    norm_pad_mel = (pad_mel - stats["log_mel_mean"]) / stats["log_mel_std"]
    norm_pad_mel_min = np.asscalar(np.float32(np.min(norm_pad_mel)))
    print(norm_pad_mel_min)

    pad_stftm = np.ones((1, 1025), dtype=np.float32) * -3
    norm_pad_stftm = (pad_stftm - stats["log_stftm_mean"]) / stats["log_stftm_std"]
    norm_pad_stftm_min = np.asscalar(np.float32(np.min(norm_pad_stftm)))
    print(norm_pad_stftm_min)

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(320)
    dataset = dataset.padded_batch(32, padded_shapes={"sr": (),
                                                     "key": (),
                                                     "frames": (),
                                                     "wav": [None],
                                                     "txt": [None],
                                                     "txt_len": (),
                                                     "norm_mel": [None, 80],
                                                     "norm_stftm": [None, 1025]}, padding_values={"sr": 0,
                                                     "key": "",
                                                     "frames": 0,
                                                     "wav": 0,
                                                     "txt": 0,
                                                     "txt_len": 0,
                                                     "norm_mel": norm_pad_mel_min,
                                                     "norm_stftm": norm_pad_stftm_min})
    return dataset


def main():
    args = get_arguments()

    data_set = get_dataset(args.tfrecord_path)
    iterator = data_set.make_one_shot_iterator()
    next_item = iterator.get_next()

    sess = tf.Session()
    while True:
        try:
            t = sess.run(next_item)
            print(t['txt'])
            print(t['norm_mel'])
            print(t['norm_stftm'])

            # print(sess.run(tf.shape(next_item["norm_mel"])))
            # print(sess.run(next_item["norm_mel"])[0])
        except Exception as e:
            print(e)

    print("Congratulations!")


if __name__ == "__main__":
    print(__file__)
    main()
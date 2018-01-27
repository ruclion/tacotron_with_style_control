import  numpy as np
import scipy.io.wavfile as siowav
import pickle as pkl
import librosa
import matplotlib.pyplot as plt
import audio
import  xml.dom.minidom
import codecs
import tensorflow as tf

OUTPUT_MEL_DIM = 80	# 80
OUTPUT_SPEC_DIM = 513 # 513
tfrecord_path = './data/th-coss_female_wav_mel_stftm.tfrecords'
pkl_path = './data/th-coss_female_stats.pkl'
BATCH_SIZE = 1
sr = 16000


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
    # print('???????????????????')
    # dataset = dataset.shuffle(10000)
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


with tf.Session() as sess:
    item = init_next_batch()
    _, _, _, t = get_next_batch(sess, item)
    # print(t)
    with open(pkl_path, "rb") as f:
        stats = pkl.load(f)
    t = t * stats["log_stftm_std"] + stats["log_stftm_mean"]
    print(t)
    pred_audio, exp_spec = audio.invert_spectrogram(t, 1.2)
    siowav.write('generate.wav', sr, pred_audio)

'''

frame_shift = 0.0125
frame_size = 0.05
n_fft = 1024
sr = 16000

def get_stftm(wav, sr, frame_shift, frame_size, n_fft, window = "hann"):
    tmp = np.abs(librosa.core.stft(y=wav, n_fft=n_fft, hop_length=int(frame_shift * sr),
                                   win_length=int(frame_size * sr), window=window))
    print(int(50 * 16 + 0.1))
    print('??', int(frame_size * sr))
    return tmp.T

wav_path = 'hjktest.wav'
this_sr, this_wav = siowav.read(wav_path)
print(this_sr)
print(this_wav, np.max(this_wav), np.min(this_wav))
print(type(this_wav[0]))
print(this_wav.shape)

stftm_wav = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft)
print(stftm_wav.shape)
print(np.max(stftm_wav), np.min(stftm_wav))
for i in range(stftm_wav.shape[0]):
    for j in range(stftm_wav.shape[1]):
        if stftm_wav[i][j] < 0.01:
            print('??', stftm_wav[i][j])
ln_stftm_wav = np.log(np.maximum(stftm_wav, 0.01))
print(np.max(ln_stftm_wav), np.min(ln_stftm_wav))
pred_audio, exp_spec = audio.invert_spectrogram(ln_stftm_wav, 1.2)
print(np.max(pred_audio), np.min(pred_audio))
siowav.write('generate.wav', sr, pred_audio)




'''




'''
with open('./data/dev_meta.pkl', "rb") as f:
    stats = pkl.load(f)
print(stats.keys())
print(stats['char2id_dic'])
# print(stats['char_inputs_dic'])
'''

'''
a = np.ones((1, 20)) * 100
b = np.ones((1, 20)) * 10
c = np.concatenate([a, b], axis=0)
print(c)

for i in range(0, 10, 3):
    for j in range(1, 4):
        print(i, j)
'''
'''
import tensorflow as tf

a = tf.get_variable('style_token', (1, 10, 2), dtype=tf.float32)

print(a)
print(a.name)
if a.name == 'style_token:0':
    print('hhh')
'''

'''
t = np.arange(0, 10)
print(t)
tt = []
tt.append(t)
t = np.arange(100, 110)
tt.append(t)
print(tt)
tt = np.concatenate(tt, 0)
print(tt)
'''

# print(np.log(0.01))
# plt.matshow(ln_stftm_wav.T, cmap='hot')
# plt.show()





'''

with open('./tot_cnt.pkl', "rb") as f:
    stats = pkl.load(f)
print(stats)
'''
'''
styles_kind = 10
style_dim = 256

linear_style_vec = np.ones((1, styles_kind, style_dim), np.float32)
for i in range(styles_kind):
    for j in range(style_dim):
        linear_style_vec[0][i][j] = np.power(2, i) / 512




print(np.min(linear_style_vec), np.max(linear_style_vec))
'''

'''
import tensorflow as tf
w1 = tf.constant(1, dtype=tf.float32, shape=(32, 1))
w2 = tf.constant(2, dtype=tf.float32, shape=(32, 1))
w = tf.concat([w1, w2], axis=-1)
w = tf.reshape(w, (32, 2))



a1 = tf.constant(100, dtype=tf.float32, shape=(32, 255))
a2 = tf.constant(10, dtype=tf.float32, shape=(32, 255))
a = a1 * tf.reshape(w[:, 0], (32, 1)) + a2 * tf.reshape(w[:, 1], (32, 1))
print(w)




sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('w is:', sess.run(w[:, 0]))
print('a is:', sess.run(a))





'''

'''
char_map = {}
cnt = 1

def get_char_no(ch):
    global char_map, cnt
    if ch in char_map:
        return char_map[ch]
    else:
        char_map[ch] = cnt
        cnt += 1
        return char_map[ch]

path = 'audioBook/All_Slices_wav_24k/AMidsummerNightsDream/0.wav'
txt_path = path.replace('.wav', '.lab').replace('_wav_24k', '_lab')
print(path, txt_path)
with open(txt_path, 'r') as f:
    text = f.read()
    # text = list(text)
print(type(text), text)
text = np.asarray(list(map(get_char_no, text)))
print(text)
text = text.tostring()
print(text)
'''
'''
this_sr, this_wav = siowav.read(path)
print(this_sr)
print(np.min(this_wav), np.max(this_wav))
'''
'''
x = np.asarray([[1, 2, 3], [1, 2, 3]])
y = np.asarray([[11, 21, 31], [11, 21, 31]])
a = [x, y]
print(a)
b = np.concatenate(a, axis=0)
print(b)
print(np.mean(b, axis=0))
'''


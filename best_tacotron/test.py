import argparse
import os
import pickle as pkl
import tqdm
import time
import math
import codecs
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#import seaborn
from Speech.tensorflow.models.Tacotron.models.generate import Tacotron
from Speech.tensorflow.models.Tacotron.hyperparameter import HyperParams
from Speech.utils.analysis import GLA, save_wav, de_emphasis


def get_arguments():
    parser = argparse.ArgumentParser(description='Train the WaveNet neural vocoder!')
    parser.add_argument('--hyper_param_path', type=str, default='./hyper_param.json',
                        help='json: hyper_param')
    parser.add_argument('--test_meta_path', type=str, default='./test_meta.pkl')
    parser.add_argument('--analysis_meta_path', type=str, default='./analysis_meta.json',
                        help='json: analysis_meta')
    parser.add_argument('--max_decode_steps', type=int, default=200)
    parser.add_argument('--decode_coef', type=float, default=0.8)
    parser.add_argument('--vocoder_impl', type=str, default='tensorflow',
                        help='"numpy" or "tensorflow"')
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--log_path', type=str, default='./log_test')
    parser.add_argument('--test_gen_path', type=str, default='./test_gen')
    return parser.parse_args()


def parse_char_seq(char_seq, char2id_dic):
    ret = [char2id_dic.get(item) for item in char_seq]
    if None in ret:
        raise ValueError
    return ret


def plot_and_save(save_path, alpha_arr):
    alpha_arr = np.squeeze(alpha_arr)
    #fig, ax = plt.subplots(figsize=(12, 5))
    fig, ax = plt.subplots()
    #seaborn.heatmap(alpha_arr.T, cmap='viridis', xticklabels=10, yticklabels=10, ax=ax)
    ax.imshow(alpha_arr.T, cmap='viridis', origin='lower', aspect='auto')
    ax.set_xlabel('Decoder timesteps')
    ax.set_ylabel('Encoder states')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def creat_batch(key_lst, char_inputs_dic, char2id_dic):
    char_seq_lst = [char_inputs_dic.get(key) for key in key_lst]
    tok_lst = [parse_char_seq(char_seq, char2id_dic) for char_seq in char_seq_lst]
    tok_len_lst = np.asarray([len(item) for item in tok_lst], dtype=np.int32)
    max_len = np.max(tok_len_lst)
    pad_id = char2id_dic.get('__PAD')
    pad_tok_lst = np.asarray([item + [pad_id]*(max_len-item_len)
                              for item, item_len in zip(tok_lst, tok_len_lst)],
                             dtype=np.int32)
    return pad_tok_lst, tok_len_lst


def main():
    args = get_arguments()
    hp = HyperParams(param_json_path=args.hyper_param_path)


    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)

    test_meta_path = args.test_meta_path
    assert os.path.exists(test_meta_path),\
        '[!] Test meta not exists! PATH: {}'.format(test_meta_path)

    with open(test_meta_path, 'rb') as f:
        test_meta = pkl.load(f)
        test_meta['reduction_rate'] = hp.reduction_rate

    inp_batch, inp_mask_batch = creat_batch(test_meta['key_lst'],
                                            test_meta['char_inputs_dic'],
                                            test_meta['char2id_dic'])

    with open(args.analysis_meta_path, 'r') as f:
        handle_dic = json.load(f)
    analysis_handle = GLA(**handle_dic)

    decode_time_steps = tf.placeholder(name='decode_time_steps', shape=(), dtype=tf.int32)
    model = Tacotron(inp, inp_mask, decode_time_steps, hyper_params=hp)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.sess = sess

        ckpt = tf.train.get_checkpoint_state(args.save_path)
        saver = tf.train.Saver()
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.save_path, ckpt_name))
        else:
            print('No trained model found!')
            exit(1)

        try:
            # synthesize speech utterances
            global_step_eval = sess.run(model.global_step)
            test_gen_root = os.path.join(args.test_gen_path, '{:d}'.format(global_step_eval))
            if not os.path.exists(test_gen_root):
                os.makedirs(test_gen_root)
            print('Recovering audios ...')
            start_time = time.time()
            if args.vocoder_impl == 'numpy':
                print('Using numpy implementation')
                eval_lst = sess.run([model.post_output, model.alpha_output],
                                    feed_dict={inp: inp_batch, inp_mask: inp_mask_batch, decode_time_steps: args.max_decode_steps})
                post_output_eval, alpha_output_eval = eval_lst
                for key, item_post in tqdm.tqdm(zip(test_meta['key_lst'], post_output_eval)):
                    wav = analysis_handle.synthesis(np.exp(item_post))
                    save_wav(os.path.join(test_gen_root, '{:s}.wav'.format(key)), wav, hp.sample_rate)
            elif args.vocoder_impl == 'tensorflow':
                print('Using tensorflow implementation')
                wav_batch = analysis_handle.tf_synthesis_v2(tf.exp(model.post_output))
                alpha_output_eval = []
                for key, inp_sig, inp_mask_sig in tqdm.tqdm(zip(test_meta['key_lst'], inp_batch, inp_mask_batch)):
                    eval_lst = sess.run([wav_batch, model.alpha_output],
                                        feed_dict={inp: np.expand_dims(inp_sig, 0),
                                                   inp_mask: np.expand_dims(inp_mask_sig, 0),
                                                   decode_time_steps: np.ceil(inp_mask_sig * args.decode_coef)})
                    wav, alpha_output = eval_lst
                    alpha_output_eval.append(alpha_output[0, :, :inp_mask_sig, 0])
                    de_wav = de_emphasis(wav)
                    save_wav(os.path.join(test_gen_root, '{:s}.wav'.format(key)), de_wav, hp.sample_rate)
            rec_cost_time = time.time() - start_time
            print('Recovering cost {:.3f} seconds.'.format(rec_cost_time))
            print('Saving text and alignments ...')
            for key, item_alpha in tqdm.tqdm(zip(test_meta['key_lst'], alpha_output_eval)):
                txt = test_meta['char_inputs_dic'].get(key)
                if isinstance(txt, list):
                    txt = ''.join(txt)
                with codecs.open(os.path.join(test_gen_root, '{:s}.txt'.format(key)), 'w', 'utf-8') as f:
                    f.write(txt)
                plot_and_save(os.path.join(test_gen_root, '{:s}.png'.format(key)), item_alpha)

        except Exception as e:
            print('An error occurred.', e)
            exit(1)
        finally:
            print('Testing stopped.')
            exit(0)


if __name__ == '__main__':
    main()
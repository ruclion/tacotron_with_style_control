import argparse
import os
import pickle as pkl
import tqdm
import math
import codecs
import tensorflow as tf
from Speech.tensorflow.models.Tacotron.models.train import Tacotron
from Speech.tensorflow.models.Tacotron.utils import Feeder
from Speech.tensorflow.models.Tacotron.hyperparameter import HyperParams


def get_arguments():
    parser = argparse.ArgumentParser(description='Train the WaveNet neural vocoder!')
    parser.add_argument('--hyper_param_path', type=str, default='./hyper_param.json',
                        help='json: hyper_param')
    parser.add_argument('--train_meta_path', type=str, default='./train_meta.pkl')
    parser.add_argument('--dev_meta_path', type=str, default='./dev_meta.pkl')
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--log_path', type=str, default='./log')
    return parser.parse_args()


def main():
    args = get_arguments()
    hp = HyperParams(param_json_path=args.hyper_param_path)


    with tf.variable_scope('data'):
        inp = tf.placeholder(name='inp', shape=(None, None), dtype=tf.int32)
        inp_mask = tf.placeholder(name='inp_mask', shape=(None,), dtype=tf.int32)
        seq2seq_gtruth = tf.placeholder(name='seq2seq_gtruth', shape=(None, None, hp.seq2seq_dim), dtype=tf.float32)
        post_gtruth = tf.placeholder(name='post_gtruth', shape=(None, None, hp.post_dim), dtype=tf.float32)

    train_meta_path = args.train_meta_path
    assert os.path.exists(train_meta_path),\
        '[!] Train meta not exists! PATH: {}'.format(train_meta_path)

    dev_meta_path = args.dev_meta_path
    assert os.path.exists(dev_meta_path), \
        '[!] Dev meta not exists! PATH: {}'.format(dev_meta_path)

    with open(train_meta_path, 'rb') as f:
        train_meta = pkl.load(f)
        train_meta['reduction_rate'] = hp.reduction_rate

    with open(dev_meta_path, 'rb') as f:
        dev_meta = pkl.load(f)
        dev_meta['reduction_rate'] = hp.reduction_rate

    coord = tf.train.Coordinator()
    train_feeder = Feeder(coord, [inp, inp_mask, seq2seq_gtruth, post_gtruth],
                          train_meta, batch_size=hp.batch_size, split_nums=hp.split_nums)

    dev_feeder = Feeder(coord, [inp, inp_mask, seq2seq_gtruth, post_gtruth],
                        dev_meta, batch_size=hp.batch_size, split_nums=hp.split_nums)

    train_model = Tacotron(*train_feeder.fed_holders, hyper_params=hp, training=True, reuse=False)
    dev_model = Tacotron(*dev_feeder.fed_holders, hyper_params=hp, training=False, reuse=True)

    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(train_model.exp_learning_rate_decay(0.001))
        grad, var = zip(*opt.compute_gradients(train_model.loss))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_upd = opt.apply_gradients(zip(grad, var), global_step=train_model.global_step)

    train_batches_per_epoch = math.ceil(len(train_meta['key_lst']) / hp.batch_size)
    dev_batches_per_epoch = math.ceil(len(dev_meta['key_lst']) / hp.batch_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_model.sess = sess
        dev_model.sess = sess
        train_feeder.start_in_session(sess)
        dev_feeder.start_in_session(sess)

        ckpt = tf.train.get_checkpoint_state(args.save_path)

        saver = tf.train.Saver()
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.save_path, ckpt_name))
        else:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        writer = tf.summary.FileWriter(args.log_path, sess.graph)
        train_scalar_summary = train_model.get_scalar_summary('train')
        train_alpha_summary = train_model.get_alpha_summary('train', 1)

        global_step_eval = sess.run(train_model.global_step)
        pbar = tqdm.tqdm(total=hp.max_global_steps)
        train_loss = 0.
        train_cnt = 0
        pbar.update(global_step_eval)
        try:
            while global_step_eval < hp.max_global_steps:
                if not coord.should_stop():
                    lr_upd_op = train_model.get_learning_rate_upd_op(global_step_eval)
                    if lr_upd_op is not None:
                        sess.run(lr_upd_op)
                    _, loss_eval = sess.run([train_upd, train_model.loss])
                global_step_eval += 1
                pbar.update(1)
                if global_step_eval % 50 == 0:
                    summary_eval = sess.run(train_scalar_summary)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 1000 == 0:
                    summary_eval = sess.run(train_alpha_summary)
                    writer.add_summary(summary_eval, global_step_eval)
                if global_step_eval % 5000 == 0:
                    train_model.save(args.save_path, global_step_eval)

                # train epoch check
                train_cnt += 1
                train_loss += loss_eval
                if train_cnt % train_batches_per_epoch == 0:
                    train_cnt = 0
                    with codecs.open('train_loss.txt', 'a') as f:
                        f.write('{:.6f}\n'.format(train_loss / train_batches_per_epoch))
                    train_loss = 0.
                    # dev epoch check
                    dev_loss = 0.
                    dev_cnt = 0
                    while dev_cnt < dev_batches_per_epoch:
                        loss_eval = sess.run(dev_model.loss)
                        dev_cnt += 1
                        dev_loss += loss_eval
                    with codecs.open('dev_loss.txt', 'a') as f:
                        f.write('{:.6f}\n'.format(dev_loss / dev_batches_per_epoch))
        except Exception as e:
            print('An error occurred.', e)
        finally:
            print('Training stopped.')
            coord.request_stop()


if __name__ == '__main__':
    main()

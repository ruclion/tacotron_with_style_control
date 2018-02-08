import os
import numpy as np
import math
import random
from Speech.tensorflow.TFCommon.DataFeeder import BaseFeeder


class Feeder(BaseFeeder):
    def read_by_key(self, key):
        try:
            raw_char_seq = self.meta['char_inputs_dic'].get(key)
        except Exception as e:
            print('[E] in read_by_key: get char seq failed. key: {}'.format(key))
            exit(1)
        try:
            tok_input = self.parse_char_seq(raw_char_seq)
        except Exception as e:
            print('[E] in read_by_key: parse char seq failed. key: {}'.format(key))
            exit(1)
        tok_len = len(tok_input)
        mel_gtruth_path = os.path.join(self.meta['mel_root'], key+'.npy')
        stftm_gtruth_path = os.path.join(self.meta['stftm_root'], key+'.npy')
        try:
            mel_gtruth = np.load(mel_gtruth_path)
            stftm_gtruth = np.load(stftm_gtruth_path)
        except Exception as e:
            print('[E] in read_by_key: load acoustic feature failed. key: {}'.format(key))
            exit(1)
        return tok_input, tok_len, mel_gtruth, stftm_gtruth

    def parse_char_seq(self, char_seq):
        ret = [self.meta['char2id_dic'].get(item) for item in char_seq]
        if None in ret:
            raise ValueError
        return ret

    def pre_process_batch(self, batch):
        tok_input_batch, tok_len_batch, mel_gtruth_batch, stftm_gtruth_batch = zip(*batch)
        tok_input_max_len = max(tok_len_batch)
        tok_input_batch = np.asarray([np.pad(item, (0, tok_input_max_len - item_len),
                                             mode='constant', constant_values=self.meta['char2id_dic'].get('__PAD'))
                                      for item, item_len in zip(tok_input_batch, tok_len_batch)])
        tok_len_batch = np.asarray(tok_len_batch)
        acoustic_len_batch = [len(item) for item in mel_gtruth_batch]
        acoustic_max_len = math.ceil(max(acoustic_len_batch) / self.meta['reduction_rate']) * self.meta['reduction_rate']
        log_mel_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                  mode='constant', constant_values=self.meta['aco_pad_value'])
                                           for item, item_len in zip(mel_gtruth_batch, acoustic_len_batch)])
        log_stftm_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                    mode='constant', constant_values=self.meta['aco_pad_value'])
                                             for item, item_len in zip(stftm_gtruth_batch, acoustic_len_batch)])
        return tok_input_batch, tok_len_batch, log_mel_gtruth_batch, log_stftm_gtruth_batch

    def split_strategy(self, records_lst):
        sorted_records = sorted(records_lst, key=lambda x: len(x[-2]), reverse=True)
        sorted_batches = [sorted_records[idx * self.batch_size : (idx+1) * self.batch_size] for idx in range(self.split_nums)]
        for idx in range(self.split_nums):
            random.shuffle(sorted_batches[idx]) # shuffle, since use stateful rnn
        random.shuffle(sorted_batches)
        for idx in range(self.split_nums):
            yield sorted_batches[idx]

class FeederMulSpk(BaseFeeder):
    def read_by_key(self, key):
        try:
            raw_char_seq = self.meta['char_inputs_dic'].get(key)
            speaker = self.meta['speaker_dic'].get(key)
        except Exception as e:
            print('[E] in read_by_key: get char seq failed. key: {}'.format(key))
            exit(1)
        try:
            tok_input = self.parse_char_seq(raw_char_seq)
        except Exception as e:
            print('[E] in read_by_key: parse char seq failed. key: {}'.format(key))
            exit(1)
        tok_len = len(tok_input)
        mel_gtruth_path = os.path.join(self.meta['mel_root'], key+'.npy')
        stftm_gtruth_path = os.path.join(self.meta['stftm_root'], key+'.npy')
        try:
            mel_gtruth = np.load(mel_gtruth_path)
            stftm_gtruth = np.load(stftm_gtruth_path)
        except Exception as e:
            print('[E] in read_by_key: load acoustic feature failed. key: {}'.format(key))
            exit(1)
        return tok_input, tok_len, mel_gtruth, stftm_gtruth, speaker

    def parse_char_seq(self, char_seq):
        ret = [self.meta['char2id_dic'].get(item) for item in char_seq]
        if None in ret:
            raise ValueError
        return ret

    def pre_process_batch(self, batch):
        tok_input_batch, tok_len_batch, mel_gtruth_batch, stftm_gtruth_batch, speaker_batch = zip(*batch)
        tok_input_max_len = max(tok_len_batch)
        tok_input_batch = np.asarray([np.pad(item, (0, tok_input_max_len - item_len),
                                             mode='constant', constant_values=self.meta['char2id_dic'].get('__PAD'))
                                      for item, item_len in zip(tok_input_batch, tok_len_batch)])
        tok_len_batch = np.asarray(tok_len_batch)
        acoustic_len_batch = [len(item) for item in mel_gtruth_batch]
        acoustic_max_len = math.ceil(max(acoustic_len_batch) / self.meta['reduction_rate']) * self.meta['reduction_rate']
        log_mel_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                  mode='constant', constant_values=self.meta['aco_pad_value'])
                                           for item, item_len in zip(mel_gtruth_batch, acoustic_len_batch)])
        log_stftm_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                    mode='constant', constant_values=self.meta['aco_pad_value'])
                                             for item, item_len in zip(stftm_gtruth_batch, acoustic_len_batch)])
        speaker_batch = np.asarray(speaker_batch, dtype=np.int32)
        return tok_input_batch, tok_len_batch, log_mel_gtruth_batch, log_stftm_gtruth_batch, speaker_batch

    def split_strategy(self, records_lst):
        sorted_records = sorted(records_lst, key=lambda x: len(x[2]), reverse=True)
        sorted_batches = [sorted_records[idx * self.batch_size : (idx+1) * self.batch_size] for idx in range(self.split_nums)]
        for idx in range(self.split_nums):
            random.shuffle(sorted_batches[idx]) # shuffle, since use stateful rnn
        random.shuffle(sorted_batches)
        for idx in range(self.split_nums):
            yield sorted_batches[idx]



class FeederMulSpkNorm(BaseFeeder):
    def read_by_key(self, key):
        try:
            raw_char_seq = self.meta['char_inputs_dic'].get(key)
            speaker = self.meta['speaker_dic'].get(key)
            mel_mean = self.meta['mel_mean'][speaker]
            mel_std = self.meta['mel_std'][speaker]
            stftm_mean = self.meta['stftm_mean'][speaker]
            stftm_std = self.meta['stftm_std'][speaker]
        except Exception as e:
            print('[E] in read_by_key: get char seq failed. key: {}'.format(key))
            exit(1)
        try:
            tok_input = self.parse_char_seq(raw_char_seq)
        except Exception as e:
            print('[E] in read_by_key: parse char seq failed. key: {}'.format(key))
            exit(1)
        tok_len = len(tok_input)
        mel_gtruth_path = os.path.join(self.meta['mel_root'], key+'.npy')
        stftm_gtruth_path = os.path.join(self.meta['stftm_root'], key+'.npy')
        try:
            mel_gtruth = np.load(mel_gtruth_path)
            stftm_gtruth = np.load(stftm_gtruth_path)
        except Exception as e:
            print('[E] in read_by_key: load acoustic feature failed. key: {}'.format(key))
            exit(1)
        return tok_input, tok_len, mel_gtruth, stftm_gtruth, speaker, mel_mean, mel_std, stftm_mean, stftm_std

    def parse_char_seq(self, char_seq):
        ret = [self.meta['char2id_dic'].get(item) for item in char_seq]
        if None in ret:
            raise ValueError
        return ret

    def pre_process_batch(self, batch):
        tok_input_batch, tok_len_batch, mel_gtruth_batch, stftm_gtruth_batch, speaker_batch, \
            mel_mean_batch, mel_std_batch, stftm_mean_batch, stftm_std_batch = zip(*batch)
        tok_input_max_len = max(tok_len_batch)
        tok_input_batch = np.asarray([np.pad(item, (0, tok_input_max_len - item_len),
                                             mode='constant', constant_values=self.meta['char2id_dic'].get('__PAD'))
                                      for item, item_len in zip(tok_input_batch, tok_len_batch)])
        tok_len_batch = np.asarray(tok_len_batch)
        acoustic_len_batch = [len(item) for item in mel_gtruth_batch]
        acoustic_max_len = math.ceil(max(acoustic_len_batch) / self.meta['reduction_rate']) * self.meta['reduction_rate']
        log_mel_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                  mode='constant', constant_values=self.meta['aco_pad_value'])
                                           for item, item_len in zip(mel_gtruth_batch, acoustic_len_batch)])
        log_stftm_gtruth_batch = np.asarray([np.pad(np.log(item), ((0, acoustic_max_len - item_len), (0, 0)),
                                                    mode='constant', constant_values=self.meta['aco_pad_value'])
                                             for item, item_len in zip(stftm_gtruth_batch, acoustic_len_batch)])
        mel_mean_batch = np.expand_dims(np.asarray(mel_mean_batch, dtype=np.float32), axis=1)
        mel_std_batch = np.expand_dims(np.asarray(mel_std_batch, dtype=np.float32), axis=1)
        stftm_mean_batch = np.expand_dims(np.asarray(stftm_mean_batch, dtype=np.float32), axis=1)
        stftm_std_batch = np.expand_dims(np.asarray(stftm_std_batch, dtype=np.float32), axis=1)
        log_mel_gtruth_batch = (log_mel_gtruth_batch - mel_mean_batch) / mel_std_batch
        log_stftm_gtruth_batch = (log_stftm_gtruth_batch - stftm_mean_batch) / stftm_std_batch
        speaker_batch = np.asarray(speaker_batch, dtype=np.int32)
        return tok_input_batch, tok_len_batch, log_mel_gtruth_batch, log_stftm_gtruth_batch, speaker_batch

    def split_strategy(self, records_lst):
        sorted_records = sorted(records_lst, key=lambda x: len(x[2]), reverse=True)
        sorted_batches = [sorted_records[idx * self.batch_size : (idx+1) * self.batch_size] for idx in range(self.split_nums)]
        for idx in range(self.split_nums):
            random.shuffle(sorted_batches[idx]) # shuffle, since use stateful rnn
        random.shuffle(sorted_batches)
        for idx in range(self.split_nums):
            yield sorted_batches[idx]
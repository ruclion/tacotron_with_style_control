import json
import codecs
import os


class HyperParams:
    def __init__(self, param_dict=None, param_json_path=None):
        self.reduction_rate = 5
        self.sample_rate = 16000
        self.embed_class = 100
        self.embed_dim = 256
        self.seq2seq_dim = 80
        self.post_dim = 513
        self.learning_rate = [0.001, 0.0005, 0.0003, 0.0001]
        self.learning_rate_decay_step = [50000, 80000, 200000]
        self.clip_norm = None
        self.batch_size = 32
        self.split_nums = 32
        self.max_global_steps = 200000
        self.styles_kind = 10
        self.style_dim = 256
        if isinstance(param_dict, dict):
            self._update_from_dict(param_dict)
        elif isinstance(param_json_path, str) and os.path.exists(param_json_path):
            with codecs.open(param_json_path, 'r', encoding='utf-8') as f:
                param_dict = json.load(f)
            self._update_from_dict(param_dict)
        else:
            print('Use default setup.')

    def _update_from_dict(self, param_dict):
        for k, v in param_dict.items():
            assert hasattr(self, k),\
                '[E] param: \"{}\" is not valid.'.format(k)
            #assert isinstance(type(v), type(getattr(self, k))),\
                #'[E] param: \"{}\" should have type: \"{}\", ' \
                #'while got type: \"{}\".'.format(k, type(getattr(self, k)), type(v))
            setattr(self, k, v)

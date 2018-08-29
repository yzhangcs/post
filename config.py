# -*- coding: utf-8 -*-


class Config(object):
    ftrain = 'data/ctb5/train.conll'
    fdev = 'data/ctb5/dev.conll'
    ftest = 'data/ctb5/test.conll'
    embed = 'data/embed.txt'


class BPNNConfig(Config):
    window = 5
    embdim = 100
    hiddim = 300
    charwise = False


class LSTMConfig(Config):
    window = 1
    embdim = 100
    hiddim = 300
    charwise = False


class LSTMCHARConfig(Config):
    window = 1
    embdim = 100
    char_hiddim = 200
    hiddim = 300
    charwise = True


class NetworkConfig(Config):
    window = 1
    embdim = 100
    char_hiddim = 200
    charwise = True


config = {
    'bpnn': BPNNConfig,
    'lstm': LSTMConfig,
    'lstm_char': LSTMCHARConfig,
    'default': NetworkConfig,
}

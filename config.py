# -*- coding: utf-8 -*-


class Config(object):
    ftrain = 'data/ctb5/train.conll'
    fdev = 'data/ctb5/dev.conll'
    ftest = 'data/ctb5/test.conll'
    fembed = 'data/embed.txt'


class BPNN_CRF_Config(Config):
    n_context = 5
    n_embed = 100
    n_hidden = 300
    use_char = False


class LSTM_CRF_Config(Config):
    n_context = 1
    n_embed = 100
    n_hidden = 150
    use_char = False


class CHAR_LSTM_CRF_Config(Config):
    n_context = 1
    n_embed = 100
    n_char_embed = 100
    n_char_out = 200
    n_hidden = 150
    use_char = True


config = {
    'bpnn_crf': BPNN_CRF_Config,
    'lstm_crf': LSTM_CRF_Config,
    'char_lstm_crf': CHAR_LSTM_CRF_Config,
}

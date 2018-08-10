# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, lstm):
        self.window = 1 if lstm else 5
        self.embed_dim = 100
        self.char_embed_dim = 100
        self.hidden_dim = 300
        self.batch_size = 25
        self.epochs = 100
        self.interval = 10
        self.eta = 0.001
        self.lmbda = 0
        self.ftrain = 'data/ctb5/train.pid.conll'
        self.fdev = 'data/ctb5/dev.pid.conll'
        self.ftest = 'data/ctb5/test.pid.conll'
        self.embed = 'data/giga.100.txt'
        self.netpt = 'network.pt'

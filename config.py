# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, bpnn=True):
        self.window = 5 if bpnn else 1
        self.embdim = 100
        self.char_embdim = 200
        self.hiddim = 300
        self.batch_size = 25
        self.epochs = 100
        self.interval = 10
        self.eta = 0.001
        self.ftrain = 'data/ctb5/train.conll'
        self.fdev = 'data/ctb5/dev.conll'
        self.ftest = 'data/ctb5/test.conll'
        self.embed = 'data/embed.txt'
        self.netpt = 'network.pt'

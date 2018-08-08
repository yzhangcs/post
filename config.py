# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, lstm):
        self.embdim = 100
        self.cembdim = 100
        self.hiddim = 300
        self.batch_size = 25
        self.window = 1 if lstm else 5
        self.epochs = 100
        self.interval = 10
        self.eta = 0.001
        self.lmbda = 0
        self.ftrain = 'data/ctb7/train.conll'
        self.fdev = 'data/ctb7/dev.conll'
        self.ftest = 'data/ctb7/test.conll'
        self.embed = 'data/giga.100.txt'
        self.netpt = 'network.pt'

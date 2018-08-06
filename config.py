# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, lstm):
        self.embdim = 50
        self.cembdim = 50
        self.hiddim = 300
        self.batch_size = 25
        self.window = 1 if lstm else 5
        self.epochs = 100
        self.interval = 10
        self.eta = 0.001
        self.lmbda = 0
        self.ftrain = 'data/train.conll'
        self.fdev = 'data/dev.conll'
        self.ftest = 'data/test.conll'
        self.embed = 'data/base_embeddings.txt'
        self.netpkl = 'network.pt'

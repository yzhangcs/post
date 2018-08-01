# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self):
        self.window = 5
        self.embdim = 50
        self.hiddim = 300
        self.epochs = 100
        self.batch_size = 25
        self.interval = 10
        self.eta = 0.001
        self.lmbda = 0
        self.ftrain = 'data/train.conll'
        self.fdev = 'data/dev.conll'
        self.ftest = 'data/test.conll'
        self.embed = 'data/base_embeddings.txt'
        self.lstmpkl = 'lstm.pkl'

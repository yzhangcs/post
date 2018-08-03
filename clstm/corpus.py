# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import pad_unsorted_sequence


class Corpus(object):
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

    def __init__(self, words, tags):
        self.words = words
        self.chars = sorted(set(''.join(words)) | {self.UNK})
        self.tags = tags

        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.swi = self.wdict[self.SOS]
        self.ewi = self.wdict[self.EOS]
        self.uwi = self.wdict[self.UNK]
        self.uci = self.cdict[self.UNK]
        self.nw = len(self.words)
        self.nc = len(self.chars)
        self.nt = len(self.tags)

    def load(self, fdata):
        wx, cx, wlens, clens, y = [], [], [], [], []
        # 句子按照长度从大到小有序
        sentences = sorted(self.preprocess(fdata),
                           key=lambda wx: len(wx[0]),
                           reverse=True)
        # 获取单词最大长度
        maxlen = max(max(len(w) for w in wordseq)
                     for wordseq, tagseq in sentences)
        for wordseq, tagseq in sentences:
            wiseq = [self.wdict.get(w, self.uwi) for w in wordseq]
            tiseq = [self.tdict.get(t, -1) for t in tagseq]
            wx.append(torch.tensor([wi for wi in wiseq], dtype=torch.long))
            cx.append(torch.tensor([
                [self.cdict.get(c, self.uci)
                 for c in w] + [0] * (maxlen - len(w))
                for w in wordseq
            ]))
            wlens.append(len(tiseq))
            clens.append(torch.tensor([len(w) for w in wordseq]))
            y.append(torch.tensor([ti for ti in tiseq], dtype=torch.long))
        wx = pad_sequence(wx, batch_first=True)
        cx = pad_sequence(cx, batch_first=True)
        wlens = torch.tensor(wlens)
        clens = pad_sequence(clens, batch_first=True)
        y = pad_sequence(y, batch_first=True)
        return wx, cx, wlens, clens, y

    def size(self):
        return self.nw - 3, self.nt

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as train:
            lines = [line for line in train]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    @staticmethod
    def get_embed(fembed):
        with open(fembed, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        words, embed = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        words = list(words)
        embed = torch.FloatTensor(embed)
        return words, embed

    @staticmethod
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(np.hstack(wordseqs)))
        tags = sorted(set(np.hstack(tagseqs)))
        return words, tags

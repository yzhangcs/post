# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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

    def load(self, fdata, charwise=False, window=0):
        x, cx, lens, clens, y = [], [], [], [], []
        # 句子按照长度从大到小有序
        sentences = sorted(self.preprocess(fdata),
                           key=lambda x: len(x[0]),
                           reverse=True)
        # 获取单词最大长度
        maxlen = max(max(len(w) for w in wordseq)
                     for wordseq, tagseq in sentences)
        for wordseq, tagseq in sentences:
            wiseq = [self.wdict.get(w, self.uwi) for w in wordseq]
            tiseq = [self.tdict.get(t, -1) for t in tagseq]  # TODO
            if window > 0:
                x.append(self.get_context(wiseq, window))
            else:
                x.append(torch.tensor([wi for wi in wiseq], dtype=torch.long))
            # 不足最大长度的部分用0填充
            cx.append(torch.tensor([
                [self.cdict.get(c, self.uci)
                    for c in w] + [0] * (maxlen - len(w))
                for w in wordseq
            ]))
            lens.append(len(tiseq))
            clens.append(torch.tensor([len(w) for w in wordseq],
                                      dtype=torch.long))
            y.append(torch.tensor([ti for ti in tiseq], dtype=torch.long))

        x = pad_sequence(x, batch_first=True, padding_value=-1)
        cx = pad_sequence(cx, batch_first=True, padding_value=-1)
        lens = torch.tensor(lens)
        clens = pad_sequence(clens, batch_first=True, padding_value=-1)
        y = pad_sequence(y, batch_first=True, padding_value=-1)

        data = (x, cx, lens, clens, y) if charwise else (x, lens, y)
        return data

    def get_context(self, wiseq, window):
        half = window // 2
        length = len(wiseq)
        wiseq = [self.swi] * half + wiseq + [self.ewi] * half
        return torch.tensor([wiseq[i:i + window] for i in range(length)],
                            dtype=torch.long)

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

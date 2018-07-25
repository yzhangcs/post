# -*- coding: utf-8 -*-

import numpy as np
import torch


class Corpus(object):
    BOS = '***'
    EOS = '$$$'
    UNK = 'UNKNOWN'

    def __init__(self, fdata):
        self.sentences = self.preprocess(fdata)
        self.wordseqs, self.tagseqs = zip(*self.sentences)
        self.words = sorted(set(np.hstack(self.wordseqs)))
        self.tags = sorted(set(np.hstack(self.tagseqs)))
        self.words.extend([self.BOS, self.EOS, self.UNK])

        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.bi = self.wdict[self.BOS]
        self.ei = self.wdict[self.EOS]
        self.ui = self.wdict[self.UNK]
        self.nw = len(self.words)
        self.nt = len(self.tags)

    def preprocess(self, fdata):
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

    def load(self, fdata, window=5):
        x, y = [], []
        half = window // 2
        sentences = self.preprocess(fdata)
        for wordseq, tagseq in sentences:
            wis = [self.wdict[w] if w in self.wdict else self.ui
                   for w in wordseq]
            wis = [self.bi] * half + wis + [self.ei] * half
            tis = [self.tdict[t] if t in self.tdict else 0
                   for t in tagseq]
            for i, ti in enumerate(tis):
                x.append(wis[i:i + window])
                y.append(ti)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def get_embed(self, fembed):
        with open(fembed, 'r') as f:
            lines = [line for line in f]
        embdim = len(lines[0].split()) - 1
        embed = torch.rand(self.nw, embdim)
        for line in lines:
            split = line.split()
            if split[0] in self.wdict:
                nums = list(map(float, split[1:]))
                embed[self.wdict[split[0]]] = torch.tensor(nums)
        return embed

    def size(self):
        return self.nw - 3, self.nt

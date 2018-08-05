# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Corpus(object):
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

    def __init__(self, fdata):
        # 获取数据的句子
        self.sentences = self.preprocess(fdata)
        # 获取数据的所有不同的词汇、词性和字符
        self.words, self.tags, self.chars = self.parse(self.sentences)
        # 增加句首词汇、句尾词汇和未知词汇
        self.words += [self.SOS, self.EOS, self.UNK]
        # 增加未知字符
        self.chars += [self.UNK]

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        # 字符字典
        self.cdict = {c: i for i, c in enumerate(self.chars)}

        # 句首词汇索引
        self.swi = self.wdict[self.SOS]
        # 句尾词汇索引
        self.ewi = self.wdict[self.EOS]
        # 未知词汇索引
        self.uwi = self.wdict[self.UNK]
        # 未知字符索引
        self.uci = self.cdict[self.UNK]

        # 句子数量
        self.ns = len(self.sentences)
        # 词汇数量
        self.nw = len(self.words)
        # 词性数量
        self.nt = len(self.tags)
        # 字符数量
        self.nc = len(self.chars)

    def extend(self, fembed):
        with open(fembed, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        # 获取预训练数据中的词汇和嵌入矩阵
        words, embed = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        unk_words = [w for w in words if w not in self.wdict]
        unk_chars = [c for c in ''.join(unk_words) if c not in self.cdict]
        # 扩展词汇和字符
        self.words = sorted(self.words + unk_words)
        self.chars = sorted(self.words + unk_chars)
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.swi = self.wdict[self.SOS]
        self.ewi = self.wdict[self.EOS]
        self.uwi = self.wdict[self.UNK]
        self.uci = self.cdict[self.UNK]
        self.nw = len(self.words)
        self.nc = len(self.chars)
        # 不在预训练矩阵中的词汇的词向量值随机初始化
        embed = torch.FloatTensor(embed)
        indices = [self.wdict[w] for w in words]
        extended_embed = torch.randn(self.nw, embed.size(1))
        extended_embed[indices] = embed
        return extended_embed

    def load(self, fdata, charwise=False, window=0):
        x, lens, cx, clens, y = [], [], [], [], []
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
            lens.append(len(tiseq))
            # 不足最大长度的部分用0填充
            cx.append(torch.tensor([
                [self.cdict.get(c, self.uci)
                    for c in w] + [0] * (maxlen - len(w))
                for w in wordseq
            ]))
            clens.append(torch.tensor([len(w) for w in wordseq],
                                      dtype=torch.long))
            y.append(torch.tensor([ti for ti in tiseq], dtype=torch.long))

        x = pad_sequence(x, True)
        lens = torch.tensor(lens)
        cx = pad_sequence(cx, True)
        clens = pad_sequence(clens, True)
        y = pad_sequence(y, True, padding_value=-1)

        data = (x, lens, cx, clens, y) if charwise else (x, lens, y)
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
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(np.hstack(wordseqs)))
        tags = sorted(set(np.hstack(tagseqs)))
        chars = sorted(set(''.join(words)))
        return words, tags, chars

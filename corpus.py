# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset

from utils import init_embedding


class Corpus(object):
    PAD = '<PAD>'
    UNK = '<UNK>'
    SOS = '<SOS>'
    EOS = '<EOS>'

    def __init__(self, fdata, fembed=None):
        # 获取数据的句子
        self.sentences = self.preprocess(fdata)
        # 获取数据的所有不同的词汇、词性和字符
        self.words, self.tags, self.chars = self.parse(self.sentences)
        # 增加句首词汇、句尾词汇和未知词汇
        self.words = [self.PAD, self.UNK, self.SOS, self.EOS] + self.words
        # 增加未知字符
        self.chars = [self.PAD, self.UNK] + self.chars

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        # 字符字典
        self.cdict = {c: i for i, c in enumerate(self.chars)}

        # 填充词汇索引
        self.pwi = self.wdict[self.PAD]
        # 未知词汇索引
        self.uwi = self.wdict[self.UNK]
        # 句首词汇索引
        self.swi = self.wdict[self.SOS]
        # 句尾词汇索引
        self.ewi = self.wdict[self.EOS]
        # 填充字符索引
        self.pci = self.cdict[self.PAD]
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

        # 预训练词嵌入
        self.embed = self.get_embed(fembed) if fembed is not None else None

    def extend(self, words):
        unk_words = [w for w in words if w not in self.wdict]
        unk_chars = [c for c in ''.join(unk_words) if c not in self.cdict]
        # 扩展词汇和字符
        self.words = sorted(set(self.words + unk_words) - {self.PAD})
        self.chars = sorted(set(self.chars + unk_chars) - {self.PAD})
        self.words = [self.PAD] + self.words
        self.chars = [self.PAD] + self.chars
        # 更新字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        # 更新索引
        self.pwi = self.wdict[self.PAD]
        self.uwi = self.wdict[self.UNK]
        self.swi = self.wdict[self.SOS]
        self.ewi = self.wdict[self.EOS]
        self.pci = self.cdict[self.PAD]
        self.uci = self.cdict[self.UNK]
        # 更新词汇和字符数
        self.nw = len(self.words)
        self.nc = len(self.chars)

    def load(self, fdata, charwise=False, window=1, max_len=10):
        sentences = self.preprocess(fdata)
        x, y, char_x, lens = [], [], [], []

        for wordseq, tagseq in sentences:
            wiseq = [self.wdict.get(w, self.uwi) for w in wordseq]
            tiseq = [self.tdict[t] for t in tagseq]
            # 获取每个词汇的上下文
            if window > 1:
                x.append(self.get_context(wiseq, window))
            else:
                x.append(torch.tensor(wiseq, dtype=torch.long))
            y.append(torch.tensor(tiseq, dtype=torch.long))
            # 不足最大长度的部分用0填充
            char_x.append(torch.tensor([
                [self.cdict.get(c, self.uci)
                 for c in w[:max_len]] + [0] * (max_len - len(w))
                for w in wordseq
            ]))
            lens.append(len(tiseq))

        x = pad_sequence(x, True)
        y = pad_sequence(y, True)
        char_x = pad_sequence(char_x, True)
        lens = torch.tensor(lens)

        if charwise:
            dataset = TensorDataset(x, y, char_x, lens)
        else:
            dataset = TensorDataset(x, y, lens)

        return dataset

    def get_context(self, wiseq, window):
        half = window // 2
        length = len(wiseq)
        wiseq = [self.swi] * half + wiseq + [self.ewi] * half
        context = [wiseq[i:i + window] for i in range(length)]
        context = torch.tensor(context, dtype=torch.long)

        return context

    def get_embed(self, fembed):
        with open(fembed, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        # 获取预训练数据中的词汇和嵌入矩阵
        words, embed = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        # 扩充词汇
        self.extend(words)
        # 初始化词嵌入
        embed = torch.tensor(embed, dtype=torch.float)
        extended_embed = torch.Tensor(self.nw, embed.size(1))
        indices = [self.wdict[w] for w in words]
        init_embedding(extended_embed)
        extended_embed[indices] = embed

        return extended_embed

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"{'':2}num of sentences: {self.ns}\n"
        info += f"{'':2}num of words: {self.nw}\n"
        info += f"{'':2}num of tags: {self.nt}\n"
        info += f"{'':2}num of chars: {self.nc}\n"
        info += f")\n"

        return info

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as f:
            lines = [line for line in f]
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
        words = sorted(set(w for wordseq in wordseqs for w in wordseq))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        chars = sorted(set(''.join(words)))

        return words, tags, chars

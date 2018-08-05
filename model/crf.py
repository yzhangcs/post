# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, nt):
        super(CRF, self).__init__()

        # 不同的词性个数
        self.nt = nt
        # 句间迁移
        self.trans = nn.Parameter(torch.randn(self.nt, self.nt))
        # 句首迁移
        self.strans = nn.Parameter(torch.randn(self.nt))
        # 句尾迁移
        self.etrans = nn.Parameter(torch.randn(self.nt))

    def forward(self, emit, y):
        T = len(emit)

        alpha = self.strans + emit[0]
        for i in range(1, T):
            scores = torch.t(self.trans + emit[i])
            alpha = torch.logsumexp(scores + alpha, dim=1)
        logZ = torch.logsumexp(alpha + self.etrans, dim=0)
        return logZ - self.score(emit, y)

    def score(self, emit, y):
        T = len(emit)
        score = 0

        score += self.strans[y[0]]
        score += self.trans[y[:-1], y[1:]].sum()
        score += self.etrans[y[-1]]
        score += emit.gather(dim=1, index=y.view(-1, 1)).sum()
        return score

    def viterbi(self, emit):
        T = len(emit)
        delta = torch.zeros(T, self.nt)
        paths = torch.zeros(T, self.nt, dtype=torch.long)

        delta[0] = self.strans + emit[0]

        for i in range(1, T):
            scores = torch.t(self.trans + emit[i]) + delta[i - 1]
            delta[i], paths[i] = torch.max(scores, dim=1)
        prev = torch.argmax(delta[-1] + self.etrans)

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return torch.tensor(predict)

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
            scores = torch.transpose(self.trans + emit[i], 0, 1)
            alpha = logsumexp(scores + alpha, dim=1)
        logZ = logsumexp(alpha + self.etrans, dim=0)
        return logZ - self.score(emit, y)

    def score(self, emit, y):
        T = len(emit)

        score = torch.tensor(0, dtype=torch.float)
        score += self.strans[y[0]]
        for i, (prev, ti) in enumerate(zip(y[:-1], y[1:])):
            score += self.trans[prev, ti] + emit[i, prev]
        score += self.etrans[y[-1]] + emit[-1, y[-1]]
        return score

    def viterbi(self, emit):
        T = len(emit)
        delta = torch.zeros(T, self.nt)
        paths = torch.zeros(T, self.nt, dtype=torch.long)

        delta[0] = self.strans + emit[0]

        for i in range(1, T):
            scores = torch.transpose(self.trans + emit[i], 0, 1) + delta[i - 1]
            delta[i], paths[i] = torch.max(scores, dim=1)
        prev = torch.argmax(delta[-1])

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return torch.tensor(predict)


def logsumexp(tensor, dim):
    offset = torch.max(tensor, dim)[0]
    broadcast = offset.unsqueeze(dim)
    tmp = torch.log(torch.sum(torch.exp(tensor - broadcast), dim))
    return offset + tmp

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, nt):
        super(CRF, self).__init__()

        # 不同的词性个数
        self.nt = nt
        # 句间迁移(FROM->TO)
        self.trans = nn.Parameter(torch.randn(nt, nt) / nt ** 0.5)
        # 句首迁移
        self.strans = nn.Parameter(torch.randn(nt) / nt ** 0.5)
        # 句尾迁移
        self.etrans = nn.Parameter(torch.randn(nt) / nt ** 0.5)

    def forward(self, emit, target, mask):
        T, B, N = emit.shape

        logZ = self.get_logZ(emit, mask)
        score = self.score(emit, target, mask)

        return (logZ - score) / B

    def get_logZ(self, emit, mask):
        T, B, N = emit.shape

        alpha = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha.masked_scatter_(mask[i].unsqueeze(1), scores)

        return torch.logsumexp(alpha + self.etrans, dim=1).sum()

    def score(self, emit, target, mask):
        T, B, N = emit.shape
        scores = torch.zeros(T, B)

        # 加上句间迁移分数
        scores[1:] += self.trans[target[:-1], target[1:]]
        # 加上发射分数
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # 通过掩码过滤分数
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.strans[target[0]].sum()
        # 加上句尾迁移分数
        score += self.etrans[target.gather(dim=0, index=ends)].sum()

        return score

    def viterbi(self, emit, mask):
        T, B, N = emit.shape
        lens = mask.sum(dim=0)
        delta = torch.zeros(T, B, N)
        paths = torch.zeros(T, B, N, dtype=torch.long)

        delta[0] = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + self.etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            # 反转预测序列并保存
            predicts.append(torch.tensor(predict).flip(0))

        return torch.cat(predicts)

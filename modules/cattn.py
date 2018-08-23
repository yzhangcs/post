# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .encoder import Encoder


class CharATTN(nn.Module):

    def __init__(self, chrdim, embdim):
        super(CharATTN, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(chrdim, embdim)
        # TODO: add params
        self.encoder = Encoder(L=3, H=5, Dk=40, Dv=40, Dm=200, Dh=400)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取掩码
        mask = torch.arange(T) < lens.unsqueeze(-1)
        # 获取字嵌入向量
        x = self.embed(x)
        x = self.encoder(x, mask)
        x[1 - mask] = 0

        return x.mean(dim=1)

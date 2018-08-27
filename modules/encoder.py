# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .renc import RNMTPlusEncoder
from .tenc import TransformerEncoder


class Encoder(nn.Module):

    def __init__(self, Dm, cascade=False):
        super(Encoder, self).__init__()

        self.Dm = Dm
        self.cascade = cascade
        self.renc = RNMTPlusEncoder(L=2,
                                    Dm=Dm,
                                    p=0.2)
        self.tenc = TransformerEncoder(L=3,
                                       H=5,
                                       Dk=Dm // 5,
                                       Dv=Dm // 5,
                                       Dm=Dm,
                                       Dh=Dm * 2,
                                       p=0.2)

        self.norm = nn.LayerNorm(Dm) if cascade else nn.LayerNorm(Dm * 2)

    def forward(self, x, lens):
        B, T, N = x.shape
        # 获取掩码
        mask = torch.arange(T) < lens.unsqueeze(-1)

        if self.cascade:
            x = self.renc(x, lens)
            x = self.norm(x)
            x = self.tenc(x, mask)
        else:
            x = torch.cat((self.renc(x, lens), self.tenc(x, mask)), dim=-1)
            x = self.norm(x)

        return x

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class REncoder(nn.Module):

    def __init__(self, L, Dm, p=0.2):
        super(REncoder, self).__init__()

        self.layers = nn.ModuleList([
            Layer(Dm, p) for _ in range(L)
        ])

    def forward(self, x, lens):
        out = x
        for layer in self.layers:
            out = layer(out, lens)

        return out


class Layer(nn.Module):

    def __init__(self, Dm, p=0.2):
        super(Layer, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_size=Dm,
                            hidden_size=Dm // 2,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(p)

    def forward(self, x, lens):
        residual = x
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)

        return x + residual

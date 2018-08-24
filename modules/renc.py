# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class REncoder(nn.Module):

    def __init__(self, L, Dm):
        super(REncoder, self).__init__()

        self.layers = nn.ModuleList([
            Layer(Dm) for _ in range(L)
        ])

        self.drop = nn.Dropout()

    def forward(self, x, lens):
        for layer in self.layers:
            x = layer(x, lens)

        return x


class Layer(nn.Module):

    def __init__(self, Dm):
        super(Layer, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_size=Dm,
                            hidden_size=Dm // 2,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout()

    def forward(self, x, lens):
        residual = x
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)
        out = x + residual

        return out

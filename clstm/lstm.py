# -*- coding: utf-8 -*-

import pickle
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class WordLSTM(nn.Module):

    def __init__(self, vocdim, chrdim, embdim, hiddim, outdim, embed):
        super(WordLSTM, self).__init__()

        # 词汇维度
        self.vocdim = vocdim
        # 字向量维度
        self.embdim = embdim
        # 隐藏层维度
        self.hiddim = hiddim
        # 输出层维度
        self.outdim = outdim
        # 词嵌入
        self.embed = nn.Embedding.from_pretrained(embed, False)
        # 词嵌入LSTM层
        self.wlstm = nn.LSTM(embdim + embdim, self.hiddim, batch_first=True)
        # 字嵌入LSTM层
        self.clstm = CharLSTM(chrdim, embdim, embdim)

    def forward(self, wx, cx, wlens, clens):
        B, T = wx.shape

        # 获取词嵌入向量
        wx = self.embed(wx).view(B, T, -1)
        # 获取字嵌入向量
        cx = pad_sequence([
            self.clstm(cx[i], clens[i]) for i, wl in enumerate(wlens)
        ], batch_first=True)

        x = torch.cat((wx, cx), dim=-1)
        x = pack_padded_sequence(x, wlens, batch_first=True)

        hidden = self.init_hidden(B)
        x, hidden = self.wlstm(x, hidden)
        return x.data

    def init_hidden(self, batch_size):
        return (nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)),
                nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)))


class CharLSTM(nn.Module):

    def __init__(self, chrdim, embdim, hiddim):
        super(CharLSTM, self).__init__()

        # 字符维度
        self.chrdim = chrdim
        # 字向量维度
        self.embdim = embdim
        # 隐藏态维度
        self.hiddim = hiddim
        # 字嵌入
        self.embed = nn.Embedding(chrdim, embdim)
        # 字嵌入LSTM层
        self.lstm = nn.LSTM(embdim, hiddim, batch_first=True)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取字嵌入向量
        x = self.embed(x)
        # 获取长度由大到小排列的字序列索引
        lens, indices = lens.sort(descending=True)
        # 获取反向索引用来恢复原有的顺序
        reversed_indices = indices.sort(descending=True)[1]
        # 序列按长度由大到小排列
        x = x.view(B, T, -1)[indices]

        x = pack_padded_sequence(x, lens, batch_first=True)
        hidden = self.init_hidden(B)

        x, hidden = self.lstm(x, hidden)
        return hidden[0].squeeze()[reversed_indices]

    def init_hidden(self, batch_size):
        return (nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)),
                nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)))

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):

    def __init__(self, chrdim, embdim, hiddim, bidirectional):
        super(CharLSTM, self).__init__()

        # 字嵌入
        weight = torch.randn(chrdim, embdim) / embdim ** 0.5
        self.embed = nn.Embedding(num_embeddings=chrdim,
                                  embedding_dim=embdim,
                                  _weight=weight)
        # 字嵌入LSTM层
        hidden_size = hiddim // 2 if bidirectional else hiddim
        self.lstm = nn.LSTM(input_size=embdim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional)

    def init_hidden(self, batch_size):
        num_layers = self.lstm.num_layers
        num_directions = 2 if self.lstm.bidirectional else 1
        hidden_size = self.lstm.hidden_size
        shape = (num_layers * num_directions, batch_size, hidden_size)
        return (torch.randn(shape) / hidden_size ** 0.5,
                torch.randn(shape) / hidden_size ** 0.5)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取长度由大到小排列的字序列索引
        lens, indices = lens.sort(descending=True)
        # 获取反向索引
        _, reversed_indices = indices.sort()
        # 获取单词最大长度
        max_len = lens[0]
        # 序列按长度由大到小排列
        x = x[indices, :max_len]
        # 获取字嵌入向量
        x = self.embed(x)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)

        x, (hidden, _) = self.lstm(x, self.init_hidden(B))
        # 获取词的字符表示
        reprs = torch.cat(torch.unbind(hidden), dim=1)
        # 恢复原有的顺序
        reprs = reprs[reversed_indices]

        return reprs

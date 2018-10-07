# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):

    def __init__(self, n_char, n_embed, n_out):
        super(CharLSTM, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(num_embeddings=n_char,
                                  embedding_dim=n_embed)
        # 字嵌入LSTM层
        self.lstm_crf = nn.LSTM(input_size=n_embed,
                                hidden_size=n_out // 2,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, x):
        B, T = x.shape
        # 获取掩码
        mask = x.gt(0)
        # 获取按长度有序的字序列索引
        lens, indices = torch.sort(mask.sum(dim=1), descending=True)
        # 获取逆序索引
        _, inverse_indices = indices.sort()
        # 获取单词最大长度
        max_len = lens[0]
        # 序列按长度由大到小排列
        x = x[indices, :max_len]
        # 获取字嵌入向量
        x = self.embed(x)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)

        x, (hidden, _) = self.lstm_crf(x)
        # 获取词的字符表示
        reprs = torch.cat(torch.unbind(hidden), dim=1)
        # 恢复原有的顺序
        reprs = reprs[inverse_indices]

        return reprs

# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader

from modules.crf import CRF
from modules.attn import Attention


class LSTM(nn.Module):

    def __init__(self, window, vocdim, chrdim,
                 embdim, char_embdim, hiddim, outdim,
                 lossfn, use_attn=False, use_crf=False, bidirectional=False,
                 pretrained=None):
        super(LSTM, self).__init__()

        if pretrained is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)

        self.clstm = CharLSTM(chrdim, embdim, char_embdim,
                              bidirectional=bidirectional)
        # 词嵌入LSTM层
        if bidirectional:
            self.wlstm = nn.LSTM(input_size=embdim * window + char_embdim * 2,
                                 hidden_size=hiddim // 2,
                                 batch_first=True,
                                 bidirectional=True)
        else:
            self.wlstm = nn.LSTM(input_size=embdim * window + char_embdim,
                                 hidden_size=hiddim,
                                 batch_first=True,
                                 bidirectional=False)
        # Attention层
        self.attn = Attention(3, hiddim, hiddim, hiddim) if use_attn else None
        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.dropout = nn.Dropout(0.6)
        self.lossfn = lossfn

    def forward(self, x, lens, mask, cx, clens):
        B, T, N = x.shape
        # 获取词嵌入向量
        x = self.embed(x)
        # 拼接上下文
        x = x.view(B, T, -1)

        # 获取字嵌入向量
        cx = self.clstm(cx[mask], clens[mask])
        cx = pad_sequence(torch.split(cx, lens.tolist()), True)

        # 拼接词表示和字表示
        x = torch.cat((x, cx), dim=-1)
        x = self.dropout(x)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, hidden = self.wlstm(x)
        x, _ = pad_packed_sequence(x, True)
        if self.attn is not None:
            x = self.attn(x, x, x, mask)
        x = self.dropout(x)

        return self.out(x)

    def fit(self, trainset, devset, file,
            epochs, batch_size, interval, eta):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0
        # 设置数据加载器
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_fn)
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.parameters(), lr=eta)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            for batch in train_loader:
                self.update(batch)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, tp, total, accuracy = self.evaluate(trainset)
            print(f"{'train:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            loss, tp, total, accuracy = self.evaluate(devset)
            print(f"{'dev:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # 保存效果最好的模型
            if accuracy > max_accuracy:
                torch.save(self, file)
                max_e, max_accuracy = epoch, accuracy
            elif epoch - max_e > interval:
                break
        print(f"max accuracy of dev is {max_accuracy:.2%} at epoch {max_e}")
        print(f"mean time of each epoch is {total_time / epoch}s\n")

    def update(self, batch):
        # 设置为训练模式
        self.train()
        # 清除梯度
        self.optimizer.zero_grad()

        x, lens, cx, clens, y = batch
        # 获取掩码
        mask = y.ge(0)
        y = y[mask]

        out = self(x, lens, mask, cx, clens)
        if self.crf is None:
            out = out[mask]
            loss = self.lossfn(out, y)
        else:
            emit = out.transpose(0, 1)  # [T, B, N]
            target = pad_sequence(torch.split(y, lens.tolist()))  # [T, B]
            mask = mask.t()  # [T, B]
            loss = self.crf(emit, target, mask)
        # 计算梯度
        loss.backward()
        # 更新参数
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=50):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        loader = DataLoader(dataset, batch_size, collate_fn=self.collate_fn)
        for x, lens, cx, clens, y in loader:
            # 获取掩码
            mask = y.ge(0)
            y = y[mask]

            out = self.forward(x, lens, mask, cx, clens)
            if self.crf is None:
                out = out[mask]
                predict = torch.argmax(out, dim=1)
                loss += self.lossfn(out, y)
            else:
                emit = out.transpose(0, 1)  # [T, B, N]
                target = pad_sequence(torch.split(y, lens.tolist()))  # [T, B]
                mask = mask.t()  # [T, B]
                predict = self.crf.viterbi(emit, mask)
                loss += self.crf(emit, target, mask)
            tp += torch.sum(predict == y).item()
            total += lens.sum().item()
        loss /= len(loader)

        return loss, tp, total, tp / total

    def collate_fn(self, data):
        # 按照长度调整顺序
        data.sort(key=lambda x: x[1], reverse=True)
        x, lens, cx, clens, y = zip(*data)
        # 获取句子的最大长度
        max_len = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :max_len]
        lens = torch.tensor(lens)
        cx = torch.stack(cx)[:, :max_len]
        clens = torch.stack(clens)[:, :max_len]
        y = torch.stack(y)[:, :max_len]

        return x, lens, cx, clens, y


class CharLSTM(nn.Module):

    def __init__(self, chrdim, embdim, hiddim, bidirectional):
        super(CharLSTM, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(chrdim, embdim)
        # 字嵌入LSTM层
        self.lstm = nn.LSTM(embdim, hiddim,
                            batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取长度由大到小排列的字序列索引
        lens, indices = lens.sort(descending=True)
        # 获取反向索引用来恢复原有的顺序
        _, reversed_indices = indices.sort()
        # 获取单词最大长度
        max_len = lens[0]
        # 序列按长度由大到小排列
        x = x[indices, :max_len]
        # 获取字嵌入向量
        x = self.embed(x)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)

        x, hidden = self.lstm(x)
        # 获取词的字符表示
        reprs = torch.cat(torch.unbind(hidden[0]), dim=1)[reversed_indices]

        return reprs
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, TensorDataset

from .crf import CRF


class LSTM(nn.Module):

    def __init__(self, vocdim, chrdim,
                 embdim, cembdim, window, hiddim, outdim,
                 lossfn, use_crf=False, bidirectional=False,
                 pretrained=None):
        super(LSTM, self).__init__()

        if pretrained is None:
            self.embed = torch.randn(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)

        self.clstm = CharLSTM(chrdim, embdim, cembdim,
                              bidirectional=bidirectional)
        # 词嵌入LSTM层
        if bidirectional:
            self.wlstm = nn.LSTM(input_size=embdim * window + cembdim,
                                 hidden_size=hiddim // 2,
                                 batch_first=True,
                                 bidirectional=True)
        else:
            self.wlstm = nn.LSTM(input_size=embdim * window + cembdim,
                                 hidden_size=hiddim,
                                 batch_first=True,
                                 bidirectional=False)

        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x, lens, cx, clens):
        B, T, N = x.shape
        # 获取词嵌入向量
        x = self.embed(x)
        # 拼接上下文
        x = x.view(B, T, -1)

        # 获取字嵌入向量
        cx = torch.cat([cx[i, :l] for i, l in enumerate(lens)])
        clens = torch.cat([clens[i, :l] for i, l in enumerate(lens)])
        cx = self.clstm(cx, clens)
        cx = pad_sequence(torch.split(cx, lens.tolist()), True)

        # 拼接词表示和字表示
        x = torch.cat((x, cx), dim=-1)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, hidden = self.wlstm(x)
        x = pad_packed_sequence(x, True)[0]
        x = self.dropout(x)
        return self.out(x)

    def fit(self, train_data, dev_data, file,
            epochs, batch_size, interval,
            eta, lmbda):
        # 设置为训练模式
        self.train()

        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0
        trainset = TensorDataset(*train_data)
        devset = TensorDataset(*dev_data)
        # 设置训练过程数据加载器
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_fn)
        # 设置评价过程数据加载器
        train_eval_loader = DataLoader(dataset=trainset,
                                       batch_size=len(trainset),
                                       collate_fn=self.collate_fn)
        dev_eval_loader = DataLoader(dataset=devset,
                                     batch_size=len(devset),
                                     collate_fn=self.collate_fn)
        # 设置优化器为Adam
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)

        for epoch in range(epochs):
            start = datetime.now()

            for x, lens, cx, clens, y in train_loader:
                # 清除梯度
                optimizer.zero_grad()
                # 获取掩码
                y = y.t()
                mask = y.ge(0)  # [T, B]

                output = self(x, lens, cx, clens)
                if self.crf is None:
                    output = pack_padded_sequence(output, lens).data
                    y = pack_padded_sequence(y, lens).data
                    loss = self.lossfn(output, y)
                else:
                    loss = self.crf(output, y, mask)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch} / {epochs}:")
            loss, tp, total, accuracy = self.evaluate(train_eval_loader)
            print(f"{'train:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            loss, tp, total, accuracy = self.evaluate(dev_eval_loader)
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
        print(f"mean time of each epoch is {total_time / (epoch + 1)}s\n")

    @torch.no_grad()
    def evaluate(self, loader):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        for x, lens, cx, clens, y in loader:
            # 获取掩码
            y = y.t()
            mask = y.ge(0)  # [T, B]

            output = self.forward(x, lens, cx, clens)
            if self.crf is None:
                output = pack_padded_sequence(output, lens).data
                y = pack_padded_sequence(y, lens).data

                predict = torch.argmax(output, dim=1)
                loss += self.lossfn(output, y, reduction='sum').item()
            else:
                predict = self.crf.viterbi(output, mask)
                loss += self.crf(output, y, mask)
            tp += torch.sum(predict == y).item()
            total += lens.sum().item()
        loss /= total
        return loss, tp, total, tp / total

    def collate_fn(self, data):
        # 按照长度调整顺序
        data.sort(key=lambda x: x[1], reverse=True)
        x, lens, cx, clens, y = zip(*data)
        # 获取句子的最大长度
        maxlen = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :maxlen]
        lens = torch.tensor(lens)
        cx = torch.stack(cx)[:, :maxlen]
        clens = torch.stack(clens)[:, :maxlen]
        y = torch.stack(y)[:, :maxlen]
        return x, lens, cx, clens, y


class CharLSTM(nn.Module):

    def __init__(self, chrdim, embdim, hiddim, bidirectional):
        super(CharLSTM, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(chrdim, embdim)
        # 字嵌入LSTM层
        if bidirectional:
            self.lstm = nn.LSTM(embdim, hiddim // 2,
                                batch_first=True,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(embdim, hiddim,
                                batch_first=True,
                                bidirectional=False)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取字嵌入向量
        x = self.embed(x)
        # 获取长度由大到小排列的字序列索引
        lens, indices = lens.sort(descending=True)
        # 获取反向索引用来恢复原有的顺序
        _, reversed_indices = indices.sort()
        # 获取单词最大长度
        maxlen = lens[0]
        # 序列按长度由大到小排列
        x = x.view(B, T, -1)[indices, :maxlen]
        x = pack_padded_sequence(x, lens, True)

        x, hidden = self.lstm(x)
        representations = torch.cat(torch.unbind(hidden[0]), dim=1)
        # 返回词的字符表示
        return representations[reversed_indices]

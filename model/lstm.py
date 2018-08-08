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

    def __init__(self, vocdim, embdim, window, hiddim, outdim,
                 lossfn, use_crf=False, bidirectional=False,
                 pretrained=None):
        super(LSTM, self).__init__()

        if pretrained is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)

        # 词嵌入LSTM层
        if bidirectional:
            self.lstm = nn.LSTM(input_size=embdim * window,
                                hidden_size=hiddim // 2,
                                batch_first=True,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=embdim * window,
                                hidden_size=hiddim,
                                batch_first=True,
                                bidirectional=False)

        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x, lens):
        B, T, N = x.shape
        # 获取词嵌入向量
        x = self.embed(x)
        # 拼接上下文
        x = x.view(B, T, -1)

        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, hidden = self.lstm(x)
        x, _ = pad_packed_sequence(x)
        x = self.dropout(x)
        return self.out(x)

    def fit(self, train_data, dev_data, file,
            epochs, batch_size, interval,
            eta, lmbda):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0
        trainset = TensorDataset(*train_data)
        devset = TensorDataset(*dev_data)
        # 设置数据加载器
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_fn)
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.parameters(),
                                    lr=eta,
                                    weight_decay=lmbda)
        for epoch in range(epochs):
            start = datetime.now()
            for batch in train_loader:
                self.update(batch)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, tp, total, accuracy = self.evaluate(train_data, batch_size)
            print(f"{'train:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            loss, tp, total, accuracy = self.evaluate(dev_data, batch_size)
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

    def update(self, batch):
        # 设置为训练模式
        self.train()
        # 清除梯度
        self.optimizer.zero_grad()

        x, lens, y = batch
        # 获取掩码
        mask = y.ge(0).t()  # [T, B]
        y = torch.cat([y[i, :l] for i, l in enumerate(lens)])

        out = self(x, lens)  # [T, B, N]
        if self.crf is None:
            out = torch.cat([out[:l, i] for i, l in enumerate(lens)])
            loss = self.lossfn(out, y)
        else:
            target = pad_sequence(torch.split(y, lens.tolist()))
            loss = self.crf(out, target, mask)
        # 计算梯度
        loss.backward()
        # 更新参数
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, data, batch_size):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        dataset = TensorDataset(*data)
        loader = DataLoader(dataset, batch_size, collate_fn=self.collate_fn)
        for x, lens, y in loader:
            # 获取掩码
            mask = y.ge(0).t()  # [T, B]
            y = torch.cat([y[i, :l] for i, l in enumerate(lens)])

            out = self.forward(x, lens)  # [T, B, N]
            if self.crf is None:
                out = torch.cat([out[:l, i] for i, l in enumerate(lens)])
                predict = torch.argmax(out, dim=1)
                loss += self.lossfn(out, y)
            else:
                target = pad_sequence(torch.split(y, lens.tolist()))
                predict = self.crf.viterbi(out, mask)
                loss += self.crf(out, target, mask)
            tp += torch.sum(predict == y).item()
            total += lens.sum().item()
        loss /= len(loader)
        return loss, tp, total, tp / total

    def collate_fn(self, data):
        # 按照长度调整顺序
        data.sort(key=lambda x: x[1], reverse=True)
        x, lens, y = zip(*data)
        # 获取句子的最大长度
        max_len = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :max_len]
        lens = torch.tensor(lens)
        y = torch.stack(y)[:, :max_len]
        return x, lens, y

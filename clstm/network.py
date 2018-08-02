# -*- coding: utf-8 -*-

import pickle
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from lstm import WordLSTM


class Network(nn.Module):

    def __init__(self, vocdim, chrdim, embdim,
                 hiddim, outdim,
                 lossfn, embed):
        super(Network, self).__init__()

        # LSTM层
        self.lstm = WordLSTM(vocdim, chrdim, embdim, hiddim, outdim, embed)
        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, wx, cx, wlens, clens):
        x = self.lstm(wx, cx, wlens, clens)
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
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)
        for epoch in range(epochs):
            start = datetime.now()

            for wx, cx, wlens, clens, y in train_loader:
                optimizer.zero_grad()
                # 获取长度由大到小排列的词序列索引
                wlens, indices = wlens.sort(descending=True)
                # 调整序列顺序
                wx = wx[indices]
                cx, clens, y = cx[indices], clens[indices], y[indices]

                output = self(wx, cx, wlens, clens)
                y = pack_padded_sequence(y, wlens, True).data
                loss = self.lossfn(output, y)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch} / {epochs}:")
            loss, tp, total, accuracy = self.evaluate(train_data)
            print(f"{'train:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            loss, tp, total, accuracy = self.evaluate(dev_data)
            print(f"{'dev:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # 保存效果最好的模型
            if accuracy > max_accuracy:
                self.dump(file)
                max_e, max_accuracy = epoch, accuracy
            elif epoch - max_e > interval:
                break
        print(f"max accuracy of dev is {max_accuracy:.2%} at epoch {max_e}")
        print(f"mean time of each epoch is {total_time / (epoch + 1)}s\n")

    def evaluate(self, data):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        wx, cx, wlens, clens, y = data
        output = self(wx, cx, wlens, clens)
        y = pack_padded_sequence(y, wlens, True).data
        loss = self.lossfn(output, y)
        tp = (torch.argmax(output, dim=1) == y).sum().item()
        total = wlens.sum().item()
        return loss, tp, total, tp / total

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            network = pickle.load(f)
        return network

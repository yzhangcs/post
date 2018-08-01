# -*- coding: utf-8 -*-

import pickle
import random
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from crf import CRF


class NetCRF(nn.Module):

    def __init__(self, vocdim, embdim, window, hiddim, outdim,
                 lossfn, embed):
        super(NetCRF, self).__init__()
        # 词汇维度
        self.vocdim = vocdim
        # 词向量维度
        self.embdim = embdim
        # 上下文窗口大小
        self.window = window
        # 隐藏层维度
        self.hiddim = hiddim
        # 输出层维度
        self.outdim = outdim
        self.embed = nn.Embedding.from_pretrained(embed, False)
        self.hid = nn.Linear(window * embdim, hiddim)
        self.out = nn.Linear(hiddim, outdim)
        self.crf = CRF(self.outdim)
        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x):
        x = self.embed(x)
        x = x.view(-1, self.window * self.embdim)
        x = self.dropout(F.relu(self.hid(x)))
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

        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)
        for epoch in range(epochs):
            start = datetime.now()

            random.shuffle(train_data)
            batches = [train_data[k:k + batch_size]
                       for k in range(0, len(train_data), batch_size)]
            for batch in batches:
                optimizer.zero_grad()
                x, y, lens = zip(*batch)
                output = self(torch.cat(x))
                output = torch.split(output, lens)
                loss = sum(self.crf(emit, tiseq)
                           for emit, tiseq in zip(output, y))
                loss /= len(y)
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
        x, y, lens = zip(*data)
        output = self(torch.cat(x))
        output = torch.split(output, lens)

        for emit, tiseq in zip(output, y):
            loss += self.crf(emit, tiseq)
            tp += torch.sum(tiseq == self.crf.viterbi(emit)).item()
            total += len(emit)
        loss /= len(y)
        return loss, tp, total, tp / total

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            network = pickle.load(f)
        return network

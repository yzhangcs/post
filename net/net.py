# -*- coding: utf-8 -*-

import pickle
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Network(nn.Module):

    def __init__(self, vocdim, embdim, window, hiddim, outdim,
                 lossfn, embed):
        super(Network, self).__init__()
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
        self.lossfn = lossfn
        self.dropout = nn.Dropout()

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

        trainset = TensorDataset(*train_data)
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)
        for epoch in range(epochs):
            start = datetime.now()
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self(x)
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

        x, y = data
        total = len(x)
        output = self.forward(x)
        loss = self.lossfn(output, y)
        predict = torch.argmax(output, dim=1)
        tp = torch.sum(y == predict).item()
        return loss, tp, total, tp / total

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            network = pickle.load(f)
        return network

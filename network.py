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
        self.vocdim = vocdim
        self.embdim = embdim
        self.window = window
        self.hiddim = hiddim
        self.outdim = outdim
        self.hid = nn.Linear(window * embdim, hiddim)
        self.out = torch.nn.Linear(hiddim, outdim)
        self.embed = nn.Embedding.from_pretrained(embed, False)
        self.lossfn = lossfn

    def forward(self, x):
        x = self.embed(x)
        x = x.view(-1, self.window * self.embdim)
        x = F.relu(self.hid(x))
        return F.relu(self.out(x))

    def train(self, train_data, dev_data, file,
              epochs, batch_size, interval,
              eta, lmbda):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0

        x, y = train_data
        trainset = TensorDataset(x, y)
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)
        for epoch in range(epochs):
            start = datetime.now()
            for i, (x, y) in enumerate(train_loader):
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
        print(f"mean time of each epoch is {total_time / (epoch + 1)}s")

    def evaluate(self, data):
        x, y = data
        total = len(x)
        output = self.forward(x)
        loss = self.lossfn(output, y)
        tp = torch.sum(y == torch.argmax(output, dim=1)).item()
        return loss, tp, total, tp / total

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            network = pickle.load(f)
        return network

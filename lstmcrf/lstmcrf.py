# -*- coding: utf-8 -*-

import pickle
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from crf import CRF


class LSTMCRF(nn.Module):

    def __init__(self, vocdim, embdim, hiddim, outdim,
                 lossfn, embed):
        super(LSTMCRF, self).__init__()
        # 词汇维度
        self.vocdim = vocdim
        # 词向量维度
        self.embdim = embdim
        # 隐藏层维度
        self.hiddim = hiddim
        # 输出层维度
        self.outdim = outdim
        self.embed = nn.Embedding.from_pretrained(embed, False)
        self.lstm = nn.LSTM(self.embdim * 5, self.hiddim, batch_first=True)
        self.out = nn.Linear(hiddim, outdim)
        self.crf = CRF(self.outdim)
        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x, lens):
        B, T, N = x.shape
        x = self.embed(x)
        x = x.view(B, T, -1)
        hidden = self.init_hidden(x.size(0))
        x = pack_padded_sequence(x, lens, batch_first=True)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x[0])
        return self.out(x)

    def init_hidden(self, batch_size):
        return (nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)),
                nn.init.orthogonal_(torch.zeros(1, batch_size, self.hiddim)))

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

            for x, y, lens in train_loader:
                optimizer.zero_grad()
                sorted_lens, indices = lens.sort(descending=True)
                sorted_x, sorted_y = x[indices], y[indices]
                y = pack_padded_sequence(sorted_y, sorted_lens, True)[0]
                output = self(sorted_x, sorted_lens)
                y = torch.split(y, sorted_lens.tolist())
                output = torch.split(output, sorted_lens.tolist())
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
        x, y, lens = data
        output = self(x, lens)
        output = torch.split(output, lens.tolist())
        y = torch.split(pack_padded_sequence(y, lens, True)[0],
                        lens.tolist())
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

# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from crf import CRF


class BPNN(nn.Module):

    def __init__(self, vocdim, embdim, window, hiddim, outdim,
                 lossfn, use_crf=False, pretrained=None):
        super(BPNN, self).__init__()

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
        # 损失函数
        self.lossfn = lossfn

        if pretrained is None:
            self.embed = torch.randn(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)
        # 隐藏层
        self.hid = nn.Linear(window * embdim, hiddim)
        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

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

            for x, lens, y in train_loader:
                # 清除梯度
                optimizer.zero_grad()

                # 打包数据
                x = torch.cat([x[i, :l] for i, l in enumerate(lens)])
                y = torch.cat([y[i, :l] for i, l in enumerate(lens)])
                output = self(x)
                if self.crf is None:
                    loss = self.lossfn(output, y)
                else:
                    output = torch.split(output, lens.tolist())
                    y = torch.split(y, lens.tolist())
                    # TODO
                    loss = sum(self.crf(emit, tiseq)
                               for emit, tiseq in zip(output, y))
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
        for x, lens, y in loader:
            # 打包数据
            x = torch.cat([x[i, :l] for i, l in enumerate(lens)])
            y = torch.cat([y[i, :l] for i, l in enumerate(lens)])

            output = self.forward(x)
            if self.crf is None:
                predict = torch.argmax(output, dim=1)
                loss += self.lossfn(output, y, reduction='sum').item()
                tp += torch.sum(y == predict).item()
            else:
                output = torch.split(output, lens.tolist())
                y = torch.split(y, lens.tolist())
                for emit, tiseq in zip(output, y):
                    predict = self.crf.viterbi(emit)
                    loss += self.crf(emit, tiseq).item()
                    tp += torch.sum(tiseq == predict).item()
            total += lens.sum().item()
        loss /= total
        return loss, tp, total, tp / total

    def collate_fn(self, data):
        x, lens, y = zip(*data)
        # 获取句子的最大长度
        maxlen = max(lens)
        # 去除无用的填充数据
        x = torch.stack(x)[:, :maxlen]
        lens = torch.tensor(lens)
        y = torch.stack(y)[:, :maxlen]
        return x, lens, y

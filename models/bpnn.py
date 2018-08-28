# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim

from modules import CRF


class BPNN(nn.Module):

    def __init__(self, window, vocdim, embdim, hiddim, outdim,
                 lossfn, use_crf=False, embed=None):
        super(BPNN, self).__init__()

        if embed is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(embed, False)

        # 隐藏层
        self.hid = nn.Sequential(nn.Linear(embdim * window, hiddim), nn.ReLU())
        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None
        # 损失函数
        self.lossfn = self.crf if use_crf else lossfn

        self.drop = nn.Dropout()

    def forward(self, x):
        B, T, N = x.shape
        # 获取词嵌入向量
        x = self.embed(x).view(B, T, -1)

        x = self.hid(x)
        x = self.drop(x)

        return self.out(x)

    def fit(self, train_loader, dev_loader, epochs, interval, eta, file):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.parameters(), lr=eta)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 更新参数
            self.update(train_loader)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, tp, total, accuracy = self.evaluate(train_loader)
            print(f"{'train:':<6} "
                  f"Loss: {loss:.4f} "
                  f"Accuracy: {tp} / {total} = {accuracy:.2%}")
            loss, tp, total, accuracy = self.evaluate(dev_loader)
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

    def update(self, loader):
        # 设置为训练模式
        self.train()

        # 从加载器中加载数据进行训练
        for x, lens, y in loader:
            # 清除梯度
            self.optimizer.zero_grad()
            # 获取掩码
            mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            target = y[mask]

            out = self(x)
            if self.crf is None:
                out = out[mask]
                loss = self.lossfn(out, target)
            else:
                out = out.transpose(0, 1)  # [T, B, N]
                y, mask = y.t(), mask.t()  # [T, B]
                loss = self.lossfn(out, y, mask)
            # 计算梯度
            loss.backward()
            # 更新参数
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        # 从加载器中加载数据进行评价
        for x, lens, y in loader:
            # 获取掩码
            mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            target = y[mask]

            out = self.forward(x)
            if self.crf is None:
                out = out[mask]
                predict = torch.argmax(out, dim=1)
                loss += self.lossfn(out, target)
            else:
                out = out.transpose(0, 1)  # [T, B, N]
                y, mask = y.t(), mask.t()  # [T, B]
                predict = self.crf.viterbi(out, mask)
                loss += self.lossfn(out, y, mask)
            tp += torch.sum(predict == target).item()
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

# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from modules import CRF, Layer


class ATTN(nn.Module):

    def __init__(self, vocdim, embdim, outdim,
                 lossfn, use_crf=False, pretrained=None):
        super(ATTN, self).__init__()

        self.encoder = Encoder(vocdim, embdim, Dm=embdim,
                               pretrained=pretrained)
        # 输出层
        self.out = nn.Linear(embdim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.lossfn = lossfn

    def forward(self, x, mask):
        x = self.encoder(x, mask)
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

        x, lens, y = batch
        # 获取掩码
        mask = y.ge(0)
        y = y[mask]

        out = self(x, mask)
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
        for x, lens, y in loader:
            # 获取掩码
            mask = y.ge(0)
            y = y[mask]

            out = self.forward(x, mask)
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
        x, lens, y = zip(*data)
        # 获取句子的最大长度
        max_len = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :max_len]
        lens = torch.tensor(lens)
        y = torch.stack(y)[:, :max_len]

        return x, lens, y


class Encoder(nn.Module):

    def __init__(self, vocdim, embdim,
                 L=6, H=5, Dk=20, Dv=20, Dm=100, Dh=200, p=0.2,
                 pretrained=None):
        super(Encoder, self).__init__()

        self.Dm = Dm
        self.embdim = embdim

        if pretrained is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)

        self.layers = nn.ModuleList([
            Layer(H, Dm, Dh, Dk, Dv, p) for _ in range(L)
        ])
        self.dropout = nn.Dropout(p)

    def init_pos(self, posdim, embdim):
        embed = torch.tensor([
            [pos / 10000 ** (i // 2 * 2 / embdim)
             for i in range(embdim)] for pos in range(posdim)
        ])
        embed[:, 0::2] = torch.sin(embed[:, 0::2])
        embed[:, 1::2] = torch.cos(embed[:, 1::2])
        return embed

    def forward(self, x, mask):
        B, T = x.shape

        x = self.embed(x)
        x += self.init_pos(T, self.embdim)

        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, mask)

        return out

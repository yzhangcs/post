# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules import CRF


class LSTM(nn.Module):

    def __init__(self, vocdim, embdim, hiddim, outdim,
                 lossfn, embed=None, crf=False, p=0.5):
        super(LSTM, self).__init__()

        if embed is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(embed, False)

        # 词嵌入LSTM层
        self.lstm = nn.LSTM(input_size=embdim,
                            hidden_size=hiddim,
                            batch_first=True,
                            bidirectional=True)

        # 输出层
        self.out = nn.Linear(hiddim * 2, outdim)
        # CRF层
        self.crf = CRF(outdim) if crf else None
        # 损失函数
        self.lossfn = lossfn if not crf else None

        self.drop = nn.Dropout(p)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取按长度有序的字序列索引
        lens, indices = torch.sort(lens, descending=True)
        # 获取逆序索引
        _, inverse_indices = indices.sort()
        # 序列按长度由大到小排列
        x = x[indices]
        # 获取字嵌入向量
        x = self.embed(x)
        x = self.drop(x)

        x = pack_padded_sequence(x, lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)

        out = self.out(x)
        # 恢复原有的顺序
        out = out[inverse_indices]

        return out

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
        for x, y, lens in loader:
            # 清除梯度
            self.optimizer.zero_grad()
            # 获取掩码
            mask = x.gt(0)
            target = y[mask]

            out = self(x, lens)
            if self.crf is None:
                out = out[mask]
                loss = self.lossfn(out, target)
            else:
                out = out.transpose(0, 1)  # [T, B, N]
                y, mask = y.t(), mask.t()  # [T, B]
                loss = self.crf(out, y, mask)
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
        for x, y, lens in loader:
            mask = x.gt(0)
            target = y[mask]

            out = self.forward(x, lens)
            if self.crf is None:
                out = out[mask]
                predict = torch.argmax(out, dim=1)
                loss += self.lossfn(out, target)
            else:
                out = out.transpose(0, 1)  # [T, B, N]
                y, mask = y.t(), mask.t()  # [T, B]
                predict = self.crf.viterbi(out, mask)
                loss += self.crf(out, y, mask)
            tp += torch.sum(predict == target).item()
            total += lens.sum().item()
        loss /= len(loader)

        return loss, tp, total, tp / total

    def collate_fn(self, data):
        x, y, lens = zip(*data)
        max_len = max(lens)
        x = torch.stack(x)[:, :max_len]
        y = torch.stack(y)[:, :max_len]
        lens = torch.tensor(lens)

        return x, y, lens

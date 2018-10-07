# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules import CRF


class LSTM_CRF(nn.Module):

    def __init__(self, n_vocab, n_embed, n_hidden, n_out,
                 embed=None, drop=0.5):
        super(LSTM_CRF, self).__init__()

        if embed is None:
            self.embed = nn.Embedding(n_vocab, n_embed)
        else:
            self.embed = nn.Embedding.from_pretrained(embed, False)

        # 词嵌入LSTM层
        self.lstm_crf = nn.LSTM(input_size=n_embed,
                                hidden_size=n_hidden,
                                batch_first=True,
                                bidirectional=True)

        # 输出层
        self.out = nn.Linear(n_hidden * 2, n_out)
        # CRF层
        self.crf = CRF(n_out)

        self.drop = nn.Dropout(drop)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取词嵌入向量
        x = self.embed(x)
        x = self.drop(x)

        x = pack_padded_sequence(x, lens, True)
        x, _ = self.lstm_crf(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)

        return self.out(x)

    def fit(self, train_loader, dev_loader, test_loader,
            epochs, interval, eta, file):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_acc = 0, 0.0
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.parameters(), lr=eta)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 更新参数
            self.update(train_loader)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, train_acc = self.evaluate(train_loader)
            print(f"{'train:':<6}  Loss: {loss:.4f} Accuracy: {train_acc:.2%}")
            loss, dev_acc = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {loss:.4f} Accuracy: {dev_acc:.2%}")
            loss, test_acc = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {loss:.4f} Accuracy: {test_acc:.2%}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # 保存效果最好的模型
            if dev_acc > max_acc:
                torch.save(self, file)
                max_e, max_acc = epoch, dev_acc
            elif epoch - max_e >= interval:
                break
        print(f"max accuracy of dev is {max_acc:.2%} at epoch {max_e}")
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
            out = out.transpose(0, 1)  # [T, B, N]
            y, mask = y.t(), mask.t()  # [T, B]
            predict = self.crf.viterbi(out, mask)
            loss += self.crf(out, y, mask)
            tp += torch.sum(predict == target).item()
            total += lens.sum().item()
        loss /= len(loader)

        return loss, tp / total

    def collate_fn(self, data):
        x, y, lens = zip(
            *sorted(data, key=lambda x: x[-1], reverse=True)
        )
        max_len = lens[0]
        x = torch.stack(x)[:, :max_len]
        y = torch.stack(y)[:, :max_len]
        lens = torch.tensor(lens)

        return x, y, lens

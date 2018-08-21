# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader

from modules import CRF, CharLSTM, Encoder


class LSTM_CHAR(nn.Module):

    def __init__(self, vocdim, chrdim,
                 embdim, char_embdim, hiddim, outdim,
                 lossfn, use_attn=False, use_crf=False, bidirectional=False,
                 pretrained=None):
        super(LSTM_CHAR, self).__init__()

        if pretrained is None:
            self.embed = nn.Embedding(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)
        # 字嵌入LSTM层
        self.clstm = CharLSTM(chrdim=chrdim,
                              embdim=embdim,
                              hiddim=char_embdim,
                              bidirectional=bidirectional)

        # 词嵌入LSTM层
        hidden_size = hiddim // 2 if bidirectional else hiddim
        self.wlstm = nn.LSTM(input_size=embdim + char_embdim,
                             hidden_size=hidden_size,
                             batch_first=True,
                             bidirectional=bidirectional)
        self.encoder = Encoder(L=1,
                               H=5,
                               Dk=hiddim // 5,
                               Dv=hiddim // 5,
                               Dm=hiddim,
                               Dh=hiddim * 2) if use_attn else None

        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.drop = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x, lens, char_x, char_lens):
        B, T, N = x.shape
        # 获取词嵌入向量
        x = self.embed(x).view(B, T, -1)

        mask = torch.arange(T) < lens.unsqueeze(-1)

        # 获取字嵌入向量
        char_x = self.clstm(char_x[mask], char_lens[mask])
        char_x = pad_sequence(torch.split(char_x, lens.tolist()), True)

        # 拼接词表示和字表示
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, hidden = self.wlstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.encoder(x, mask) if self.encoder is not None else self.drop(x)

        return self.out(x)

    def fit(self, trainset, devset, file, epochs, batch_size, interval, eta):
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

        x, lens, char_x, char_lens, y = batch
        B, T, N = x.shape
        mask = torch.arange(T) < lens.unsqueeze(-1)
        target = y[mask]

        out = self(x, lens, char_x, char_lens)
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
    def evaluate(self, dataset, batch_size=50):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        loader = DataLoader(dataset, batch_size, collate_fn=self.collate_fn)
        for x, lens, char_x, char_lens, y in loader:
            B, T, N = x.shape
            mask = torch.arange(T) < lens.unsqueeze(-1)
            target = y[mask]

            out = self.forward(x, lens, char_x, char_lens)
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
        # 按照长度调整顺序
        data.sort(key=lambda x: x[1], reverse=True)
        x, lens, char_x, char_lens, y = zip(*data)
        # 获取句子的最大长度
        max_len = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :max_len]
        lens = torch.tensor(lens)
        char_x = torch.stack(char_x)[:, :max_len]
        char_lens = torch.stack(char_lens)[:, :max_len]
        y = torch.stack(y)[:, :max_len]

        return x, lens, char_x, char_lens, y

# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, TensorDataset

from crf import CRF


class LSTM(nn.Module):

    def __init__(self, vocdim, chrdim, embdim, cembdim, hiddim, outdim,
                 lossfn, use_crf=False, bidirectional=False,
                 pretrained=None):
        super(LSTM, self).__init__()

        if pretrained is None:
            self.embed = torch.randn(vocdim, embdim)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained, False)

        self.clstm = CharLSTM(chrdim, embdim, cembdim,
                              bidirectional=bidirectional)
        # 词嵌入LSTM层
        if bidirectional:
            self.wlstm = nn.LSTM(embdim + cembdim, hiddim // 2,
                                 batch_first=True,
                                 bidirectional=True)
        else:
            self.wlstm = nn.LSTM(embdim + cembdim, hiddim,
                                 batch_first=True,
                                 bidirectional=False)

        # 输出层
        self.out = nn.Linear(hiddim, outdim)
        # CRF层
        self.crf = CRF(outdim) if use_crf else None

        self.dropout = nn.Dropout()
        self.lossfn = lossfn

    def forward(self, x, cx, lens, clens):
        B, T = x.shape
        # 获取词嵌入向量
        x = self.embed(x)
        x = x.view(B, T, -1)
        # 获取字嵌入向量
        cx = torch.cat([cx[i, :l] for i, l in enumerate(lens)])
        clens = torch.cat([clens[i, :l] for i, l in enumerate(lens)])
        cx = self.clstm(cx, clens)
        cx = pad_sequence(torch.split(cx, lens.tolist()), True)

        # 拼接词表示和字表示
        x = torch.cat((x, cx), dim=-1)
        # 打包数据
        x = pack_padded_sequence(x, lens, True)
        x, hidden = self.wlstm(x)
        x = pad_packed_sequence(x, True)[0]
        x = self.dropout(x)
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
                                  shuffle=True)
        # 设置评价过程数据加载器
        train_eval_loader = DataLoader(trainset, len(trainset))
        dev_eval_loader = DataLoader(devset, len(devset))
        # 设置优化器为Adam
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=lmbda)

        for epoch in range(epochs):
            start = datetime.now()
            for x, cx, lens, clens, y in train_loader:
                optimizer.zero_grad()

                # 获取长度由大到小排列的词序列索引
                lens, indices = lens.sort(descending=True)
                maxlen = lens[0]
                # 调整序列顺序并去除无用数据
                x = x[indices, :maxlen]
                y = y[indices, :maxlen]
                cx = cx[indices, :maxlen]
                clens = clens[indices, :maxlen]

                output = self(x, cx, lens, clens)
                if self.crf is None:
                    output = pack_padded_sequence(output, lens, True).data
                    y = pack_padded_sequence(y, lens, True).data
                    loss = self.lossfn(output, y)
                else:
                    # TODO
                    loss = sum(self.crf(output[i, :l], y[i, :l])
                               for i, l in enumerate(lens))
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

    def evaluate(self, loader):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        with torch.no_grad():
            for x, cx, lens, clens, y in loader:
                maxlen = lens[0]
                # 去除无用数据
                x, y = x[:, :maxlen], y[:, :maxlen]

                output = self.forward(x, cx, lens, clens)
                if self.crf is None:
                    output = pack_padded_sequence(output, lens, True).data
                    y = pack_padded_sequence(y, lens, True).data

                    predict = torch.argmax(output, dim=1)
                    loss += self.lossfn(output, y, size_average=False)
                    tp += (predict == y).sum().item()
                else:
                    # TODO
                    for i, length in enumerate(lens):
                        emit, tiseq = output[i, :length], y[i, :length]
                        predict = self.crf.viterbi(emit)
                        loss += self.crf(emit, tiseq)
                        tp += torch.sum(predict == tiseq).sum().item()
                total += lens.sum().item()
        loss /= total
        return loss, tp, total, tp / total


class CharLSTM(nn.Module):

    def __init__(self, chrdim, embdim, hiddim, bidirectional):
        super(CharLSTM, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(chrdim, embdim)
        # 字嵌入LSTM层
        if bidirectional:
            self.lstm = nn.LSTM(embdim, hiddim // 2,
                                batch_first=True,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(embdim, hiddim,
                                batch_first=True,
                                bidirectional=False)

    def forward(self, x, lens):
        B, T = x.shape
        # 获取字嵌入向量
        x = self.embed(x)
        # 获取长度由大到小排列的字序列索引
        lens, indices = lens.sort(descending=True)
        # 获取反向索引用来恢复原有的顺序
        _, reversed_indices = indices.sort()
        # 获取单词最大长度
        maxlen = lens[0]
        # 序列按长度由大到小排列
        x = x.view(B, T, -1)[indices, :maxlen]
        x = pack_padded_sequence(x, lens, True)

        x, hidden = self.lstm(x)
        representations = torch.cat(torch.unbind(hidden[0]), dim=1)
        # 返回词的字符表示
        return representations[reversed_indices]

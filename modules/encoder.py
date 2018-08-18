# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, L=6, H=5, Dk=20, Dv=20, Dm=100, Dh=200, p=0.2):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            Layer(H, Dm, Dh, Dk, Dv, p) for _ in range(L)
        ])
        self.dropout = nn.Dropout(p)

    def init_pos(self, T, N):
        embed = torch.tensor([
            [pos / 10000 ** (i // 2 * 2 / N)
             for i in range(N)] for pos in range(T)
        ])
        embed[:, 0::2] = torch.sin(embed[:, 0::2])
        embed[:, 1::2] = torch.cos(embed[:, 1::2])
        return embed

    def forward(self, x, mask):
        B, T, N = x.shape

        x += self.init_pos(T, N)
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, mask)

        return out


class Layer(nn.Module):

    def __init__(self, H, Dm, Dh, Dk, Dv, p=0.2):
        super(Layer, self).__init__()

        self.attn = MultiHeadAttn(H, Dm, Dk, Dv, p)
        self.ffn = PosWiseFFN(Dm, Dh, p)

    def forward(self, x, mask=None):
        out = self.attn(x, x, x, mask)
        out = self.ffn(out)
        return out


class MultiHeadAttn(nn.Module):

    def __init__(self, H, Dm, Dk, Dv, p=0.2):
        super(MultiHeadAttn, self).__init__()

        self.H = H
        self.Dk = Dk
        self.Dv = Dv

        self.wq = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wk = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wv = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dv)))

        self.norm = nn.LayerNorm(Dm)
        self.proj = nn.Linear(H * Dv, Dm)

        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        residual = q
        B, Tq, Dm = q.shape
        B, Tk, Dm = k.shape
        B, Tv, Dm = v.shape
        H, Dk, Dv = self.H, self.Dk, self.Dv

        q = q.repeat(H, 1, 1).view(H, -1, Dm)
        k = k.repeat(H, 1, 1).view(H, -1, Dm)
        v = v.repeat(H, 1, 1).view(H, -1, Dm)
        # [H * B, Tq, Dk]
        q = torch.bmm(q, self.wq).view(-1, Tq, Dk)
        # [H * B, Tk, Dk]
        k = torch.bmm(k, self.wk).view(-1, Tk, Dk)
        # [H * B, Tv, Dv]
        v = torch.bmm(v, self.wv).view(-1, Tv, Dv)

        # Scaled Dot-Product Attention
        attn = torch.bmm(q, k.transpose(1, 2)) / Dm ** 0.5
        if mask is not None:
            mask = mask.repeat(H, 1).unsqueeze(1)
            attn.masked_fill_(1 - mask, -float('inf'))
        # [H * B, Tq, Tk]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # [H * B, Tq, Dv]
        out = torch.matmul(attn, v)

        # [B, Tq, H * Dv]
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)

        out = self.proj(out)
        out = self.dropout(out)

        return self.norm(out + residual)


class PosWiseFFN(nn.Module):

    def __init__(self, Dm, Dh, p=0.2):
        super(PosWiseFFN, self).__init__()

        self.w1 = nn.Linear(Dm, Dh)
        self.w2 = nn.Linear(Dh, Dm)
        self.norm = nn.LayerNorm(Dm)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        residual = x
        out = F.relu(self.w1(x))
        out = self.w2(out)
        out = self.dropout(out)

        return self.norm(out + residual)

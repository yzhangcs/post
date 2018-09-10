# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, L, H, Dk, Dv, Dm, Dh, p=0.2):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([
            Layer(H, Dk, Dv, Dm, Dh, p) for _ in range(L)
        ])
        self.drop = nn.Dropout(p)

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

        out = self.drop(x)
        for layer in self.layers:
            out = layer(out, mask)

        return out


class Layer(nn.Module):

    def __init__(self, H, Dk, Dv, Dm, Dh, p=0.2):
        super(Layer, self).__init__()

        self.attn = MultiHeadATTN(H, Dk, Dv, Dm, p)
        self.ffn = PosWiseFFN(Dm, Dh, p)

    def forward(self, x, mask):
        out = self.attn(x, x, x, mask)
        out = self.ffn(out)

        return out


class MultiHeadATTN(nn.Module):

    def __init__(self, H, Dk, Dv, Dm, p=0.2):
        super(MultiHeadATTN, self).__init__()

        self.H = H
        self.Dk = Dk
        self.Dv = Dv
        self.scale = Dk ** 0.5

        self.wq = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wk = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wv = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dv)))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(H * Dv, Dm)
        self.norm = nn.LayerNorm(Dm)
        self.drop = nn.Dropout(p)

    def forward(self, q, k, v, mask):
        residual = q
        B, Tq, Dm = q.shape
        B, Tk, Dm = k.shape
        B, Tv, Dm = v.shape
        H, Dk, Dv = self.H, self.Dk, self.Dv

        q = (q @ self.wq.unsqueeze(1)).view(-1, Tq, Dk)  # [H * B, Tq, Dk]
        k = (k @ self.wk.unsqueeze(1)).view(-1, Tk, Dk)  # [H * B, Tk, Dk]
        v = (v @ self.wv.unsqueeze(1)).view(-1, Tv, Dv)  # [H * B, Tv, Dv]

        # Scaled Dot-Product Attention
        mask = mask.repeat(H, 1).unsqueeze(1)  # [H * B, 1, Tk]
        attn = (q @ k.transpose(1, 2)) / self.scale  # [H * B, Tq, Tk]
        attn = attn.masked_fill(1 - mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.drop(attn)

        out = attn @ v  # [H * B, Tq, Dv]
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)  # [B, Tq, H * Dv]
        out = self.proj(out)  # [B, Tq, Dm]
        out = self.drop(out)

        return self.norm(out + residual)


class PosWiseFFN(nn.Module):

    def __init__(self, Dm, Dh, p=0.2):
        super(PosWiseFFN, self).__init__()

        self.w1 = nn.Sequential(nn.Linear(Dm, Dh), nn.ReLU())
        self.w2 = nn.Linear(Dh, Dm)
        self.norm = nn.LayerNorm(Dm)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = self.w2(x)
        x = self.drop(x)

        return self.norm(x + residual)

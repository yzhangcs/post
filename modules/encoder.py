# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, L, H, Dk, Dv, Dm, Dh, p=0.2):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            Layer(H, Dm, Dh, Dk, Dv, p) for _ in range(L)
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

    def __init__(self, H, Dm, Dh, Dk, Dv, p=0.2):
        super(Layer, self).__init__()

        self.attn = MultiHeadAttn(H, Dm, Dk, Dv, p)
        self.ffn = PosWiseFFN(Dm, Dh, p)

    def forward(self, x, mask):
        out = self.attn(x, x, x, mask)
        out = self.ffn(out)
        return out


class MultiHeadAttn(nn.Module):

    def __init__(self, H, Dm, Dk, Dv, p=0.2):
        super(MultiHeadAttn, self).__init__()

        self.H = H
        self.Dk = Dk
        self.Dv = Dv
        self.scale = Dk ** 0.5

        self.wq = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wk = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dk)))
        self.wv = nn.Parameter(nn.init.xavier_normal_(torch.empty(H, Dm, Dv)))

        self.norm = nn.LayerNorm(Dm)
        self.proj = nn.Linear(H * Dv, Dm)

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
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = attn @ v  # [H * B, Tq, Dv]
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)  # [B, Tq, H * Dv]
        out = self.proj(out)
        out = self.drop(out)

        return self.norm(out + residual)


class PosWiseFFN(nn.Module):

    def __init__(self, Dm, Dh, p=0.2):
        super(PosWiseFFN, self).__init__()

        self.w1 = nn.Linear(Dm, Dh)
        self.w2 = nn.Linear(Dh, Dm)
        self.norm = nn.LayerNorm(Dm)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        residual = x
        out = F.relu(self.w1(x))
        out = self.w2(out)
        out = self.drop(out)

        return self.norm(out + residual)

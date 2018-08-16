# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):

    def __init__(self, H, Dm, Dh, Dk, Dv, p=0.1):
        super(Layer, self).__init__()

        self.attn = MultiHeadAttn(H, Dm, Dk, Dv, p)
        self.ffn = PosWiseFFN(Dm, Dh, p)

    def forward(self, x, mask=None):
        out = self.attn(x, x, x, mask)
        out = self.ffn(out)
        return out


class MultiHeadAttn(nn.Module):

    def __init__(self, H, Dm, Dk, Dv, p=0.1):
        super(MultiHeadAttn, self).__init__()

        self.H = H
        self.Dk = Dk
        self.Dv = Dv
        self.wq = nn.init.xavier_normal_(torch.empty(H, Dm, Dk))
        self.wk = nn.init.xavier_normal_(torch.empty(H, Dm, Dk))
        self.wv = nn.init.xavier_normal_(torch.empty(H, Dm, Dv))

        self.attn = ScaledDotProductAttention(Dm)
        self.norm = nn.LayerNorm(Dm)
        self.proj = nn.Linear(H * Dv, Dm)

        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        residual = q
        B, Lq, Dm = q.shape
        B, Lk, Dm = k.shape
        B, Lv, Dm = v.shape
        H, Dk, Dv = self.H, self.Dk, self.Dv

        # [H * B, Lq, Dk]
        q = torch.matmul(q.unsqueeze(1), self.wq).view(-1, Lq, Dk)
        # [H * B, Lk, Dk]
        k = torch.matmul(k.unsqueeze(1), self.wk).view(-1, Lk, Dk)
        # [H * B, Lv, Dv]
        v = torch.matmul(v.unsqueeze(1), self.wv).view(-1, Lv, Dv)
        if mask is not None:
            mask = mask.repeat(H, 1)

        # [H * B, Lq, Dv]
        out, _ = self.attn(q, k, v, mask)
        # [B, Lq, H * Dv]
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)

        out = self.proj(out)
        out = self.dropout(out)

        return self.norm(out + residual)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, Dm, p=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = Dm ** 0.5
        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        if mask is not None:
            attn[1 - mask] = -float('inf')
        attn = F.softmax(attn, dim=0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        return out, attn


class PosWiseFFN(nn.Module):

    def __init__(self, Dm, Dh, p=0.1):
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

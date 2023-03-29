from torch import nn
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(PreNorm(dim, SelfAttention(dim, heads=heads))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim

        self.to_qkv1 = nn.Linear(dim, dim * 3, bias=False)
        self.to_q2 = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x1 = x[:, :, :]
        qkv = self.to_qkv1(x1)
        q, k, v = rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=h)  # h: head t: feature num   d:feat vector

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
from torch import nn
#from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
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
        return self.fn(x) + x  #直接和x相加，即残差是fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim):   #input_dim：就是embedding-dim 96, mlp_dim:hidden-dim 48, depth: 层数
        super().__init__()
        layers = []
        for _ in range(depth):  #4个block
            layers.extend(
                [
                    Residual(PreNorm(input_dim, SelfAttention(input_dim, heads=heads))), #自注意力+残差，+前置归一标准化。  post-norm和pre-norm其实各有优势，post-norm在残差之后做归一化，对参数正则化的效果更强，进而模型的鲁棒性也会更好；pre-norm相对于post-norm，因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，正好可以防止模型的梯度爆炸或者梯度消失，因此，这里笔者可以得出的一个结论是如果层数少post-norm的效果其实要好一些，如果要把层数加大，为了保证模型的训练，pre-norm显然更好一些。
                    Residual(PreNorm(input_dim, FeedForward(input_dim, mlp_dim))),# 96*48
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, heads=8):  #头数，默认8
        super().__init__()
        self.heads = heads  #多头自注意力
        self.scale = input_dim ** -0.5   #1/input_dim开根号
        self.input_dim = input_dim #自注意力的输入数据维度

        self.to_qkv1 = nn.Linear(input_dim, input_dim * 3, bias=False)   #线性连接层，又叫全连接层，是通过矩阵的乘法将前一层的矩阵变换为下一层矩阵。  和tensorflow的dense把激活函数设为None的效果相同：activation=None， QKV三个参数矩阵合并了，称为一个参数矩阵。没有bias.
        self.to_out = nn.Linear(input_dim, input_dim) #out 线性层

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  #b，n, _:x的维度5*32*96，h:多头注意力的头数
        x1 = x[:, :, :]
        qkv = self.to_qkv1(x1)# 调用全连接层， 得到x对应的QKV矩阵参数，每个矩阵96*96，矩阵参数尺寸：总共96*（96*3）元素 因为to_qkv1是一个input_dim, input_dim * 3的全连接层，没有偏置，只有Wx； x: 5*32*96
        #下面再把QKV里的Q,K,V分离出来，b:5, h:8, m:32, d:12
        q, k, v = rearrange(qkv, 'b m (qkv h d) -> qkv b h m d', qkv=3, h=h)  # b: batch-size h: head 多头注意力的头数 m: feature num , 特征向量的个数，本程序32   d:feat vector， d: input_dimension/head = 12， 作为输入的qkv是张量通过rearrange分解为 q, k, v 张量,作为参数的qkv=3
        #下面开始K^T.q  b:5, h:8, i: 32, j:12, k^T, 5, 8, 12, 32
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale    #内积：实现q . k^T (给定bhi,q[bhi]是d维的行向量，k[bhj]是d维行向量) q维度（b,h, i, d），k^T维度（b, h, d, j）  爱因斯坦求和，可实现向量内积、矩阵乘法等张量操作。以箭头分隔，箭头左边表示输入张量，箭头右边则表示输出张量。输入张量部分的逗号分割多个输入张量，。
        #alpha 计算， softmax, alpha的尺寸: 5,8,32，32 ,所以一个alpha向量是32维的
        attn = dots.softmax(dim=-1)  #注意力的权重因子  alpha = softmax(K^Tq)
        #用alpha做权重，求V的向量的线性组合。 alpha的尺寸: 5,8,32，32 , v尺寸：5,8,32,12， 求出out: 5,8,32,12 (b h m d )
        out = torch.einsum('bhij,bhjd->bhid', attn, v) #上下文向量 = 用权重因子把V张量（维度是b,h m,d）的各个列向量进行线性组合。结果的维度是b,h m,d
        #b,h m,d张量 rearrange, b=5, m=32, hd=96（8*12）即：5*32*96
        out = rearrange(out, 'b h m d -> b m (h d)')   # b, h, m, d, b=5, h=8, m=32, d=96 ----> 多头合并：b=5, m=32, hd（8*12）
        out = self.to_out(out)  #5*32*96
        return out
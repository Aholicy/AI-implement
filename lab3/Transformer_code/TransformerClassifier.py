import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from transformers import BertModel, BertConfig
from Transformers import TransformerModel


class TransformerClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(TransformerClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding嵌入，将token转化为向量
        self.transformer = TransformerModel(
            input_dim=embedding_dim, depth=4, heads=8, mlp_dim=hidden_dim      #按本程序来说：embedding后每个词向量的维度是96， mlp_dim =48, depth = block-num: 4
        )
        self.hidden2label = nn.Linear(embedding_dim, label_size)   # 线性层分类器

    def forward(self, sentence):
        # sentence: batch-size 5 x sentence_len 32
        embeds = self.word_embeddings(sentence)
        # wor_embedding之后:  5 x 32 x embedding_dim 96
        transformer_out = self.transformer(embeds)
        y = self.hidden2label(transformer_out[:, -1, :])   # 取序列中最后一个隐藏状态作为输出， 也有取所有隐藏状态平均的做法
        return y  #5 x 32 x embedding_dim（96）

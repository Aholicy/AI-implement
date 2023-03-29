import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from transformers import BertModel, BertConfig
from LSTM import myLstm
#这个文件相当于定义了接口，可以采用不同的模型。
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding嵌入，将token转化为向量 在本程序向量维度是96
        self.lstm = myLstm(embedding_dim, hidden_dim)   # lstm主体  传入input_sz, hidden_sz class myLstm(nn.Module): myLSTM: __init__(self, input_sz, hidden_sz):
        self.hidden2label = nn.Linear(hidden_dim, label_size)   # 线性层分类器  隐藏状态h_t --->类

    def forward(self, sentence):
        # sentence:  5 x 32 x 1 其中，5:batch size, 32： sentence length, 1: token, 就是字典里的idx
        embeds = self.word_embeddings(sentence)
        # wor_embedding:  5 x 32 x embedding_dim, 嵌入映射成向量，token映射为96维的向量 embedding函数内部有参数簇，通过反向传播，最终得出最好的表达。
        lstm_out = self.lstm(embeds) #启动LSTM
        y = self.hidden2label(lstm_out[:, -1, :])   # 取序列中最后一个隐藏状态作为输出， 也有取所有隐藏状态作平均的做法
        return y

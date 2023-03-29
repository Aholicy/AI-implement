import torch
import torch.nn as nn
import math

class myLstm(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # 输入门 input 参数  𝑖_𝑡 = 𝜎(𝑊_𝑖 𝑥_𝑡+𝑈_𝑖 ℎ_(𝑡−1)+𝑏_𝑖)
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # 遗忘门f_t forget 参数 𝑓_𝑡 = 𝜎(𝑊_𝑓 𝑥_𝑡+𝑈_𝑓 ℎ_(𝑡−1)+𝑏_𝑓 )
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # 传输带c_t cell state 参数 𝑐_𝑡 = 𝑓_𝑡⊙𝑐_(𝑡−1)+𝑖_𝑡⊙𝑐 ̃_𝑡 \\ c ̃_t = tanh(𝑊_𝑐 𝑥_𝑡+𝑈_𝑐 ℎ_(𝑡−1)+𝑏_𝑐) 
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # 输出门o_t  参数 𝑜_𝑡 = 𝜎(𝑊_𝑜 𝑥_𝑡+𝑈_𝑜 ℎ_(𝑡−1)+𝑏_𝑜),   ℎ_𝑡 = 𝑜_𝑡⊙tanh⁡(𝑐_𝑡)
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)  #
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)# 这里uniform()把数值范围限定在[-stdv, stdv]均匀分布

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()  #数据
        hidden_seq = []

        # 初始话隐藏状态ht以及记忆状态ct
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):     # 将序列依次放入LSTM单元中
            x_t = x[:, t, :]
            # @ 为矩阵运算
            i_t = torch.relu(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            #i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)     # 输入门，调整输入的x_t及h_t比例  𝑖_𝑡 = 𝜎(𝑊_𝑖 𝑥_𝑡+𝑈_𝑖 ℎ_(𝑡−1)+𝑏_𝑖)
            f_t = torch.relu(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            #f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)     # 遗忘门，以往上一次的 𝑓_𝑡 = 𝜎(𝑊_𝑓 𝑥_𝑡+𝑈_𝑓 ℎ_(𝑡−1)+𝑏_𝑓 )
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)        # 补给记忆,中间结果 c ̃_t = tanh(𝑊_𝑐 𝑥_𝑡+𝑈_𝑐 ℎ_(𝑡−1)+𝑏_𝑐) 
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)     # 输出门，用于生成输出
            c_t = f_t * c_t + i_t * g_t     # 计算下一个cell state， 记忆状态c_t+1  𝑐_𝑡 = 𝑓_𝑡⊙𝑐_(𝑡−1)+𝑖_𝑡⊙𝑐 ̃_𝑡 \\ c ̃_t = tanh(𝑊_𝑐 𝑥_𝑡+𝑈_𝑐 ℎ_(𝑡−1)+𝑏_𝑐) 
            h_t = o_t * torch.tanh(c_t)     # 计算下一个隐藏状态, h_t+1  ℎ_𝑡=𝑜_𝑡⊙tanh⁡(𝑐_𝑡)

            hidden_seq.append(h_t.unsqueeze(1))     # 一直以来的隐藏状态记录到列表中， 添加一个dim=1的向量 [batch_size, 1, hidden_dim]， unsqueeze（）函数对数据维度进行扩充，给指定位置加上维数为一的维度，比如原本有个两行两列的数据【2，3】，在1的位置加了一维就变成一行两列【2，1，3】。
        hidden_seq = torch.cat(hidden_seq, dim=1)   # 连接为tensor  concate，torch.cat函数： 在给定维度dim=1上对输入的张量序列seq 进行连接操作。 batch_size，32， hidden_dim   32个h_t（序列维度一般都设为第2个），每个h_t的维度是
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()        # contiguous 使矩阵连续存放，不止改变表现形式
        return hidden_seq   # 输出隐藏状态 [batch_size, seq_len, hidden_dim]
import os
import torch
import copy
from torch.utils.data import DataLoader
import DataProcessing as DP
import TransformerClassifier as TransformerCL  #Transformer分类器
import torch.optim as optim
import torch.nn as nn
import PlotFigure as PF
from datetime import datetime
import pickle

use_plot = True
use_save = True

# R8 数据集，新闻分类，类别如下
# 船，运输
# 金钱外汇
# 粮食
# 收购
# 贸易
# 赚钱
# 原油
# 利益，利息，利润

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

# 参数设置
epochs = 25
batch_size = 5  #batch的大小
use_gpu = torch.cuda.is_available()
device = 'cuda:0' if use_gpu else 'cpu'
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch): 
    lr = learning_rate * (0.1 ** (epoch // 10))#整除10，每10个epoch就把学习率*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    # 数据加载参数
    embedding_dim = 96  #token转换的维度，把token映射成96维度的向量。embedding layer
    hidden_dim = 48  #隐藏状态的维度
    sentence_len = 32 #多于32就截断，少于32就补0
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)#训练文件路径+所有训练文件的列表文件名
    test_file = os.path.join(DATA_DIR, TEST_FILE)#测试文件路径+文件名
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train] #真正的训练文件名 如:2.txt
    filenames = copy.deepcopy(train_filenames)#deepcopy 把所有字节都拷贝
    fp_train.close()#列表文件关闭
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]#测试文件夹下面真正的测试文件名 如:2.txt
    fp_test.close()
    filenames.extend(test_filenames)#根据所有文件名组成字典

    corpus = DP.Corpus(DATA_DIR, filenames)#NLP常用数据预处理，在DataProcessing.py中实现，它先建立字典，知道了字典的大小，把列表文件中所有的文件的句子转换为数字代表的token
    nlabel = 8


    # 模型
    model = TransformerCL.TransformerClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           vocab_size=len(corpus.dictionary), label_size=nlabel)  #embedding （x_t）的维度 96(对应 512),  m=32，hidden (h_t)维度 48，字典的大小，类的个数
    if use_gpu:
        model = model.cuda()

    # 训练数据加载及预处理
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
    train_loader = DataLoader(dtrain_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
    # 测试数据加载及预处理
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)
    test_loader = DataLoader(dtest_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4
                         )

    # 优化器及损失函数
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 存放loss 和 acc 准确率
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    # 训练主体
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        # 训练阶段
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            # 获得 输入 输出
            train_inputs, train_labels = traindata
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            train_labels = torch.squeeze(train_labels)

            optimizer.zero_grad()
            output = model(train_inputs)

            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum().cpu()
            total += len(train_labels)
            total_loss += loss.item()
        # 计算单个epoch的训练集平均准确率和总损失
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)

        # 测试阶段
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        with torch.no_grad():
            for iter, testdata in enumerate(test_loader):
                test_inputs, test_labels = testdata
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_labels = torch.squeeze(test_labels)

                output = model(test_inputs)
                loss = loss_function(output, test_labels)

                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == test_labels).sum().cpu()
                total += len(test_labels)
                total_loss += loss.item()
            # 计算单个epoch的测试集平均准确率和总损失
            test_loss_.append(total_loss / total)
            test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))


    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_

    if use_plot:    # 保存图片
        PF.PlotFigure(result, use_save)

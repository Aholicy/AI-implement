# minist手写数字分类任务
# 模型使用多层感知机（mlp）


# 导入必要的包
import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

# 数据标准化处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 下载数据集
data_train = datasets.MNIST(root="../data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="../data/",
                           transform=transform,
                           train=False,
                           download=True)

# 装载数据
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

num_i = 28 * 28  # 输入层维度；
num_h = 100  # 隐含层维度；
num_o = 10  # 输出层维度，0-9共十个数字，所以输出是一个10维的向量，向量每一个元素是0-1的概率；
batch_size = 64


#class Model(torch.nn.Module):

    # def __init__(self, num_i, num_h, num_o):
    #     super(Model, self).__init__()

    #     self.linear1 = torch.nn.Linear(num_i, num_h)  # 输入 28*28 ---100
    #     self.relu = torch.nn.ReLU()
    #     self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层 100---100
    #     self.relu2 = torch.nn.ReLU()
    #     self.linear3 = torch.nn.Linear(num_h, num_o)  # 100 - 10


    # def forward(self, x):
    #     x = self.linear1(x)
    #     x = self.relu(x)
    #     x = self.linear2(x)
    #     x = self.relu2(x)
    #     x = self.linear3(x)
    #     return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 首先找到CNN的父类（比如是类A），然后把类CNN的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,
                               padding=1)  # 添加第一个卷积层,调用了nn里面的Conv2d()，输入的灰度图，所以 in_channels=1, out_channels=32 说明使用了32个滤波器/卷积核
        self.pool1 = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window 即最大池化层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 2层， 输入通道in_channels 要等于上一层的 out_channels
        self.pool2 = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window 即最大池化层

        # 接着三个全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)  # 全连接层的输入特征维度为64*7*7，因为上一层Conv2d的out_channels=64，两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # in_features：每个输入（x）样本的特征的大小
        # out_features：每个输出（y）样本的特征的大小

    def forward(self, x):  # 这里定义前向传播的方法
        x = self.pool1(F.relu(self.conv1(x)))  # torch.nn.relu()将ReLU层添加到网络。
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1,
                   128 * 7 * 7)  # 和flatten类似，将特征图转换为一个1维的向量。第一个参数-1是说这个参数由另一个参数确定， 比如：矩阵在元素总数一定的情况下，确定列数就能确定行数。第一个全连接层的首参数是64*7*7，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# model = Model(num_i, num_h, num_o)
model = CNN()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters()) #默认学习率1e-3。
epochs = 5
for epoch in range(epochs):
    sum_loss = 0
    train_correct = 0
    for data in data_loader_train:
        inputs, labels = data  # inputs 维度：[64,1,28,28]  batch_size, 灰度， 宽， 高
        #     print(inputs.shape)

        # 卷积网络直接输入，不用展平
        # inputs = torch.flatten(inputs, start_dim=1)  # 展平数据，转化为[64,784]，把图像二维矩阵变成一维向量

        #     print(inputs.shape)
        outputs = model(inputs)  # pytorch 的torch.nn.Module继承类调用本类时，会自动把参数传给forward()函数
        optimizer.zero_grad()  # 把优化器的梯度清0，不然其中会有上一次训练的梯度数据
        loss = cost(outputs, labels)  # cost定义在上面，交叉熵损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 调用优化器更新参数，本程序调用Adam
        # 这里没有用softmax, torch.max更省事一些。有时候可以用softmax，在本数据集测试时发现收敛会较慢，但一般情况下并不能确定。本版本直接用torch.max。
        _, id = torch.max(outputs.data, 1)  # dim=? 取outputs.data里面的第几维数据的最大值，从0开始，0：数据的第一维。
        sum_loss += loss.data  # 交叉熵损失累计
        train_correct += torch.sum(id == labels.data)  # 评估在训练集的准确率
    print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(data_loader_train)))
    print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
# 以下还可加入验证集保存最优模型参数，用最优模型再去做预测效果可能更好。
model.eval()  # 开启预测模式，以下内容操作和上方类似
test_correct = 0
for data in data_loader_test:
    inputs, lables = data
    # 用卷积网络不用展平
    # inputs = torch.flatten(inputs, start_dim=1)  # 展并数据
    outputs = model(inputs)
    _, id = torch.max(outputs.data, 1)
    test_correct += torch.sum(id == lables.data)
print("correct:%.3f%%" % (100 * test_correct / len(data_test)))
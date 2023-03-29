import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {} #key, value字典是另一种可变容器模型，且可存储任意类型对象。字典的每个键值 key:value 对用冒号 : key---> value,通过单词给出idx。
        self.idx2word = [] #idx --> word, 跟上面的word2idx = {} 保存了数组下标，通过下标就能给出word.

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word) #把word加入 idx2word，同时给数组下标idx。
            self.word2idx[word] = len(self.idx2word) - 1 #把word加入 word2idx，到时候可以用word查idx  
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>'] #连成一句话words line每次读一行，line.split()按空格切分单词放到单词列表words。 '<eos>'是end of string
                    tokens += len(words)#统计多少个单词
                    for word in words:
                        self.dictionary.add_word(word)#把words里的word都加入字典（上面定义的，按顺序的idx）

            # Tokenize file content
            with open(path, 'r') as f:  #这里是否有必要再打开文件？其实可以把文件指针重设为0 f.seek(0, 0)
                ids = torch.LongTensor(tokens)#定义一个文件所有单词个数那么大的tensor
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']  #这里重复操作了，可以优化成边加字典的单词，边查出idx,进行tokenize
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1 # 真正的token, 

        return ids

class TxtDatasetProcessing(Dataset):   #训练和测试集txt包括标签文件文件处理
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path) # 
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)
        fp = open(txt_filepath, 'r')
        self.txt_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len


    def __getitem__(self, index): #读一个文件的所有数据item
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        clip = False
        for words in fp:
            for word in words.split():
                if word.strip() in self.corpus.dictionary.word2idx:
                    if count > self.sen_len - 1:
                        clip = True  #截断 （本程序设定了句子长度sen_len为32）
                        break
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()]
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])
        return txt, label
    def __len__(self):
        return len(self.txt_filename)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import japanize_matplotlib
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.pardir)
from common.trainer import *
from common.shower import *
from common.model import CNN_v2

# 概要
# CNNをtuningして、より精度の高い画像認識モデルを実装。
# **tuningポイント**
#- Batch Normalization(正規化処理=なんらかの計算で、全てのデータを0～1の間の大きさにする)->過学習回避
#- Dropout(要素をランダムに選んで0にする)->過学習回避
#- 最適化関数として、Adamを採用。
#- modelの多層化


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
])

data_root = '../data/'
train_set = datasets.CIFAR10(root=data_root,train=True,transform=transform,download=True)
test_set = datasets.CIFAR10(root=data_root,train=False,transform=transform,download=True)

batch_size = 25
device = "mps" if torch.backends.mps.is_available() else "cpu"
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(device)
model = CNN_v2(len(classes)).to(device)
num_epochs = 10
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 学習結果
history = train(model,num_epochs,train_loader,test_loader,optimizer,device,criterion)

# 学習結果の表示
title = 'TuningしたCNNによる画像認識の結果'
show_history(history,title)


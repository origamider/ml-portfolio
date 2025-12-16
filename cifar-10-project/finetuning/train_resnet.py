import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import japanize_matplotlib
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import sys
import os
sys.path.append(os.pardir)
from common.model import *
from common.trainer import *
from common.shower import *

# 概要
# CNNをtuningして、より精度の高い画像認識モデルを実装。
# **tuningポイント**
#- Batch Normalization(正規化処理=なんらかの計算で、全てのデータを0～1の間の大きさにする)->過学習回避
#- Dropout(要素をランダムに選んで0にする)->過学習回避
#- 最適化関数として、Adamを採用。
#- modelの多層化


transform_train = transforms.Compose([
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(p=0.5), # ランダムに画像反転
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    transforms.RandomErasing(p=0.5,scale=(0.02,0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
transform_test = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
])

data_root = '../data/'
train_set = datasets.CIFAR10(root=data_root,train=True,transform=transform_train,download=True)
test_set = datasets.CIFAR10(root=data_root,train=False,transform=transform_test,download=True)

batch_size = 100
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = models.resnet18(pretrained=True)
fc_in_features = model.fc.in_features # 全結合層の入力次元
model.fc = nn.Linear(fc_in_features,len(classes))# デフォルトが1000種類の出力なので、10種類に限定
model = model.to(device)
num_epochs = 10
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
print(optimizer)

# 学習結果
history = train(model,num_epochs,train_loader,test_loader,optimizer,device,criterion)

# 学習結果の表示
title = 'ResNet-18による画像認識の結果'
show_history(history,title)


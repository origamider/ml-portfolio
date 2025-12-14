from sklearn.datasets import load_iris
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

# 概要
# CIFAR-10データセットを用いた、CNNによる画像認識モデルの実装。

# データ準備

# transform:正規化する。CNNなので1階テンソルに変形する必要がない
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
])

data_root = '../data/'
train_set = datasets.CIFAR10(root=data_root,train=True,transform=transform,download=True)
test_set = datasets.CIFAR10(root=data_root,train=False,transform=transform,download=True)

batch_size = 50
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self,n_output,n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3) # Conv2d(入力チャネル、出力チャネル、カーネルサイズ)
        self.conv2 = nn.Conv2d(32,32,3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2)) # フィルターサイズ(2,2)とする。
        self.flatten = nn.Flatten() # (100,32,14,14)->(100,6272)に変換。なお、100は画像の枚数ね。
        self.l1 = nn.Linear(6272,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_output)
        
        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool)
        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2)
    
    def forward(self,x):
        val = self.features(x)
        val = self.flatten(val)
        val = self.classifier(val)
        return val

n_input = 3*32*32
n_hidden = 128
n_output = len(classes)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
model = CNN(n_output,n_hidden).to(device)
num_epochs = 10
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr)
history = []

# # 学習するよ
for epoch in range(num_epochs):
    n_train_acc,n_eval_acc = 0,0
    train_loss,eval_loss = 0,0
    n_train,n_eval = 0,0
    
    model.train() # 訓練モード発動
    
    for inputs,labels in(train_loader):
        train_batch_size = len(labels)
        n_train += train_batch_size
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        predicted = torch.max(outputs,dim=1)[1]
        n_train_acc += (predicted==labels).sum().item()
        train_loss += loss.item() * train_batch_size
        
    model.eval() # 検証モード発動
    with torch.no_grad():
        for inputs_test,labels_test in(test_loader):
            test_batch_size = len(labels_test)
            n_eval += test_batch_size
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test,labels_test)
            predicted_test = torch.max(outputs_test,dim=1)[1]
            n_eval_acc += (predicted_test==labels_test).sum().item()
            eval_loss += loss.item() * test_batch_size
    
    # 精度計算
    train_acc = n_train_acc / n_train
    eval_acc = n_eval_acc / n_eval
    avg_train_loss = train_loss / n_train
    avg_eval_loss = eval_loss / n_eval
    print(f'Epoch:{epoch} avg_train_loss:{avg_train_loss} train_acc:{train_acc} avg_eval_loss:{avg_eval_loss} eval_acc:{eval_acc}')
    history.append([epoch,avg_train_loss,train_acc,avg_eval_loss,eval_acc])
    
# 学習曲線の表示
history = np.array(history)
fig,axes = plt.subplots(1,2)
plt.title('CIFAR-10データセットによる、CNNを用いた画像認識モデルの学習結果')
axes[0].set_title('学習曲線(損失)')
axes[0].plot(history[:,0],history[:,1],c='b',label='訓練用データ')
axes[0].plot(history[:,0],history[:,3],c='r',label='検証用データ')
axes[0].set_xlabel('繰り返し数')
axes[0].set_ylabel('損失')

axes[1].set_title('学習曲線(精度)')
axes[1].plot(history[:,0],history[:,2],c='b',label='訓練用データ')
axes[1].plot(history[:,0],history[:,4],c='r',label='検証用データ')
axes[1].set_xlabel('繰り返し数')
axes[1].set_ylabel('精度')

plt.show()


# イメージ理解

for inputs,labels in(test_loader):
    break;

inputs = inputs.to(device)
labels = labels.to(device)
outputs = model(inputs)
predicted = torch.max(outputs,dim=1)[1]

plt.figure()

for i in range(batch_size):
    ax = plt.subplot(5,10,i+1)
    pred = predicted[i]
    label = labels[i]
    input = inputs[i]
    if pred==label:
        color = 'blue'
        mark = 'O'
    else:
        color = 'red'
        mark = 'X'

    input = (input+1)/2 # [-1,1] -> [0,1]
    plt.imshow(input.to('cpu').permute(1,2,0).numpy(),cmap='gray')
    pred = classes[pred]
    label = classes[label]
    ax.set_title(f'{mark}\n予想:{pred}\n正解:{label}',c=color,fontsize=7)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout(h_pad=2.0)
plt.show()


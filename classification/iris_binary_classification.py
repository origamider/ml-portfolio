from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import japanize_matplotlib

# 概要
# バッチ学習。二値分類を実装。
# 与えられた二つの特徴量から、SetosaかVersicolorかを判定する簡易的なニューラルネットワーク。

X,y = load_iris(return_X_y=True)

# irisの中身
# 0〜49番目: Setosa
# 50〜99番目: Versicolor
# 100〜149番目: Virginica
# data -> 3つのあやめ(Setosa,Versicolour,Virginica)の4つの特徴量(sepal length,sepal width,petal length,petal width) sepal->ガク,petal->花弁
# target -> それぞれのdataに対する正解ラベル。(0->Setosa,1->Versicolor,2->Virginica)
# 今回はSetosaとVersicolorで2つの特徴量(sepal length,sepal width)で関係性を考える。

x_data = X[:100,:2]
y_data = y[:100]

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=70,test_size=30,random_state=123)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
x_0 = x_train[y_train==0]
x_1 = x_train[y_train==1]
# plt.scatter(x_0[:,0],x_0[:,1],marker='x',label="Setosa",color="b")
# plt.scatter(x_1[:,0],x_1[:,1],marker='o',label="Versicolor",color="k")
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

n_input = 2
n_output = 1

class Net(nn.Module):
    def __init__(self,n_input,n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        val = self.l1(x) # (70,2)->(70,1)に変換。
        val = self.sigmoid(val)
        return val

net = Net(n_input,n_output)
print(net)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(),lr=0.01)

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()
labels = labels.reshape(-1,1) #　criterionに渡す引数の型の関係上、(n,1)に変形する
labels_test = labels_test.reshape(-1,1) #　criterionに渡す引数の型の関係上、(n,1)に変形する
num_epochs = 10000

history = np.zeros((0,5))
# 学習
for epoch in range(num_epochs):
    optimizer.zero_grad() # 勾配を初期化
    outputs = net(inputs) # (100,2)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
        
    predicted = torch.where(outputs < 0.5,0,1)
    train_acc = (predicted == labels).sum() / len(y_train)
    
    # 検証用テスト
    outputs_test = net(inputs_test)
    test_loss = criterion(outputs_test,labels_test).item()
    predicted_test = torch.where(outputs_test < 0.5,0,1)
    test_acc = (predicted_test == labels_test).sum() / len(inputs_test)
    if epoch % 100 == 0:
        print(f'Epoch:{epoch},train_loss={train_loss:.5f},train_acc={train_acc:.5f},test_loss={test_loss:.5f},test_acc={test_acc:.5f}')
        message = np.array([epoch,train_loss,train_acc,test_loss,test_acc])
        history = np.vstack((history,message))
        
        
fig,axes = plt.subplots(1,2,figsize=(20,8))
fig.suptitle('あやめの学習結果')


# 学習曲線(損失)
axes[0].plot(history[:,0],history[:,1],label='訓練用')
axes[0].plot(history[:,0],history[:,3],label='テスト用')
axes[0].set_xlabel('繰り返し回数')
axes[0].set_ylabel('損失')
axes[0].set_title('学習曲線(損失)')
axes[0].legend()
# 学習曲線(精度)
axes[1].plot(history[:,0],history[:,2],label='訓練用')
axes[1].plot(history[:,0],history[:,4],label='テスト用')
axes[1].set_xlabel('繰り返し回数')
axes[1].set_ylabel('精度')
axes[1].set_title('学習曲線(精度)')
axes[1].legend()

plt.show()
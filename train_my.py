import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
# from torch.utils import data
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class mydataset(Data.Dataset):
    def __init__(self):   # self 参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        data_csv = pd.read_csv('./data.csv')
        features = data_csv.iloc[:, 1:-1].to_numpy()  # data
        labels = data_csv.iloc[:, -1].to_numpy()
        features = features.astype(np.float32)
        # labels = labels.astype(np.float32)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data = self.features[idx]
        return data

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net = nn.Sequential(nn.Linear(16, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),

                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),

                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 7)
                    )
net.apply(init_weights)


def train_nn2(net, num_epochs, train_iter, loss_function, optimizer,batch_size):

    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_iter):
            #         x = x.view(x.size(0), 28 * 28)

            out = net(x)
            # y_onehot =F.one_hot(y,num_classes=7).float()    # 转为one-hot编码

            loss = loss_function(out, y)

            optimizer.zero_grad()
            loss.sum().backward()

            optimizer.step()

            if batch_idx % 100 == 0:
                print(epoch, batch_idx*batch_size)
                print(loss.sum())


def show_result(net, test_iter,test_num):
    i = 0
    for batch_idx, (x, y) in enumerate(test_iter):
        out = net(x)
        index = torch.argmax(out, axis=1)
        # print(out)
        print("outY", index)
        i = i + 1
        print("true", y)
        p = accuracy(index,y)
        print('%d acc:%.2f'%(i,p))
        if (i > test_num):
            break


def accuracy(y_hat, y):
    acc = (y_hat == y).sum()
    return acc/len(y)

if __name__ == '__main__':
    batch_size = 100
    lr = 0.001
    num_epochs = 100
    
    data = mydataset()
    x = torch.tensor(data)         # 前五个数据
    y = torch.tensor(data.labels)  # 标签
    torch_dataset = Data.TensorDataset(x, y)
    train_iter = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,       # torch TensorDataset format
        batch_size=batch_size,                # mini batch size
        shuffle=True,                  # 要不要打乱数据 (打乱比较好)
        num_workers=0,                 # 多线程来读数据
    )
    test_iter = train_iter
  
    #batch_size = 20 num_epochs = 30 lr = 0.01     

    
    loss_function = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_nn2(net, num_epochs, train_iter, loss_function, optimizer,batch_size)
    show_result(net, test_iter,30)

    torch.save({'model': net.state_dict()}, 'classification.pth')

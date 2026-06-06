import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
import time
from torchsummary import summary




def create_dataset():
    # 1.加载csv文件数据集
    data = pd.read_csv('/data/Diabetes.csv')

    # print(f'data:{data.head()}')
    # print(f'data:{data.shape}')

    # 2.获取x特征列和y标签列
    x, y = data.iloc[:, :8].copy(), data.iloc[:, 8]
    y = LabelEncoder().fit_transform(y)

    # 2.1 处理异常值：将0替换为NaN，再用中位数填充
    # Glucose, BloodPressure, SkinThickness, Insulin, BMI 为0不合理
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    x[zero_cols] = x[zero_cols].replace(0, np.nan)
    x = x.fillna(x.median())

    # 特征缩放：标准化（均值0，方差1）—— 对神经网络收敛至关重要
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    #print(f'x:{x.head()},{x.shape}')
    #print(f'y:{y.head()},{y.shape}')


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test))
    print(train_dataset,test_dataset)


    return train_dataset, test_dataset, x_test.shape[1], len(np.unique(y))



class ModelDemo(nn.Module):
    # 1.在init魔法方法中，完成初始化：父类成员， 及神经网络搭建
    def __init__(self, intput_dim, output_dim):
        # 1.1初始化父类成员
        super().__init__()
        # 1.2搭建神经网络：隐藏层，输入层
        self.linear1 = nn.Linear(intput_dim, 32)      #隐藏层1
        self.bn1 = nn.BatchNorm1d(32)                  #批归一化
        self.linear2 = nn.Linear(32, 16)               #隐藏层2
        self.bn2 = nn.BatchNorm1d(16)                  #批归一化
        self.output = nn.Linear(16, 1)                 #二分类：1个输出


    # 2.前向传播
    def forward(self, x):
        # 1.1第一层隐藏层计算
        # x = self.linear1(x)
        # x = torch.sigmoid(x)

        x = torch.relu(self.bn1(self.linear1(x)))

        # 1.2第2层隐藏层计算
        x = torch.relu(self.bn2(self.linear2(x)))

        # 1.3输出层计算（BCEWithLogitsLoss自带sigmoid，输出层不加激活）
        x = self.output(x)

        # 1.4返回预测值
        return x



def train(train_dataset, intput_dim, output_dim):
    # 1.创建模型对象
    model = ModelDemo(intput_dim, output_dim)

    # 2.定义损失函数和优化器（二分类用BCEWithLogitsLoss）
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 每30轮学习率减半

    # 3.创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 4.训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        batch_num = 0
        total_loss = 0
        start = time.time()
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()  # 清零梯度
            outputs = model(x.float()).squeeze(1)  # 前向传播，去掉最后一维
            loss = loss_fn(outputs, y.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
            batch_num += 1
        scheduler.step()
        print(f'Epoch [{epoch+1}], Loss: {total_loss/batch_num:.4f}, time: {time.time() - start:.2f} seconds')



    torch.save(model.state_dict(), '/model/model.pth')


def evaluate(test_dataset, intput_dim, output_dim):
    # 1.创建模型对象
    model = ModelDemo(intput_dim, output_dim)

    # 2.加载训练好的模型参数
    model.load_state_dict(torch.load('/model/model.pth'))

    # 3.创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4.评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x.float()).squeeze(1)
            predicted = (torch.sigmoid(outputs) > 0.5).long()  # sigmoid + 阈值0.5
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    # create_dataset()
    train_dataset, test_dataset, intput_dim, output_dim = create_dataset()


    model = ModelDemo(intput_dim, output_dim)

    summary(model, input_size=(intput_dim,))

    train(train_dataset, intput_dim, output_dim)

    evaluate(test_dataset, intput_dim, output_dim)
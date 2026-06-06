class ImageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2, 2,0)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(3, 2, 0)

        # 第一个隐藏层
        self.linear1 = nn.Linear(576, 120)
        # 第二个隐藏层
        self.linear1 = nn.Linear(120, 84)
        # 第三个隐藏层
        self.linear1 = nn.Linear(84, 10)


        # 前向传播
    def forward(self, x):
        # 第1层：卷积层+激活函数+池化
        # x = self.conv1(x)
        # x = torch.relu(x)
        # x = self.pool1(x)
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第2层
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.reshape(x.size(0), -1)

        # 第3层全连接层+激活函数
        x = torch.relu(self.linear1(x))
        # 第4层全连接层+激活函数
        x = torch.relu(self.linear2(x))

        # 第5层输出
        return self.output(x)


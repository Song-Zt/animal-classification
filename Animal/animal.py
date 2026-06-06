"""
动物图片分类 - 卷积神经网络 (CNN)
数据集: data/raw-img/ 下10个动物文件夹
类别: butterfly, cat, chicken, cow, dog, elefante, horse, sheep, spider, squirrel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import time
from torchsummary import summary


# ==================== 超参数 ====================
BATCH_SIZE = 64       # 每批次样本数（显存不够就改小）
IMG_SIZE = 64         # 统一缩放到 64×64
NUM_EPOCHS = 10       # 训练轮数
LEARNING_RATE = 0.001 # 学习率
DATA_ROOT = 'D:/python/lianxi/data/raw-img'  # 数据集路径


# ==================== 1.自定义数据集类 ====================
class AnimalDataset(Dataset):
    """
    自定义数据集：遍历每个类别文件夹，加载所有图片及其标签
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []   # 图片路径列表
        self.labels = []   # 对应标签列表

        # 获取所有类别文件夹名（按字母排序保证标签一致性）
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 遍历每个类别文件夹，收集图片路径和标签
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                # 只处理常见图片格式
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(fpath)
                    self.labels.append(self.class_to_idx[cls_name])

        print(f'共加载 {len(self.images)} 张图片, {len(self.classes)} 个类别')
        print(f'类别映射: {self.class_to_idx}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图片并转为RGB（避免RGBA等格式问题）
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ==================== 2.卷积神经网络模型 ====================
class AnimalCNN(nn.Module):
    """
    CNN结构:
    输入(3×64×64) → Conv1 → ReLU → Pool → Conv2 → ReLU → Pool → Conv3 → ReLU → Pool
    → 展平 → FC1 → ReLU → FC2 → ReLU → FC3(输出10类)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # ---- 卷积部分 ----
        # 卷积层1: 输入3通道(RGB), 输出32通道, 3×3卷积核, padding=1保持尺寸
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 池化层1: 2×2窗口, 步长2, 图片尺寸减半 64→32
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2: 输入32通道, 输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层2: 图片尺寸再减半 32→16
        self.pool2 = nn.MaxPool2d(2, 2)

        # 卷积层3: 输入64通道, 输出128通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 池化层3: 图片尺寸再减半 16→8
        self.pool3 = nn.MaxPool2d(2, 2)

        # ---- 全连接部分 ----
        # 卷积输出展平后: 128通道 × 8×8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)   # 隐藏层1
        self.fc2 = nn.Linear(256, 128)             # 隐藏层2
        self.fc3 = nn.Linear(128, num_classes)     # 输出层

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层1: Conv → ReLU → Pool  (3×64×64 → 32×32×32)
        x = self.pool1(torch.relu(self.conv1(x)))
        # 卷积层2: Conv → ReLU → Pool  (32×32×32 → 64×16×16)
        x = self.pool2(torch.relu(self.conv2(x)))
        # 卷积层3: Conv → ReLU → Pool  (64×16×16 → 128×8×8)
        x = self.pool3(torch.relu(self.conv3(x)))

        # 展平: (batch, 128, 8, 8) → (batch, 8192)
        x = x.view(x.size(0), -1)

        # 全连接层 + Dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))

        # 输出层（CrossEntropyLoss自带softmax，这里输出原始分数）
        x = self.fc3(x)
        return x


# ==================== 3.训练函数 ====================
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 训练模式（启用Dropout）
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 清零梯度
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, train_acc


# ==================== 4.评估函数 ====================
def evaluate(model, test_loader, device):
    model.eval()  # 评估模式（关闭Dropout）
    correct = 0
    total = 0
    with torch.no_grad():  # 评估不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ==================== 5.主程序 ====================
if __name__ == '__main__':
    # ----- 5.1 数据预处理 -----
    # 训练集：随机翻转 + 随机旋转 + 标准化（数据增强，防止过拟合）
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),     # 缩放到统一尺寸
        transforms.RandomHorizontalFlip(),           # 随机水平翻转
        transforms.RandomRotation(15),               # 随机旋转±15度
        transforms.ToTensor(),                       # 转为张量 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet标准化
                             std=[0.229, 0.224, 0.225])
    ])
    # 测试集：只缩放 + 标准化（不做数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ----- 5.2 加载数据集 -----
    print('=' * 50)
    print('加载数据集...')
    train_dataset_full = AnimalDataset(DATA_ROOT, transform=train_transform)
    test_dataset_full = AnimalDataset(DATA_ROOT, transform=test_transform)

    # 按 80% 训练 / 20% 测试 划分
    total_size = len(train_dataset_full)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # 固定种子保证可复现
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_dataset_full, [train_size, test_size], generator=generator)
    _, test_dataset = random_split(test_dataset_full, [train_size, test_size], generator=generator)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'训练集: {train_size} 张, 测试集: {test_size} 张')
    print('=' * 50)

    # ----- 5.3 创建模型 -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    num_classes = len(train_dataset_full.classes)
    model = AnimalCNN(num_classes=num_classes).to(device)

    # 打印模型结构
    summary(model, input_size=(3, IMG_SIZE, IMG_SIZE))

    # ----- 5.4 训练模型 -----
    criterion = nn.CrossEntropyLoss()  # 分类损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5轮学习率减半

    print('=' * 50)
    print('开始训练...')
    print('=' * 50)
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - start
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]  '
              f'Loss: {train_loss:.4f}  '
              f'Train Acc: {train_acc:.2f}%  '
              f'Test Acc: {test_acc:.2f}%  '
              f'Time: {elapsed:.1f}s')

    # ----- 5.5 保存模型 -----
    torch.save(model.state_dict(), 'D:/python/lianxi/model/animal_model.pth')
    print('=' * 50)
    print(f'最终测试准确率: {test_acc:.2f}%')
    print('模型已保存到 model/animal_model.pth')

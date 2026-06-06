"""
动物图片分类 - 改进版（迁移学习 + BatchNorm + 更强数据增强）
数据集: data/raw-img/ 下10个动物文件夹
类别: butterfly, cat, chicken, cow, dog, elefante, horse, sheep, spider, squirrel

改进点:
  1. 修复测试集 transform bug（原来测试集也用了数据增强）
  2. 使用预训练 ResNet18 做迁移学习（替换最后的全连接层）
  3. BatchNorm 加速收敛
  4. 图片尺寸 64→128，保留更多细节
  5. 更强的数据增强（ColorJitter, RandomErasing）
  6. 训练轮数增加到 25 轮
  7. 冻结前面层的参数，只训练最后几层（fine-tune）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
import time

# 设置预训练权重下载目录为当前目录下的 model 文件夹
WEIGHTS_DIR = 'D:/python/lianxi/model'
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = WEIGHTS_DIR


# ==================== 超参数 ====================
BATCH_SIZE = 32        # ResNet 更大，batch 改小一些避免显存不足
IMG_SIZE = 128         # 改为 128×128，保留更多图片细节
NUM_EPOCHS = 5        # 训练轮数增加到 25
LEARNING_RATE = 0.001  # 学习率
DATA_ROOT = 'D:/python/lianxi/data/raw-img'


# ==================== 1.自定义数据集类 ====================
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(fpath)
                    self.labels.append(self.class_to_idx[cls_name])

        print(f'共加载 {len(self.images)} 张图片, {len(self.classes)} 个类别')
        print(f'类别映射: {self.class_to_idx}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ==================== 2.构建迁移学习模型 ====================
def build_model(num_classes, freeze_backbone=True):
    """
    使用预训练的 ResNet18，替换最后的全连接层
    Args:
        num_classes: 输出类别数
        freeze_backbone: 是否冻结前面的卷积层（只训练最后的全连接层）
    """
    # 加载预训练的 ResNet18（自动下载权重）
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 冻结所有层的参数（不参与训练）
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 替换最后的全连接层（原: 512→1000，改为: 512→num_classes）
    # 新建的层默认 requires_grad=True，会参与训练
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ==================== 3.训练函数 ====================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, train_acc


# ==================== 4.评估函数 ====================
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
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
    # 训练集：更强的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),           # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.2),        # 随机垂直翻转（20%概率）
        transforms.RandomRotation(30),               # 随机旋转±30度
        transforms.ColorJitter(                      # 颜色抖动
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),             # 随机擦除
    ])
    # 测试集：只缩放 + 标准化，不做增强
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ----- 5.2 加载数据集 -----
    print('=' * 60)
    print('加载数据集...')

    # 修复bug：分别创建两个 Dataset，各自用不同的 transform
    train_dataset_full = AnimalDataset(DATA_ROOT, transform=train_transform)
    test_dataset_full = AnimalDataset(DATA_ROOT, transform=test_transform)

    # 按 80% 训练 / 20% 测试 划分
    total_size = len(train_dataset_full)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_dataset_full, [train_size, test_size], generator=generator)
    _, test_dataset = random_split(test_dataset_full, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'训练集: {train_size} 张, 测试集: {test_size} 张')
    print('=' * 60)

    # ----- 5.3 创建模型 -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    num_classes = len(train_dataset_full.classes)
    model = build_model(num_classes, freeze_backbone=True).to(device)

    # 打印哪些参数需要训练
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数: {trainable_params:,} 可训练 / {total_params:,} 总计')
    print(f'冻结 ResNet 前面的层，只训练最后的全连接层')
    print('=' * 60)

    # ----- 5.4 训练模型 -----
    criterion = nn.CrossEntropyLoss()
    # 只优化需要训练的参数（被冻结的层不会更新）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    # CosineAnnealing 学习率调度，比 StepLR 更平滑
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print('开始训练（迁移学习模式）...')
    print('=' * 60)
    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - start

        # 记录最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'D:/python/lianxi/model/animal_model_best.pth')

        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  '
              f'Loss: {train_loss:.4f}  '
              f'Train Acc: {train_acc:.2f}%  '
              f'Test Acc: {test_acc:.2f}%  '
              f'Time: {elapsed:.1f}s')

    # ----- 5.5 保存模型 -----
    torch.save(model.state_dict(), 'D:/python/lianxi/model/animal_model.pth')
    print('=' * 60)
    print(f'最终测试准确率: {test_acc:.2f}%')
    print(f'最佳测试准确率: {best_acc:.2f}%')
    print('模型已保存到 model/animal_model.pth')
    print('最佳模型已保存到 model/animal_model_best.pth')

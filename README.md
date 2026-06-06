# 🐾 动物图片分类 - CNN

使用卷积神经网络 (CNN) 对 10 种动物图片进行分类。

## 📋 类别

| 类别 | 英文名 |
|------|--------|
| 蝴蝶 | butterfly |
| 猫 | cat |
| 鸡 | chicken |
| 牛 | cow |
| 狗 | dog |
| 大象 | elefante |
| 马 | horse |
| 羊 | sheep |
| 蜘蛛 | spider |
| 松鼠 | squirrel |

## 📁 项目结构

```
Animal/
├── animal.py              # 基础版：简单 CNN（3层卷积）
└── animal_improved.py     # 改进版：ResNet18 迁移学习
```

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install torch torchvision torchsummary pillow
```

### 2. 准备数据集

从 [Kaggle - Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) 下载数据集，解压后目录结构如下：

```
data/
└── raw-img/
    ├── butterfly/
    ├── cat/
    ├── chicken/
    ├── cow/
    ├── dog/
    ├── elefante/
    ├── horse/
    ├── sheep/
    ├── spider/
    └── squirrel/
```

### 3. 修改路径

打开 `animal.py` 或 `animal_improved.py`，将路径改为你本地的实际路径：

```python
DATA_ROOT = 'Your_Path/data/raw-img'   # 数据集路径
WEIGHTS_DIR = 'Your_Path/model'        # 权重保存路径（仅改进版）
```

### 4. 运行训练

```bash
# 基础版
python Animal/animal.py

# 改进版（迁移学习，准确率更高）
python Animal/animal_improved.py
```

## 📊 两个版本对比

| | 基础版 `animal.py` | 改进版 `animal_improved.py` |
|---|---|---|
| 模型 | 3层简单CNN | 预训练 ResNet18 |
| 图片尺寸 | 64×64 | 128×128 |
| 训练轮数 | 25 | 25 |
| BatchNorm | ❌ | ✅（ResNet自带）|
| 数据增强 | 翻转+旋转 | 翻转+旋转+颜色抖动+擦除 |
| 预计准确率 | ~80% | ~90%+ |

## 💡 进阶优化

- 提高图片尺寸到 `224×224`（ResNet 原始输入尺寸）
- 训练几轮后解冻所有层，用更小学习率微调
- 使用更大的模型如 ResNet34/50
- 增加更多数据增强策略

## 📄 License

MIT

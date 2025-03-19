# TransUNet医学图像分割项目

这是一个基于TransUNet的医学图像分割项目，用于二分类分割任务（背景为0，前景为255）。

## 项目结构

```
transunet_project/
├── config.py             # 配置文件
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── requirements.txt      # 项目依赖
├── models/               # 模型定义
│   ├── __init__.py
│   ├── transformer.py    # Transformer组件
│   └── transunet.py      # TransUNet模型
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── dataset.py        # 数据集加载
│   └── losses.py         # 损失函数
├── training/             # 训练数据
│   ├── images/           # 训练图像
│   └── segmentations/    # 训练分割标注
└── validation/           # 验证数据
    ├── images/           # 验证图像
    └── segmentations/    # 验证分割标注
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据准备

请将您的训练和验证数据组织为以下结构：

- `training/images/`: 训练集PNG图像
- `training/segmentations/`: 训练集分割掩码（二值图像，0为背景，255为前景）
- `validation/images/`: 验证集PNG图像
- `validation/segmentations/`: 验证集分割掩码

## 训练模型

```bash
python train.py
```

训练完成后，最佳模型将保存在`checkpoints/best_model.pth`。

## 预测

使用训练好的模型对新图像进行预测：

```bash
python predict.py --image path/to/image.png --output prediction.png
```

参数说明：
- `--image`: 输入图像路径（必需）
- `--output`: 输出预测掩码路径（可选，默认为`prediction.png`）
- `--checkpoint`: 模型权重路径（可选，默认为`checkpoints/best_model.pth`）
- `--threshold`: 分割阈值（可选，默认为0.5）

## 模型说明

本项目实现了TransUNet模型，该模型结合了Transformer和U-Net的优点：

1. 使用CNN提取多尺度特征
2. 利用Transformer捕获全局依赖关系
3. 通过跳跃连接和上采样恢复空间细节

适用于需要同时关注局部细节和全局上下文的医学图像分割任务。 
# Segformer 语义分割项目

这是一个基于Segformer架构的语义分割项目，用于二分类语义分割任务。

## 项目结构

```
segformer/
│
├── configs/           # 配置文件目录
│   └── config.yaml    # 主配置文件
│
├── data/              # 数据集目录
│   ├── training/      # 训练集
│   │   ├── images/    # 训练图像
│   │   └── segmentations/ # 训练标签
│   │
│   └── validation/    # 验证集
│       ├── images/    # 验证图像
│       └── segmentations/ # 验证标签
│
├── src/               # 源代码目录
│   ├── dataset.py     # 数据集处理
│   ├── model.py       # 模型定义
│   └── losses.py      # 损失函数
│
├── utils/             # 工具函数目录
│   ├── metrics.py     # 评估指标
│   ├── optimizer.py   # 优化器和学习率调度器
│   └── utils.py       # 通用工具函数
│
├── scripts/           # 脚本目录
│
├── checkpoints/       # 模型检查点保存目录(训练时自动创建)
│
├── requirements.txt   # 项目依赖
├── train.py           # 训练脚本
├── predict.py         # 预测脚本
└── README.md          # 项目说明
```

## 环境设置

1. 创建虚拟环境：

```bash
# 使用conda创建虚拟环境
conda create -n seg python=3.9
conda activate seg

# 或使用venv创建虚拟环境
python -m venv seg
# 在Windows上
seg\Scripts\activate
# 在Linux/Mac上
source seg/bin/activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python train.py --config configs/config.yaml
```

参数说明：
- `--config`: 配置文件路径，默认为 `configs/config.yaml`
- `--resume`: 恢复训练的检查点路径，默认为 `None`
- `--no-wandb`: 禁用Weights & Biases日志记录
- `--seed`: 随机种子，默认为 `42`

### 预测

```bash
python predict.py --checkpoint checkpoints/best_model.pth --input-dir test_images --output-dir predictions
```

参数说明：
- `--config`: 配置文件路径，默认为 `configs/config.yaml`
- `--checkpoint`: 模型检查点路径，必需
- `--input-dir`: 输入图像目录，必需
- `--output-dir`: 输出目录，默认为 `predictions`
- `--device`: 使用的设备 (cuda/cpu)，默认为 `cuda`
- `--visual`: 可视化预测结果，添加此选项将保存可视化结果

## 配置说明

配置文件 `configs/config.yaml` 包含模型、数据、训练、验证等配置项：

- 模型配置：选择Segformer骨干网络、类别数等
- 数据配置：数据路径、图像大小、批量大小等
- 训练配置：轮次数、学习率、优化器等
- 验证配置：验证间隔、评估指标等

## 数据集格式

数据集应组织为以下结构：

```
data/
├── training/
│   ├── images/      # 原始训练图像（PNG格式）
│   └── segmentations/ # 训练标签（二值PNG图像，值为0和255）
│
└── validation/
    ├── images/      # 原始验证图像
    └── segmentations/ # 验证标签
```

## 注意事项

1. 标签图像应为二值图像，值为0（背景）和255（前景）
2. 训练前请检查配置文件参数是否符合需求
3. 如果显存不足，可以调整批量大小和模型大小
4. 默认使用GPU训练，如果没有GPU，请在配置文件中将设备改为`cpu` 
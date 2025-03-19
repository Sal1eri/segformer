import os
import torch
# 数据集路径 - 调整为新的目录结构
TRAIN_IMAGE_DIR = 'data/training/images'
TRAIN_MASK_DIR = 'data/training/segmentations'
VAL_IMAGE_DIR = 'data/validation/images'
VAL_MASK_DIR = 'data/validation/segmentations'

# 训练参数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用 'cpu' 如果没有GPU
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
IMAGE_SIZE = 256  # 输入图像尺寸
PATCH_SIZE = 16   # Transformer的patch大小
NUM_CLASSES = 1   # 二分类任务
HIDDEN_SIZE = 768  # Transformer隐层大小
NUM_LAYERS = 12    # Transformer层数
NUM_HEADS = 12     # 注意力头数
MLP_RATIO = 4      # MLP比例

# 随机种子
SEED = 42

# 检查点保存路径
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
import os
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_config(config_path):
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed=42):
    """
    设置随机种子以确保结果可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, config, best_score, metrics, save_path):
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        config: 配置字典
        best_score: 最好的分数
        metrics: 评估指标
        save_path: 保存路径
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'config': config,
        'best_score': best_score,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"检查点已保存到 {save_path}")


def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """
    加载检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        path: 检查点路径
    
    Returns:
        dict: 检查点字典
    """
    if not os.path.exists(path):
        print(f"检查点不存在: {path}")
        return None
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"从 {path} 加载检查点，轮次: {checkpoint['epoch']}")
    return checkpoint


def visualize_predictions(images, masks, predictions, save_dir=None, max_samples=4):
    """
    可视化预测结果
    
    Args:
        images: 图像张量，形状为[B, C, H, W]
        masks: 标签张量，形状为[B, H, W]
        predictions: 预测张量，形状为[B, H, W]
        save_dir: 保存目录
        max_samples: 最大样本数
    """
    batch_size = images.shape[0]
    samples = min(batch_size, max_samples)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, axes = plt.subplots(samples, 3, figsize=(15, 5 * samples))
    
    for i in range(samples):
        # 处理图像
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        
        # 处理掩码和预测
        mask = masks[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()
        
        if samples == 1:
            axes[0].imshow(img)
            axes[0].set_title('原图')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('真实标签')
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('预测结果')
            axes[2].axis('off')
        else:
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('原图')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('真实标签')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('预测结果')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'predictions.png'))
        plt.close()
    else:
        plt.show() 
import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.dataset import create_dataloaders
from src.model import create_model
from src.losses import create_loss_fn
from utils.optimizer import create_optimizer, create_scheduler
from utils.metrics import calculate_metrics
from utils.utils import load_config, set_seed, save_checkpoint, visualize_predictions

# 禁用wandb
WANDB_AVAILABLE = False
# try:
#     import wandb
#     WANDB_AVAILABLE = True
# except ImportError:
#     WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Segformer训练脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--no-wandb', action='store_true', help='禁用Weights & Biases日志记录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, config, scaler):
    model.train()
    epoch_loss = 0
    epoch_metrics = {metric: 0 for metric in config['validation']['metrics']}
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{config['training']['epochs']}")
    
    # 获取梯度累积步数
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 使用混合精度训练
        with autocast():
            # 前向传播
            outputs = model(images)
            logits = outputs['logits']
            
            # 计算损失
            loss = criterion(logits, masks)
            # 考虑梯度累积
            loss = loss / gradient_accumulation_steps
        
        # 反向传播和优化
        if config['training'].get('mixed_precision', False):
            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 更新学习率
        if scheduler is not None and (batch_idx + 1) % gradient_accumulation_steps == 0:
            scheduler.step()
        
        # 计算指标
        preds = outputs['preds']
        batch_metrics = calculate_metrics(preds, masks, config['validation']['metrics'])
        
        # 更新进度条
        epoch_loss += loss.item() * gradient_accumulation_steps
        for metric, value in batch_metrics.items():
            epoch_metrics[metric] += value
        
        progress_bar.set_postfix({
            'loss': epoch_loss / (batch_idx + 1),
            **{metric: epoch_metrics[metric] / (batch_idx + 1) for metric in epoch_metrics}
        })
        
        # 记录日志
        # if WANDB_AVAILABLE and not args.no_wandb:
        #     wandb.log({
        #         'train/loss': loss.item(),
        #         'train/lr': optimizer.param_groups[0]['lr'],
        #         **{f'train/{metric}': value for metric, value in batch_metrics.items()}
        #     })
        
        # 可视化
        if (batch_idx + 1) % config['training']['log_interval'] == 0:
            visualize_dir = os.path.join(config['training']['save_dir'], 'visualizations', f'epoch_{epoch+1}')
            os.makedirs(visualize_dir, exist_ok=True)
            visualize_predictions(
                images, masks, preds,
                save_dir=visualize_dir,
                max_samples=min(4, images.size(0))
            )
    
    # 计算平均指标
    epoch_loss /= num_batches
    for metric in epoch_metrics:
        epoch_metrics[metric] /= num_batches
    
    return epoch_loss, epoch_metrics


def validate(model, val_loader, criterion, device, config):
    model.eval()
    val_loss = 0
    val_metrics = {metric: 0 for metric in config['validation']['metrics']}
    num_batches = len(val_loader)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            logits = outputs['logits']
            
            # 计算损失
            loss = criterion(logits, masks)
            
            # 计算指标
            preds = outputs['preds']
            batch_metrics = calculate_metrics(preds, masks, config['validation']['metrics'])
            
            # 更新指标
            val_loss += loss.item()
            for metric, value in batch_metrics.items():
                val_metrics[metric] += value
            
            progress_bar.set_postfix({
                'val_loss': val_loss / (batch_idx + 1),
                **{f'val_{metric}': val_metrics[metric] / (batch_idx + 1) for metric in val_metrics}
            })
            
            # 记录日志
            # if WANDB_AVAILABLE and not args.no_wandb and batch_idx == num_batches - 1:
            #     wandb.log({
            #         'val/loss': val_loss / (batch_idx + 1),
            #         **{f'val/{metric}': val_metrics[metric] / (batch_idx + 1) for metric in val_metrics}
            #     })
        
        # 可视化验证集样本
        visualize_dir = os.path.join(config['training']['save_dir'], 'visualizations', 'validation')
        os.makedirs(visualize_dir, exist_ok=True)
        visualize_predictions(
            images, masks, preds,
            save_dir=visualize_dir,
            max_samples=min(4, images.size(0))
        )
    
    # 计算平均指标
    val_loss /= num_batches
    for metric in val_metrics:
        val_metrics[metric] /= num_batches
    
    return val_loss, val_metrics


def main(args):
    # 强制禁用wandb
    args.no_wandb = True
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化wandb
    # if WANDB_AVAILABLE and not args.no_wandb:
    #     wandb.init(
    #         project="segformer-segmentation",
    #         config=config,
    #         name=f"segformer-{config['model']['backbone']}-{time.strftime('%Y%m%d-%H%M%S')}"
    #     )
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    print(f"训练集大小: {len(train_loader.dataset)}，验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    model = create_model(config)
    model.to(device)
    
    # 创建损失函数
    criterion = create_loss_fn(config)
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建学习率调度器
    num_steps = len(train_loader) * config['training']['epochs']
    scheduler = create_scheduler(optimizer, config, num_steps)
    
    # 初始化混合精度训练的scaler
    scaler = GradScaler() if config['training'].get('mixed_precision', False) else None
    
    # 初始化训练状态
    start_epoch = 0
    best_score = 0
    
    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print(f"恢复训练，从轮次 {start_epoch} 开始，最佳分数: {best_score:.4f}")
    
    # 训练循环
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n轮次 {epoch+1}/{config['training']['epochs']}")
        
        # 训练一个轮次
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, config, scaler
        )
        
        print(f"训练 - 损失: {train_loss:.4f}, 指标: {train_metrics}")
        
        # 验证
        if (epoch + 1) % config['validation']['interval'] == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, config)
            print(f"验证 - 损失: {val_loss:.4f}, 指标: {val_metrics}")
            
            # 记录日志
            # if WANDB_AVAILABLE and not args.no_wandb:
            #     wandb.log({
            #         'epoch': epoch + 1,
            #         'train/epoch_loss': train_loss,
            #         'val/epoch_loss': val_loss,
            #         **{f'train/epoch_{metric}': value for metric, value in train_metrics.items()},
            #         **{f'val/epoch_{metric}': value for metric, value in val_metrics.items()}
            #     })
            
            # 保存检查点
            if (epoch + 1) % config['training']['save_interval'] == 0:
                save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    config, best_score, val_metrics, save_path,
                    scaler=scaler
                )
            
            # 保存最佳模型
            current_score = val_metrics['iou']  # 使用IoU作为评估指标
            if current_score > best_score:
                best_score = current_score
                best_path = os.path.join(save_dir, "best_model.pth")
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    config, best_score, val_metrics, best_path,
                    scaler=scaler
                )
                print(f"保存最佳模型，分数: {best_score:.4f}")
    
    print(f"训练完成！最佳分数: {best_score:.4f}")
    
    # 结束wandb
    # if WANDB_AVAILABLE and not args.no_wandb:
    #     wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args) 
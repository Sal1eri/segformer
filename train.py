import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.transunet import TransUNet
from utils.dataset import get_data_loaders
from utils.losses import BCEDiceLoss
import config

# 添加评价指标计算函数
def calculate_iou(pred, target):
    """计算IoU (Intersection over Union)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    # 防止分母为0
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target):
    """计算Dice系数"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def calculate_pixel_accuracy(pred, target):
    """计算像素准确率"""
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / (target.shape[0] * target.shape[1] * target.shape[2] * target.shape[3])

def train_model():
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 初始化模型
    model = TransUNet(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=3,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.HIDDEN_SIZE,
        depth=config.NUM_LAYERS,
        n_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO
    )
    
    # 将模型移至设备
    device = torch.device(config.DEVICE)
    model = model.to(device)
    
    # 初始化损失函数和优化器
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    # 创建记录评价指标的列表
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_dices = []
    val_dices = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        train_acc = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算评价指标
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                batch_iou = calculate_iou(probs, masks)
                batch_dice = calculate_dice(probs, masks)
                batch_acc = calculate_pixel_accuracy(probs, masks)
                
                train_iou += batch_iou.item()
                train_dice += batch_dice.item()
                train_acc += batch_acc.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}', iou=f'{batch_iou.item():.4f}', dice=f'{batch_dice.item():.4f}')
        
        # 计算平均评价指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # 记录评价指标
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_dices.append(avg_train_dice)
        train_accs.append(avg_train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_acc = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 计算评价指标
                probs = torch.sigmoid(outputs)
                batch_iou = calculate_iou(probs, masks)
                batch_dice = calculate_dice(probs, masks)
                batch_acc = calculate_pixel_accuracy(probs, masks)
                
                val_iou += batch_iou.item()
                val_dice += batch_dice.item()
                val_acc += batch_acc.item()
                
                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.4f}', iou=f'{batch_iou.item():.4f}', dice=f'{batch_dice.item():.4f}')
        
        # 计算平均评价指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # 记录评价指标
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_dices.append(avg_val_dice)
        val_accs.append(avg_val_acc)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f'模型已保存，验证损失: {best_val_loss:.4f}')
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}, Dice: {avg_train_dice:.4f}, Acc: {avg_train_acc:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}, Acc: {avg_val_acc:.4f}')
    
    # 创建保存图表的目录
    os.makedirs('plots', exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # IoU曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU Curves')
    plt.legend()
    plt.grid(True)
    
    # Dice系数曲线
    plt.subplot(2, 2, 3)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Curves')
    plt.legend()
    plt.grid(True)
    
    # 像素准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/metrics.png')
    print(f'指标曲线图已保存至 plots/metrics.png')
    
    # 加载最佳模型进行测试
    evaluate_model()

def evaluate_model():
    # 获取测试数据加载器
    _, _, test_loader = get_data_loaders()
    
    # 初始化模型
    model = TransUNet(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=3,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.HIDDEN_SIZE,
        depth=config.NUM_LAYERS,
        n_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO
    )
    
    # 加载最佳模型
    device = torch.device(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')))
    model = model.to(device)
    model.eval()
    
    # 初始化指标
    criterion = BCEDiceLoss()
    test_loss = 0
    test_iou = 0
    test_dice = 0
    test_acc = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='测试中'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            # 计算评价指标
            probs = torch.sigmoid(outputs)
            test_iou += calculate_iou(probs, masks).item()
            test_dice += calculate_dice(probs, masks).item()
            test_acc += calculate_pixel_accuracy(probs, masks).item()
            
            # 二值化预测结果
            preds = (probs > 0.5).float().cpu().numpy()
            masks = masks.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(masks.flatten())
    
    # 计算平均指标
    avg_test_loss = test_loss / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    
    # 计算基于像素的指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # 打印结果
    print("\n====== 测试集评估结果 ======")
    print(f'测试损失: {avg_test_loss:.4f}')
    print(f'IoU: {avg_test_iou:.4f}')
    print(f'Dice系数: {avg_test_dice:.4f}')
    print(f'像素准确率: {avg_test_acc:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')

if __name__ == "__main__":
    # 创建检查点目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 训练模型
    train_model() 
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
    train_losses = []
    val_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f'模型已保存，验证损失: {best_val_loss:.4f}')
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.savefig('loss_curve.png')
    
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
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='测试中'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            # 二值化预测结果
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            masks = masks.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(masks.flatten())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 打印结果
    print(f'测试损失: {avg_test_loss:.4f}')
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')

if __name__ == "__main__":
    # 创建检查点目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 训练模型
    train_model() 
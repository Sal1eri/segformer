import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random
import config

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        
        # 获取所有图像和掩码文件名
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 掩码预处理
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 加载图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # 数据增强（仅用于训练集）
        if self.is_train and random.random() > 0.5:
            # 随机水平翻转
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
                
            # 随机垂直翻转
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # 随机旋转
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
        
        # 应用变换
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # 将掩码二值化: 0 -> 0, 255 -> 1
        mask = (mask > 0.5).float()
        
        return image, mask

def get_data_loaders():
    # 创建训练集和验证集数据集
    train_dataset = SegmentationDataset(
        config.TRAIN_IMAGE_DIR, 
        config.TRAIN_MASK_DIR, 
        config.IMAGE_SIZE, 
        is_train=True
    )
    
    val_dataset = SegmentationDataset(
        config.VAL_IMAGE_DIR,
        config.VAL_MASK_DIR,
        config.IMAGE_SIZE,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 由于测试时使用验证集评估，返回val_loader两次
    return train_loader, val_loader, val_loader 
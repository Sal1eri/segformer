import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, img_size=(512, 512), transform=None, is_train=True):
        """
        语义分割数据集
        
        Args:
            root_dir (str): 数据集根目录，包含images和segmentations子目录
            img_size (tuple): 图像大小
            transform (callable, optional): 数据增强转换
            is_train (bool): 是否是训练集
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.transform = transform
        
        # 获取图像和标签的路径
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'segmentations')
        
        self.image_files = sorted(os.listdir(self.images_dir))
        self.mask_files = sorted(os.listdir(self.masks_dir))
        
        # 检查文件对应关系
        assert len(self.image_files) == len(self.mask_files), "图像和掩码文件数量不匹配"
        
        # 检查文件名匹配
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_name = os.path.splitext(img_file)[0]
            mask_name = os.path.splitext(mask_file)[0]
            assert img_name == mask_name, f"图像和掩码文件名不匹配: {img_name} vs {mask_name}"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像和标签路径
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # 读取图像和标签
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 将255二值标签转换为0-1标签
        mask = (mask > 0).astype(np.uint8)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'filename': self.image_files[idx]
        }

def get_transforms(img_size, is_train=True):
    """
    获取数据增强转换
    
    Args:
        img_size (tuple): 图像大小
        is_train (bool): 是否是训练集
    
    Returns:
        albumentations.Compose: 数据增强转换
    """
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.2),
            # 注意不要应用ToTensorV2，我们在Dataset中手动转换
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            # 注意不要应用ToTensorV2，我们在Dataset中手动转换
        ])

def create_dataloaders(config):
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置字典
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    train_transform = get_transforms(img_size, is_train=True) if config['data']['augmentation'] else get_transforms(img_size, is_train=False)
    val_transform = get_transforms(img_size, is_train=False)
    
    train_dataset = SegmentationDataset(
        root_dir=config['data']['train_dir'],
        img_size=img_size,
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = SegmentationDataset(
        root_dir=config['data']['val_dir'],
        img_size=img_size,
        transform=val_transform,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 
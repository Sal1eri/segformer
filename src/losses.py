import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 展平预测和标签
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 获取BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt（预测概率）
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(torch.sigmoid(inputs), targets)
        
        return self.weights[0] * bce + self.weights[1] * dice


def create_loss_fn(config):
    """
    创建损失函数实例
    
    Args:
        config: 配置字典
    
    Returns:
        loss_fn: 损失函数实例
    """
    loss_name = config['loss']['name']
    class_weights = config['loss']['class_weights']
    
    if loss_name == "cross_entropy":
        # 使用带权重的交叉熵损失
        weights = torch.FloatTensor(class_weights)
        if config['training']['device'] == "cuda":
            weights = weights.cuda()
        return nn.CrossEntropyLoss(weight=weights)
    
    elif loss_name == "dice":
        return DiceLoss()
    
    elif loss_name == "focal":
        return FocalLoss(alpha=0.25, gamma=2)
    
    elif loss_name == "bce":
        return BCEWithLogitsLoss()
    
    elif loss_name == "combined":
        return CombinedLoss(weights=[0.5, 0.5])
    
    else:
        # 默认使用交叉熵损失
        return nn.CrossEntropyLoss() 
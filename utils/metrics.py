import torch
import numpy as np


def calculate_iou(pred, target, n_classes=2, ignore_index=None):
    """
    计算IoU（交并比）
    
    Args:
        pred: 预测结果，形状为[B, H, W]
        target: 目标标签，形状为[B, H, W]
        n_classes: 类别数
        ignore_index: 忽略的类别索引
    
    Returns:
        float: IoU值
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 创建混淆矩阵
    if ignore_index is not None:
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
    
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(target, pred):
        confusion_matrix[t.long(), p.long()] += 1
    
    # 计算IoU
    iou = torch.diag(confusion_matrix) / (
        torch.sum(confusion_matrix, dim=1) + 
        torch.sum(confusion_matrix, dim=0) - 
        torch.diag(confusion_matrix)
    )
    
    # 对NaN值进行处理（避免除以0）
    iou[torch.isnan(iou)] = 0
    
    # 如果是二分类问题，只返回前景类的IoU
    if n_classes == 2:
        return iou[1].item()
    
    return iou.mean().item()


def calculate_dice(pred, target, n_classes=2, ignore_index=None):
    """
    计算Dice系数
    
    Args:
        pred: 预测结果，形状为[B, H, W]
        target: 目标标签，形状为[B, H, W]
        n_classes: 类别数
        ignore_index: 忽略的类别索引
    
    Returns:
        float: Dice系数
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 处理忽略索引
    if ignore_index is not None:
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
    
    # 将预测和目标转换为one-hot编码
    pred_one_hot = torch.eye(n_classes)[pred.long()]
    target_one_hot = torch.eye(n_classes)[target.long()]
    
    # 计算Dice系数
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=0)
    union = torch.sum(pred_one_hot, dim=0) + torch.sum(target_one_hot, dim=0)
    
    dice = (2. * intersection) / (union + 1e-8)
    
    # 对NaN值进行处理
    dice[torch.isnan(dice)] = 0
    
    # 如果是二分类问题，只返回前景类的Dice系数
    if n_classes == 2:
        return dice[1].item()
    
    return dice.mean().item()


def calculate_accuracy(pred, target, ignore_index=None):
    """
    计算准确率
    
    Args:
        pred: 预测结果，形状为[B, H, W]
        target: 目标标签，形状为[B, H, W]
        ignore_index: 忽略的类别索引
    
    Returns:
        float: 准确率
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 处理忽略索引
    if ignore_index is not None:
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
    
    correct = (pred == target).float().sum()
    total = pred.numel()
    
    return (correct / total).item()


def calculate_metrics(preds, targets, metrics=["iou", "dice", "accuracy"], n_classes=2):
    """
    计算多个评估指标
    
    Args:
        preds: 预测结果，形状为[B, H, W]
        targets: 目标标签，形状为[B, H, W]
        metrics: 要计算的指标列表
        n_classes: 类别数
    
    Returns:
        dict: 包含各项指标的字典
    """
    results = {}
    
    if "iou" in metrics:
        results["iou"] = calculate_iou(preds, targets, n_classes)
    
    if "dice" in metrics:
        results["dice"] = calculate_dice(preds, targets, n_classes)
    
    if "accuracy" in metrics:
        results["accuracy"] = calculate_accuracy(preds, targets)
    
    return results 
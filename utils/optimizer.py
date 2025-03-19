import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR


def create_optimizer(model, config):
    """
    创建优化器
    
    Args:
        model: 模型实例
        config: 配置字典
    
    Returns:
        optimizer: 优化器实例
    """
    optimizer_name = config['optimizer']['name']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    elif optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
            weight_decay=weight_decay
        )
    
    elif optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
            weight_decay=weight_decay
        )
    
    else:
        # 默认使用AdamW
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


def create_scheduler(optimizer, config, num_training_steps):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器实例
        config: 配置字典
        num_training_steps: 总训练步数
    
    Returns:
        scheduler: 学习率调度器实例
    """
    scheduler_name = config['training']['scheduler']
    num_epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']
    
    # 创建预热调度器
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs * num_training_steps // num_epochs
        )
    
    # 创建主调度器
    main_scheduler = None
    if scheduler_name == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(num_epochs - warmup_epochs) * num_training_steps // num_epochs,
            eta_min=1e-6
        )
    
    elif scheduler_name == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=num_epochs // 3 * num_training_steps // num_epochs,
            gamma=0.1
        )
    
    elif scheduler_name == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=(num_epochs - warmup_epochs) * num_training_steps // num_epochs
        )
    
    else:
        # 默认使用余弦退火
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(num_epochs - warmup_epochs) * num_training_steps // num_epochs,
            eta_min=1e-6
        )
    
    # 如果有预热，则使用顺序调度器
    if warmup_scheduler is not None:
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs * num_training_steps // num_epochs]
        )
    
    return main_scheduler 
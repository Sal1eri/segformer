import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SegformerModel(nn.Module):
    def __init__(self, config):
        """
        Segformer模型
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        self.num_classes = config['model']['num_classes']
        
        # 选择Segformer骨干网络
        backbone = config['model']['backbone']
        # 将配置中的backbone名字映射到Hugging Face模型id
        backbone_map = {
            'mit_b0': 'nvidia/mit-b0',
            'mit_b1': 'nvidia/mit-b1',
            'mit_b2': 'nvidia/mit-b2',
            'mit_b3': 'nvidia/mit-b3',
            'mit_b4': 'nvidia/mit-b4',
            'mit_b5': 'nvidia/mit-b5',
        }
        model_id = backbone_map.get(backbone, 'nvidia/mit-b2')
        
        # 加载预训练的Segformer模型
        if config['model']['pretrained']:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                reshape_last_stage=True
            )
        else:
            # 创建配置
            model_config = SegformerConfig.from_pretrained(model_id)
            model_config.num_labels = self.num_classes
            # 从头开始初始化模型
            self.model = SegformerForSemanticSegmentation(model_config)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为[B, C, H, W]
        
        Returns:
            dict: 包含logits和预测结果的字典
        """
        # SegformerForSemanticSegmentation需要pixel_values作为输入
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        
        # 对logits进行插值，调整回输入图像的大小
        h, w = x.size(2), x.size(3)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        
        # 获取预测结果
        preds = torch.argmax(logits, dim=1)
        
        return {
            'logits': logits,
            'preds': preds
        }
    
    def get_attention_maps(self, x):
        """
        获取注意力图（用于可视化）
        
        Args:
            x: 输入图像张量
        
        Returns:
            attention_maps: 注意力图
        """
        # 这个函数需要根据实际情况修改，因为transformers库的SegformerForSemanticSegmentation
        # 可能不直接提供注意力图
        # 这里只是一个占位符
        return None


def create_model(config):
    """
    创建模型实例
    
    Args:
        config: 配置字典
    
    Returns:
        model: Segformer模型实例
    """
    model = SegformerModel(config)
    return model 
import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import create_model
from utils.utils import load_config, visualize_predictions
import albumentations as A


def parse_args():
    parser = argparse.ArgumentParser(description='Segformer预测脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input-dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output-dir', type=str, default='predictions', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备 (cuda/cpu)')
    parser.add_argument('--visual', action='store_true', help='可视化预测结果')
    return parser.parse_args()


def predict_image(model, image_path, config, device):
    """
    预测单张图像
    
    Args:
        model: 模型
        image_path: 图像路径
        config: 配置字典
        device: 设备
    
    Returns:
        tuple: (原图, 预测结果)
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 预处理
    img_size = config['data']['img_size']
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
    ])
    transformed = transform(image=image)
    image_transformed = transformed['image']
    
    # 转换为张量
    image_tensor = torch.from_numpy(image_transformed).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs['logits']
        pred = outputs['preds'][0].cpu().numpy()
    
    # 调整大小为原始图像尺寸
    pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    return image, pred_resized


def predict_directory(model, input_dir, output_dir, config, device, visual=False):
    """
    预测目录中的所有图像
    
    Args:
        model: 模型
        input_dir: 输入目录
        output_dir: 输出目录
        config: 配置字典
        device: 设备
        visual: 是否可视化
    """
    os.makedirs(output_dir, exist_ok=True)
    if visual:
        visual_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(visual_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc='预测中'):
        image_path = os.path.join(input_dir, image_file)
        
        # 预测
        image, pred = predict_image(model, image_path, config, device)
        
        # 将预测结果转换为二值图像（0和255）
        pred_binary = (pred > 0).astype(np.uint8) * 255
        
        # 保存预测结果
        output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + '_mask.png')
        cv2.imwrite(output_path, pred_binary)
        
        # 可视化
        if visual:
            # 创建可视化图像
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(image)
            axes[0].set_title('原图')
            axes[0].axis('off')
            
            axes[1].imshow(pred_binary, cmap='gray')
            axes[1].set_title('预测掩码')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visual_dir, os.path.splitext(image_file)[0] + '_visual.png'))
            plt.close()


def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(f"从 {args.checkpoint} 加载模型")
    
    # 预测
    predict_directory(model, args.input_dir, args.output_dir, config, device, args.visual)
    print(f"预测完成，结果保存在 {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 
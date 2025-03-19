import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse

from models.transunet import TransUNet
import config

def predict_image(model, image_path, output_path=None, threshold=0.5):
    # 加载和预处理图像
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # 预测
    device = torch.device(config.DEVICE)
    image_tensor = image_tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()
    
    # 转换为图像
    pred_np = pred.squeeze().cpu().numpy()
    
    # 缩放回原始大小
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8)).resize(original_size)
    
    # 保存或显示结果
    if output_path:
        pred_image.save(output_path)
    
    # 创建可视化结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, cmap='gray')
    plt.title('预测 (调整大小)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(pred_image), cmap='gray')
    plt.title('预测 (原始大小)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path.replace('.png', '_comparison.png'))
    else:
        plt.show()
    
    return pred_image

def main():
    parser = argparse.ArgumentParser(description='使用TransUNet预测分割掩码')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default=None, help='保存输出掩码的路径')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'), 
                        help='模型检查点路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值分割的阈值')
    
    args = parser.parse_args()
    
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
    
    # 加载模型权重
    device = torch.device(config.DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    
    # 预测
    output_path = args.output if args.output else 'prediction.png'
    predict_image(model, args.image, output_path, args.threshold)
    
    print(f'预测结果已保存到 {output_path}')

if __name__ == '__main__':
    main() 
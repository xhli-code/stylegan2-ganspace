import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import pairwise
import numpy as np
import os
from PIL import Image

# 预训练模型加载
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的分类层

    def forward(self, x):
        return self.features(x)

# 特征提取
def extract_features(image_paths, model, transform):
    model.eval()
    features_list = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)  # 添加批次维度
            img = img.cuda() if torch.cuda.is_available() else img  # 将图像移动到GPU
            feature = model(img)
            features_list.append(feature.view(-1).cpu().numpy())  # 将特征展平并移回CPU
    return np.array(features_list)

# MMD计算
def compute_mmd(X, Y):
    # 计算均值和协方差矩阵
    X_kernel = pairwise.rbf_kernel(X, X)
    Y_kernel = pairwise.rbf_kernel(Y, Y)
    XY_kernel = pairwise.rbf_kernel(X, Y)

    mmd_squared = X_kernel.mean() + Y_kernel.mean() - 2 * XY_kernel.mean()
    return mmd_squared

# 主程序
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Compute MMD between generated and real images')
    parser.add_argument('--generated_images_path', type=str, required=True, help='Path to the generated images folder')
    parser.add_argument('--real_images_path', type=str, required=True, help='Path to the real images folder')
    parser.add_argument('--resize_dim', type=int, default=256, help='Resize dimension for input images (default: 256)')
    args = parser.parse_args()

    # 读取图像路径
    generated_images = [os.path.join(args.generated_images_path, f) for f in os.listdir(args.generated_images_path)]
    real_images = [os.path.join(args.real_images_path, f) for f in os.listdir(args.real_images_path)]

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((args.resize_dim, args.resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
    model = ResNetFeatureExtractor().cuda() if torch.cuda.is_available() else ResNetFeatureExtractor()

    # 提取特征
    generated_features = extract_features(generated_images, model, transform)
    real_features = extract_features(real_images, model, transform)

    # 计算MMD
    mmd_value = compute_mmd(generated_features, real_features)

    print(f'MMD Value: {mmd_value}')

if __name__ == '__main__':
    main()


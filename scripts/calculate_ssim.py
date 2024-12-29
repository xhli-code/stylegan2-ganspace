import argparse
import os
import random
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim_for_folder(generated_folder, real_folder, num_iterations=10):
    # 保存每次迭代的平均SSIM值
    ssim_values_all = []

    # 获取文件夹中的图像文件名列表，并确保排序一致
    generated_images = sorted(os.listdir(generated_folder))
    real_images = sorted(os.listdir(real_folder))

    for i in range(num_iterations):
        # 设置随机种子
        random.seed(i)

        # 从生成图像中随机抽取与真实图像数量一致的图像
        sampled_generated_images = random.sample(generated_images, len(real_images))
        
        # 保存当前迭代的SSIM值
        ssim_values_iteration = []

        for gen_img_name, real_img_name in zip(sampled_generated_images, real_images):
            # 获取图像的完整路径
            gen_img_path = os.path.join(generated_folder, gen_img_name)
            real_img_path = os.path.join(real_folder, real_img_name)

            # 读取图像
            gen_img = cv2.imread(gen_img_path, cv2.IMREAD_GRAYSCALE)
            real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)

            # 确保图像读取成功
            if gen_img is None or real_img is None:
                print(f"无法读取图像: {gen_img_path} 或 {real_img_path}")
                continue

            # 计算SSIM值
            ssim_value = ssim(gen_img, real_img,
                              win_size=11,               # 设置窗口大小
                              gaussian_weights=True,     # 启用高斯权重
                              sigma=1.5,                 # 设置高斯标准差
                              data_range=255)            # 数据范围
            ssim_values_iteration.append(ssim_value)

        # 计算当前迭代的平均SSIM值
        average_ssim_iteration = np.mean(ssim_values_iteration) if ssim_values_iteration else None
        if average_ssim_iteration is not None:
            ssim_values_all.append(average_ssim_iteration)
            print(f"随机种子 {i} 的平均SSIM值: {average_ssim_iteration}")

    # 计算所有迭代的平均SSIM值
    overall_average_ssim = np.mean(ssim_values_all) if ssim_values_all else None
    return overall_average_ssim

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Calculate SSIM between generated and real images")
    parser.add_argument("generated_folder", type=str, help="Path to the generated images folder")
    parser.add_argument("real_folder", type=str, help="Path to the real images folder")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations to calculate SSIM")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用计算SSIM的函数
    overall_average_ssim = calculate_ssim_for_folder(args.generated_folder, args.real_folder, args.num_iterations)

    if overall_average_ssim is not None:
        print("所有迭代的平均SSIM值:", overall_average_ssim)
    else:
        print("未能计算所有迭代的平均SSIM值")

if __name__ == "__main__":
    main()


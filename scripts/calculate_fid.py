import argparse
from pytorch_fid import fid_score

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Calculate FID score between two image folders.")
    parser.add_argument('--real_images_folder', type=str, required=True, help='Path to the real images folder.')
    parser.add_argument('--generated_images_folder', type=str, required=True, help='Path to the generated images folder.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for FID computation (default: 16).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation (default: cuda).')
    parser.add_argument('--dims', type=int, default=2048, help='Feature dimensions for Inception network (default: 2048).')
    args = parser.parse_args()

    # 计算FID距离值
    fid_value = fid_score.calculate_fid_given_paths(
        [args.real_images_folder, args.generated_images_folder],
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims
    )

    print(f'FID value: {fid_value}')

if __name__ == '__main__':
    main()


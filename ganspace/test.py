import torch
import numpy as np
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
#CUDA是NVIDIA推出的并行计算架构，该架构使GPU能够解决复杂的计算问题。
x = torch.ones((100,100))
device = torch.device(('cuda') if torch.cuda.is_available() else 'cpu')#如果没有GPU则用CPU计算
y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
x = x.to(device)                       # 或者使用`.to("cuda")`方法
z = x + y
print(z)
print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype

# python interactive.py --model=StyleGAN2 --class=shdgzma --layer=style --use_w -n=1_000_000 -b=10_000

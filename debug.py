import os
import sys
import subprocess

print("="*60)
print("CUDA 问题诊断工具")
print("="*60)

# 1. 检查 Python 版本
print(f"\n1. Python 版本: {sys.version}")

# 2. 检查 PyTorch 版本
import torch
print(f"\n2. PyTorch 版本: {torch.__version__}")
print(f"   安装路径: {torch.__file__}")

# 3. 检查 CUDA 编译版本
print(f"\n3. PyTorch CUDA 编译版本: {torch.version.cuda}")

# 4. 检查 cuDNN 版本
print(f"\n4. cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

# 5. 检查系统 CUDA
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n5. NVIDIA 驱动状态: ✅ 已安装")
        # 提取驱动版本和 CUDA 版本
        for line in result.stdout.split('\n'):
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
    else:
        print("\n5. NVIDIA 驱动状态: ❌ 未正常运行")
except FileNotFoundError:
    print("\n5. NVIDIA 驱动状态: ❌ nvidia-smi 未找到")

# 6. 检查 CUDA 路径
print("\n6. 环境变量检查:")
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
print(f"   CUDA_PATH: {cuda_path}")
print(f"   PATH: {os.environ.get('PATH', '')}")

# 7. 检查可能的冲突
print("\n7. 已安装的 CUDA 相关包:")
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if any(x in line.lower() for x in ['cuda', 'torch', 'nv']):
        print(f"   {line}")

# 8. 尝试加载 CUDA
print("\n8. 尝试加载 CUDA:")
try:
    torch.cuda.current_device()
    print(f"   ✅ CUDA 可以访问，设备: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ❌ CUDA 访问失败: {e}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)
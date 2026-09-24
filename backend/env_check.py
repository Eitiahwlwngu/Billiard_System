import torch
from ultralytics import YOLO
import cv2
import flask

print(f"1. PyTorch 版本: {torch.__version__}")
print(f"2. 是否支持 GPU (CUDA): {torch.cuda.is_available()}")
print(f"3. YOLO / Ultralytics 已就绪")
print(f"4. Flask 版本: {flask.__version__}")

# 尝试加载一下模型（它会自动下载一个小的 v8 权重用于测试环境）
try:
    model = YOLO("models/yolov8n.pt")
    print("5. 模型加载测试成功！")
except Exception as e:
    print(f"5. 模型加载测试失败: {e}")
import os
import sys
import torch

# 1. 确保调用的是你刚才 pip install -e . 的本地 YOLOv13 源码 [cite: 2026-02-14]
base_path = os.path.dirname(os.path.abspath(__file__))
# 这里的文件夹名称请务必与你解压出来的官方源码文件夹名一致
src_path = os.path.join(base_path, "YOLOv13-main")
sys.path.insert(0, src_path)

from ultralytics import YOLO


def train_full_power():
    # --- 环境检查 ---
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f" 显卡加速已就绪: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # --- 路径匹配 ---
    model_path = os.path.join(base_path, "backend", "models", "yolov13n.pt")
    data_yaml = os.path.join(base_path, "datasets", "data.yaml")

    # --- 加载模型 ---
    model = YOLO(model_path)

    # --- 开启火力全开训练 ---
    model.train(
        # 基础配置
        data=data_yaml,
        epochs=100,  # 50轮，本科毕设的黄金分割点
        imgsz=640,  # 输入图像尺寸 [cite: 2026-02-14]

        # 性能配置
        batch=16,  # 显存够就用16，报 OutOfMemory 就改8 [cite: 2026-02-14]
        device=device,  # 强制使用 RTX 3060 [cite: 2026-02-14]
        workers=0,  # Windows 系统设为 0 最稳，防止多线程死锁 [cite: 2026-02-14]

        # 优化与保存配置
        project='runs/train',  # 训练结果存放的主目录
        name='billiard_v13_exp',  # 本次实验的子目录名
        exist_ok=True,  # 如果文件夹存在，直接覆盖，不报错
        pretrained=True,  # 使用预训练权重加速收敛 [cite: 2026-02-14]
        optimizer='auto',  # 自动选择优化器（SGD/AdamW）

        # 论文素材配置（自动生成图表）
        plots=True,  # 自动生成 PR 曲线、Loss 曲线、F1 曲线 [cite: 2026-02-14]
        save=True,  # 训练完自动保存 best.pt 和 last.pt
        val=True,  # 每一轮训练完都进行一次验证
    )


if __name__ == "__main__":
    train_full_power()
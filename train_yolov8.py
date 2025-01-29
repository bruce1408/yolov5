import os
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 加载YOLOv8模型
model = YOLO('/mnt/share_disk/bruce_trie/yolov5/yolov8s.pt')  # 你可以选择不同的模型权重文件

# 设置训练参数
train_params = {
    'data': '/mnt/share_disk/bruce_trie/yolov5/data/coco.yaml',  # 数据集配置文件路径
    'epochs': 100,  # 训练轮数
    'imgsz': 640,  # 输入图像大小
    'device': '2',  # 使用的GPU设备编号
    'batch': 128,
}

# 开始训练
model.train(**train_params)

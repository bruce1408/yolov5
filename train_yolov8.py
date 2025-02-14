import os
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# 加载YOLOv8模型
# model = YOLO('/mnt/share_disk/bruce_trie/yolov5/yolov8n.pt')  # 你可以选择不同的模型权重文件
model = YOLO('/mnt/share_disk/bruce_trie/runs/detect/train13/weights/last.pt')  # 你可以选择不同的模型权重文件

# 获取上次训练的epoch数
# 可以从模型对象中获取训练信息
last_epoch = model.ckpt['epoch'] if hasattr(model, 'ckpt') else 0
current_epoch = 100 - last_epoch 
# 设置训练参数
train_params = {
    'data': '/mnt/share_disk/bruce_trie/yolov5/data/coco.yaml',  # 数据集配置文件路径
    'epochs': current_epoch,  # 训练轮数
    'imgsz': 640,  # 输入图像大小
    'device': '7',  # 使用的GPU设备编号
    'batch': 128,
}

# 开始训练
model.train(**train_params)

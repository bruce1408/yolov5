import torch
from models_2.yolo_2 import DetectionModel
from models_2.experimental import attempt_load

model = attempt_load("./runs/train/exp7/weights/best.pt", device=torch.device('cpu'))

model = model.half()

torch.save(model.state_dict(), "ss.pt")

# a = 1


# ss_model = torch.load("ss.pt")


# yolo_model = DetectionModel("./models/yolov5m.yaml")

# torch.save(yolo_model.state_dict(), "yolo.pt")

# yolo_ckpt = torch.load("yolo.pt")

# a = 1
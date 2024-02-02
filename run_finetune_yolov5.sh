#!/bin/bash

workdir=$(cd "$(dirname "$0")" || exit; pwd)

cd "$workdir" || exit

python train.py --batch 64 \
    --cfg yolov5m.yaml \
    --data data/coco.yaml \
    --weights /mnt/share_disk/cdd/model_prune/yolov5/runs/train/exp7/weights/best_relu.pt \
    --device 1
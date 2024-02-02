#!/bin/bash

workdir=$(cd "$(dirname "$0")" || exit; pwd)

cd "$workdir" || exit

python val.py \
    --data data/coco.yaml \
    --batch-size 256 \
    --weights /mnt/share_disk/cdd/model_prune/yolov5/runs/train/exp7/weights/best_relu.pt \
    --device 0,1
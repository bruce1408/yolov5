from __future__ import absolute_import

import torch
from pathlib import Path
import numpy as np
import glob
import os
import pickle
from copy import deepcopy

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)



    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    box1_expanded = np.expand_dims(box1, axis=1)
    (a1, a2) = np.split(box1_expanded, 2, axis=2)

    box2_expanded = np.expand_dims(box2, axis=0)
    (b1, b2) = np.split(box2_expanded, 2, axis=2)

    # (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = np.prod(np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None), axis=2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (np.prod((a2 - a1), axis=2) + np.prod((b2 - b1), axis=2) - inter + eps)



def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            # matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            matches = np.concatenate((np.stack(x, axis=1), iou[x[0], x[1]][:, None]), axis=1)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y



def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec



def metrics(preds, targets, img_path, img_size, names):

    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = np.size(iouv)
    jdict, stat = [], []
    seen = 0

    tmp =[]
    for idx in targets:
        idx[0] = names[idx[0]]
        tmp.append(idx)

    targets = np.array(tmp)


    tmp =[]
    for idx in preds:
        idx[5] = names[idx[5]]
        tmp.append(idx)

    preds = np.array(tmp)

    labels = targets
    nl, npr = len(labels), len(pred)  # number of labels, predictions
    # path, shape = Path(img_path), img_size
    correct = np.zeros((npr, niou), dtype=bool)  # init
    seen += 1

    # if npr == 0:
    #     if nl:
    #         stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))

    # Predictions
    predsn = deepcopy(preds)
    # scale_boxes(input_img.shape[1:], predn[:, :4], shape, img_size[1])  # native-space pred

    # Evaluate
    if nl:
        # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        # tbox = labels[:, 1:5]
        # scale_boxes(input_img.shape[1:], tbox, shape, img_size[1])  # native-space labels
        # labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
        labelsn = deepcopy(labels)
        correct = process_batch(predsn, labelsn, iouv)
        # if plots:
        #     confusion_matrix.process_batch(predn, labelsn)
        return (correct, preds[:, 4], preds[:, 5], labels[:, 0])  # (correct, conf, pcls, tcls)
    else:
        return (correct, preds[:, 4], preds[:, 5], np.array([]))  # (correct, conf, pcls, tcls)


if __name__ == '__main__':

    names = []

    with open('datasets/coco/coco.names', 'r') as f:
        lines = f.readlines()

    with open('datasets/coco/coco.sizes.pkl', 'rb') as f:
        size = pickle.load(f)

    dt_path = "./mAP/input/detection-results"
    gt_path = "./mAP/input/ground-truth"

    names = [line.strip() for line in lines]
    names_dict = {i:idx for idx , i in enumerate(names)}

    nc = 80
    jdict, stats, ap, ap_class = [], [], [], []

    img_txt_path = []

    with open('datasets/coco/val2017.txt', 'r') as f:
        lines = f.readlines()

    img_txt_path = [line.strip().split('/')[-1].split('.')[0] for line in lines]

    for img_txt in img_txt_path:
        dt_file_path = os.path.join(dt_path, img_txt + '.txt')
        gt_file_path = os.path.join(gt_path, img_txt + '.txt')
        if os.path.exists(dt_file_path) and os.path.exists(gt_file_path):

            with open(dt_file_path, 'r') as f:
                lines = f.readlines()

            dt = [line.split(' ') for line in lines]
            pred = []
            for box in dt:
                cls_name = box[0]
                conf = float(box[1])
                x_s = float(box[2])
                y_s = float(box[3])
                x_e = float(box[4])
                y_e = float(box[5])

                pred.append([x_s, y_s, x_e, y_e, conf, cls_name])



            with open(gt_file_path, 'r') as f:
                lines = f.readlines()


            gt = [line.split(' ') for line in lines]
            label = []
            for box in gt:
                cls_name = box[0]
                x_s = float(box[1])
                y_s = float(box[2])
                x_e = float(box[3])
                y_e = float(box[4])

                label.append([cls_name, x_s, y_s, x_e, y_e])
            stats.append(metrics(preds=pred, targets=label, img_path=img_txt, img_size=size[img_txt], names=names_dict))

        else:
            print("pass a empty img pred")


    stats = [np.concatenate(x, axis=0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names=names_dict)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print(f"mAP@50: {map50}, mAP@50:95: {map}")

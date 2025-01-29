# 基于[ultralytics的yolov5](https://github.com/ultralytics/yolov5)仓库修改而来

## 修改 | 去掉YOLOv5的detect, 并导出为onnx

**修改`models/yolo.py`**

删去用于处理detect的部分, 只保留transpose以及之前的处理.

```python
# Detect::forward
def forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()


    #     if not self.training:  # inference
    #         if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #             self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

    #         if isinstance(self, Segment):  # (boxes + masks)
    #             xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
    #             xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
    #             wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
    #             y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
    #         else:  # Detect (boxes only)
    #             xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
    #             xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
    #             wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
    #             y = torch.cat((xy, wh, conf), 4)

    #         z.append(y.view(bs, self.na * nx * ny, self.no))
    # return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    return x[0], x[1], x[2]
```

**在`export.py`中添加output的名字**

```python
# export_onnx函数中, 行160
    # output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0', 'output1', 'output2']
```

*在`val.py`添加detect的处理**

已经另存为`val_cut_head.py`

将原来的val获取预测结果部分
```python
def run():
    ...
    preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
    ...
```
修改为
```python
def run():
    ...
    x_0, x_1, x_2 = model(im, augment=augment)
    x = [x_0, x_1, x_2]
    z = []
    strides = [8, 16, 32]
    anchor = torch.FloatTensor(np.array([[[10,13], [16,30], [33,23]],
                [[30,61], [62,45], [59,119]],
                [[116,90], [156,198], [373,326]]], dtype=np.float32))
    anchor = anchor.to(device)
    # print(anchor[1].shape)
    anchor_grid = [torch.empty(0) for _ in range(3)]
    grid = [torch.empty(0) for _ in range(3)]

    for i in range(len(x)):
        # print (x[i].shape)
        bs, na, nx, ny, no = x[i].shape
        # input_x = np.loadtxt(f"./result_{i}_tensor.txt", delimiter='\n')
        with open(f"./result_{i}_tensor.txt") as f:
            output_value = [float(eachline.strip("\n")) for eachline in f]
        x[i] = torch.from_numpy(np.array(output_value).reshape((bs, na, nx, ny, no)))


        print(x[i].shape)

        grid[i], anchor_grid[i] = _make_grid(nx, ny,na,  anchor[i])
        xy, wh, conf = x[i].sigmoid().split((2, 2, 80 + 1), 4)
        xy = (xy * 2 + grid[i]) * strides[i]  # xy
        wh = (wh * 2) ** 2 * anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, na * ny * nx, no))

    preds, train_out = torch.cat(z, 1), None
    ...
```
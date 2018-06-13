import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable


def GetBox(outdata, conf_thresh, anchors):
    print('inprostprocess', outdata.dim())
    if outdata.dim() ==3:
        outdata = outdata.unsqueeze(0)
    print('inprostprocess', outdata.dim())
    batch = outdata.size(0)
    print(outdata.size(0))
    print(outdata.size(1) == 6*5)
    print(outdata.size(2) == 13)
    print(outdata.size(3) == 13)
    h = outdata.size(2)
    w = outdata.size(3)

    all_boxes = []

    output = outdata.view(5, 6, 13*13).transpose(0,1).contiguous().view(6, 5*h*w)
    # print(output.size(0), output.size(1))
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(5, 1, 1).view(5*h*w)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(5, 1, 1).view(5*h*w)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
    anchor_step = int(1) #int(len(anchors)/5)
    anchor_w = anchors[:, 0].contiguous().view(5, anchor_step)
    anchor_h = anchors[:, 1].contiguous().view(5, anchor_step)
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(5*h*w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(5*h*w)
    ws = torch.exp(output[2]) * anchor_w.data
    hs = torch.exp(output[3]) * anchor_h.data

    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax()(Variable(output[5:6].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h*w
    sz_hwa = sz_hw*5
    det_confs = torch.FloatTensor(det_confs.size()).copy_(det_confs)
    cls_max_confs = torch.FloatTensor(cls_max_confs.size()).copy_(cls_max_confs)
    cls_max_ids = torch.FloatTensor(cls_max_ids.size()).copy_(cls_max_ids)
    xs = torch.FloatTensor(xs.size()).copy_(xs)
    ys = torch.FloatTensor(ys.size()).copy_(ys)
    ws = torch.FloatTensor(ws.size()).copy_(ws)
    hs = torch.FloatTensor(hs.size()).copy_(hs)

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(5):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    if conf > conf_thresh:
                        box = {'x':xs[ind], 'y':ys[ind], 'w':ws[ind], 'h':hs[ind], 'max_indx':cls_max_ids[ind], 'max_prob':cls_max_confs[ind]}
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes

def BoxInfo(b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    if max_prob > threshold:
        left = int((b.x - b.w/2.)*w)
        right = int((b.x + b.w/2.)*w)
        top = int((b.y - b.h/2.)*h)
        bot = int((b.y + b.h/2.)*h)
        if left < 0: left = 0
        if right > w - 1: right = w - 1
        if top < 0: top = 0
        if bot > h - 1: bot = h - 1
        return (left, right, top, bot, max_indx, max_prob)
    return None

def IOU(expectedB, actualB):
    BoxE = boxInfo(expectedB)
    BoxA = boxInfo(actualB)

    #intersect
    minX = max(BoxE[0], BoxA[0])
    maxX = min(BoxE[1], BoxA[1])
    maxY = max(BoxE[2], BoxA[2])
    minY = min(BoxE[3], BoxA[3])

    #no overlap
    if minX > maxX or minY > maxY:
        result = 0.0
        return result

    areaE = (BoxE[1]-BoxE[0]) * (BoxE[3]-BoxE[2])
    areaA = (BoxA[1]-BoxA[0]) * (BoxA[3]-BoxA[2])
    intersect = (maxX-minX) * (maxY-minY)
    union = areaE + areaA - intersect
    accuracy = intersect/union
    return accuracy

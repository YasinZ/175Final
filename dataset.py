import os
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from parser import imageInfo
from processing import ImgLoss

class GetData(data.Dataset):
    def __init__(self, root, train=False, transform=None, use_cuda=True):
        self.datas = imageInfo(root)
        self.transform = transform
        self.train = train
        self.shape = (13 * 32, 13 * 32)
        self.use_cuda = use_cuda

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img, loss= ImgLoss(self.datas[index], self.train)

        if self.transform is not None:
            img = self.transform(img)
        return img, loss

class Filtered(GetData):
    __init__ = GetData.__init__

    def __getitem__(self, index):
        return super(Filtered, self).__getitem__(index)

def none_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return data.dataloader.default_collate(batch)

def targetToVariable(target, train=True, use_cuda=True):
    for k, v in target.items():
        if train:
            target[k] = Variable(v).cuda()
        else:
            target[k] = Variable(v, volatile=True).cuda()
    return target

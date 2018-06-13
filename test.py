import argparse
import os
import shutil

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import torch.utils.data as data
import torchvision.transforms as transforms


from dataset import Filtered, none_collate, targetToVariable
from tinyYOLO import TinyYOLO, TinyDarkNet
from loss import Loss
from box import IOU, BoxInfo

import shutil


def test(args):
    print('Loading network....')
    pretrained = TinyDarkNet()
    checkpoint = torch.load('./best_checkpoint.pth.tar')

    print('Loading model....')
    model = TinyYOLO()
    model.darknet = pretrained.net
    model.load_state_dict(checkpoint['state_dict'])

    if args['use_cuda']:
        model = nn.DataParallel(model)
        model.cuda()
        cudnn.benchmark = True
    model.eval()

    creterion = Loss(args)
    transformer = transforms.Compose([transforms.ToTensor()])
    kwargs = {'num_workers':1, 'pin_memory':True} if args['use_cuda'] else {}

    print('Loading test data....')
    testloader = data.DataLoader(
            Filtered(args['test'],
                      train=False,
                      transform=transformer,
                      use_cuda=args['use_cuda']),
            batch_size=args['batch_size'],
            collate_fn=none_collate,
            shuffle=False,
            **kwargs)

    print('Testing data...')
    anchors = Variable(torch.FloatTensor(args['anchors']))
    #nms_thresh = 0.45
    predictions = testloader.data #imageList

    if not os.path.exists('result'):
        os.makedirs('result')
    outfile = 'result/accuracy_eval.txt'
    if args['result_file_name'] != None:
        outfile = 'result/' + args['result_file_name'] + '.txt'
    result = open(outfile,”w”)

    #first line of the output file
    str = '\t{0:10}{1:10}{2:10}{3:10}{4:10}'.format('minx','miny', 'width', 'height', 'accuracy')
    result.write(str)

    best_loss = 10000000
    loss_per_epoch = 0

    #for each test image data
    for i, (x, y) in enumerate(testloader, 1):
        x = Variable(x).cuda() if args['use_cuda'] else Variable(x)
        y = targetToVariable(y, use_cuda=args['use_cuda'])
        y_pred = model(x)

        boxes = postprocess(y_pred, anchors.cpu(), predictions[i], result)
        #postprocess(y_pred[0].cpu(), anchors.cpu(), valid_images[lineId])
    result.close()


def postprocess(output, anchors, pred, fp):


    #output = output.permute(1, 2, 0)

    im = pred['name']
    pred_left = pred['xmin']
    pred_bottom = pred['ymin']
    pred_right = pred['xmin'] + pred['width']
    pred_top = pred['ymin'] + pred['height']
    pred_Box = [pred_left, pred_right, pred_top, pred_bottom]

    #boxes = box_constructor(np.ascontiguousarray(output.data.numpy()),
    #np.ascontiguousarray(anchors.view(-1).data.numpy().astype('float64')))


    imgcv = cv2.imread(im)
    h, w, _ = imgcv.shape

    #constructor
    conf_thresh = 0.005
    boxes = GetBox(output.data, conf_thresh, anchors)
    # max 2 boxes
    box0 = {'box': None, 'prob': None}
    box1 = {'box': None, 'prob': None}
    for i, b in enumerate(boxes):
        if i == 0:
            b0 = {'box': b, 'prob': b.probs[0]}
        elif b0['prob'] < b.probs[0]:
            b1 = {'box': b0['box'], 'prob': b0['prob']}
            b0 = {'box': b, 'prob': b.probs[0]}
        else:
            if i == 1:
                b1 = {'box': b, 'prob': b.probs[0]}
            elif b1['prob'] < b.probs[0]:
                b1 = {'box': b, 'prob': b.probs[0]}
    twoBoxes = [b0['box'], b1['box']]
    fp.write(im)

    threshold = 0.2
    for b in twoBoxes:
        boxResult = BoxInfo(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, max_indx, confidence = boxResult
        thick = int((h+w) //300)
        cv2.rectangle(imgcv, (left, top), (right,bot), (0, 255, 0), thick)
        accuracy = IOU(pred_Boxm [left, right, top, bot])
        str = '\t{0:10}{1:10}{2:10}{3:10}{4:10}'.format(left, bot, right-left, top-bot, accuracy)
        fp.write(str)
    outfolder = os.path.join('./', 'outImage')
    img_name = os.path.join(outfolder, os.path.basename(im))
    cv2.imwrite(img_name, imgcv)



if __name__ == '__main__':
    args = {'epoch': 155,
            'lr': 0.001,
            'steps': [-1, 1, 77, 116],
            'scales': [.1, 10, .1, .1],
            'use_cuda':True,
            'class_scale':1.,
            'object_scale':5.,
            'noobject_scale':1.,
            'coord_scale':1.,
            'anchors':[
                [1.08,1.19],
                [3.42,4.41],
                [6.63,11.38],
                [9.42,5.11],
                [16.62,10.52],
                ],
            'batch_size': 1,
            'result_file_name': 'temp',
            'train': 'Dataset/small_train.txt',
            'test': 'Dataset/test_data.txt',}
    test(args)

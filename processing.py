from copy import deepcopy
import pickle
import numpy as np
import os
import cv2

def ImgLoss(chunk, train=False):
    h, w = 416, 416
    H, W = 13, 13
    C, B = 1, 5

    jpg = 'ProcessedImages/' + chunk['name'] + '.jpg'
    xmin = chunk['xmin']
    xmax = chunk['xmin'] + chunk['width']
    ymin = chunk['ymin']
    ymax = chunk['ymin'] + chunk['height']
    annotation = [chunk['name'], xmin, ymin, xmax, ymax]

    obj = deepcopy(annotation)
    img = cv2.imread(jpg)

    # 13x13 grids
    cellx = 1. * w / W
    celly = 1. * h / H

    centerx = .5*(obj[1]+obj[3]) #xmin, xmax
    centery = .5*(obj[2]+obj[4]) #ymin, ymax
    cx = centerx / cellx
    cy = centery / celly
    if cx >= W or cy >= H: return None, None
    obj[3] = float(obj[3]-obj[1]) / w
    obj[4] = float(obj[4]-obj[2]) / h
    obj[3] = np.sqrt(obj[3])
    obj[4] = np.sqrt(obj[4])
    obj[1] = cx - np.floor(cx) # centerx
    obj[2] = cy - np.floor(cy) # centery
    obj += [int(np.floor(cy) * W + np.floor(cx))]

    #for each anchorbox in grid
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])

    probs[:, :, :] = [[0.]*C] * B
    probs[:, :, :] = 1.
    proid[:, :, :] = [[1.]*C] * B
    coord[:, :, :] = [obj[1:5]] * B
    prear[:,0] = obj[1] - obj[3]**2 * .5 * W # xleft
    prear[:,1] = obj[2] - obj[4]**2 * .5 * H # yup
    prear[:,2] = obj[1] + obj[3]**2 * .5 * W # xright
    prear[:,3] = obj[2] + obj[4]**2 * .5 * H # ybot
    confs[:, :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft;
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    loss_feed = {
    'probs': probs.astype('float32'), 'confs': confs.astype('float32'),
    'coord': coord.astype('float32'), 'proid': proid.astype('float32'),
    'areas': areas.astype('float32'), 'upleft': upleft.astype('float32'),
    'botright': botright.astype('float32')
    }

    return img, loss_feed

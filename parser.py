import os
import sys
import glob

def imageInfo(root):
    imageList = []
    #root =  = 'Dataset/processed_data.txt'
    f = open(root, 'r')
    for line in f:
        infoList = {}
        a, b, c, d, e = line.rstrip().split(' ')
        infoList['name'] = a
        infoList['xmin'] = float(b)
        infoList['ymin'] = float(c)
        infoList['width'] = float(d)
        infoList['height']= float(e)
        imageList.append(infoList)

    return imageList

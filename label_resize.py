import sys
import json
import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt


def writeInfo(filename):
    with open(filename) as f:
        data = json.load(f)

        processed_data = dict()
        processed_data_formatted = dict()
        for key, value in data.items():
            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = -sys.maxsize -1
            max_y = -sys.maxsize -1
            for x in value:
                if(min_x > x[0]):
                    min_x = int(x[0])
                if(min_y > x[1]):
                    min_y = int(x[1])
                if(max_x < x[0]):
                    max_x = int(x[0])
                if(max_y < x[1]):
                    max_y = int(x[1])

        min_x = int(min_x * (416/1920))
        max_x = int(max_x * (416/1920))
        min_y = int(min_y * (416/1920) + (416-416*1080/1920) * .5)
        max_y = int(max_y * (416/1920) + (416-416*1080/1920) * .5)
        processed_data[key] = [[min_x, min_y], [max_x, max_y]]
        processed_data_formatted[key] = [min_x, min_y, (max_x-min_x), (max_y-min_y)]

        File = open('Dataset/processed_data.txt', 'w')
        i = 1
        for key, value in processed_data_formatted.items():
            File.write(key + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + str(value[2]) + ' ' + str(value[3]) + '\n')

def resize(filename):
    i = 0
    for file in glob.glob("Dataset/Color/*.jpg"):
        image = cv2.imread(file)
        image = cv2.resize(image, (416, int(416*(1080/1920))))
        image = cv2.copyMakeBorder(image, top=91, bottom=91, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
        name = file[14:]
        name, ext = name.rstrip().split('.')
        if not os.path.exists('ProcessedImages'):
            os.makedirs('ProcessedImages')
        cv2.imwrite('ProcessedImages/' + name + '_R.jpg', image)
        cv2.imwrite('ProcessedImages/' + name + '_L.jpg', image)

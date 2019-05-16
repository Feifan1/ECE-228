# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:58:21 2019

@author: Feifan
"""
import numpy as np
import scipy.io as sio 
import os
#import cv2
from PIL import Image




image_dir = "C:/Users/56284/Downloads/ECE228/cars_train/"
class_dir = "C:/Users/56284/Downloads/ECE228/devkit/cars_meta.mat"
image_train_label_dir = "C:/Users/56284/Downloads/ECE228/devkit/cars_train_annos.mat"

def getclass():
    annos = sio.loadmat(class_dir)
    _, num_class = annos['class_names'].shape
    class_label = {i:annos['class_names'][0,i][0] for i in range(num_class)}
    return class_label


def getlabel():
    annos = sio.loadmat(image_train_label_dir)
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    pic_idx = ['' for i in range(total_size)]
    for i in range(total_size):
        path = annos["annotations"][:,i][0][5][0].split(".")
        pic_idx[i] = path[0]
        id = int(path[0]) - 1
        #print(path)
        for j in range(5): #('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O')
            labels[id, j] = int(annos["annotations"][:,i][0][j][0])
    return labels, pic_idx


def getitem(ind, labels): # ind is choose from pic_idx
    image_path = os.path.join(image_dir, ind, '.jpg')
    image = Image.open(image_path).convert('RGB')
    
    label = labels[int(ind)-1, 4]
    
    return (image, label)
    
    
    
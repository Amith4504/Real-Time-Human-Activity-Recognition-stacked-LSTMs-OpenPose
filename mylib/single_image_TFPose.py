#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 05:11:55 2019

@author: amith
"""

import numpy as np
import cv2
from mylib.displays import drawActionResult
import mylib.funcs as myfunc
import mylib.feature_proc as myproc
import mylib.io as myio
from mylib.action_classifier import ClassifierOnlineTest
from mylib.action_classifier import *
import sys,os,time,argparse,logging
import simplejson
import math


# to check inputs to ClassifierOnlineTest
CURR_PATH = os.path.dirname(os.path.abspath("__file__"))+"/"
sys.path.append(CURR_PATH + "githubs/tf-pose-estimation")
import tf_pose
from tf_pose.networks import get_graph_path,model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch= logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()


# to detect skeleton from a single input image
model = "cmu"
image_size = "432x368"

resize_out_ratio = 4.0

w,h = model_wh(image_size)

e = TfPoseEstimator(get_graph_path(model),
            target_size=(w,h),
            tf_config=config)

            
fps_time = time.time()
cnt_image = 0

def detect(image2):
    image_h = image2.shape[0]
    image_w = image2.shape[1]
    scale_y = 1.0 * image_h / image_w
    # multiple humans are detected
    humans = e.inference(image,resize_to_default=(w>0 and h>0),
                         upsample_size = resize_out_ratio)
    
    return humans

def draw(imag_disp,humans):
    
    imag_disp = TfPoseEstimator.draw_humans()
    
    if DRAW_FPS:
        cv2.putText(img_disp,
                    "fps = {:.1f}".format((1.0 / (time.time() - fps_time))),
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,1
                    (0,0,255),2)
        fps_time = time.time()
        
def humans_to_skelsList(humans):
        
    
    skeletons = []
    NaN = 0
    
    for human in humans :
        skeleton = [NaN]*(18*2)
        for i,body_part in human.body_parts.items():
            idx = body_part.part_idx
            skeleton[2*idx] = body_part.x
            skeleton[2*idx+1] = body_part.y * scale_y
        skeleton.append(skeleton)
        """
        for i in range(len(skeleton)):
            
            if i ==(len(skeleton)-1):
                file_skel.write(str(skeleton[i]))
            else:
                file_skel.write(str(skelton[i]) + ",")
                
        file_skel.write("\n")
        """
    return skeletons,scale_y


image = cv2.imread("football.jpg")
k = image.shape
scale_y = 1.0 * image.shape[0] / image.shape[1]
humans = detect(image)
skeletons , scale_y = humans_to_skelsList(humans)
print(skeletons)
print(humans)

for human in humans:
    print(human)
    k= type(human)
    print(k)
NaN = 0
skeleton = [NaN]*(18*2)
skeletons = []
for human in humans:
    skeleton = [NaN]*(18*2)
    for i , body_part in human.body_parts.items():
        idx = body_part.part_idx
        skeleton[2*idx] = body_part.x
        skeleton[2*idx +1] = body_part.y * scale_y
    skeletons.append(skeleton)
        
        
        
        
        
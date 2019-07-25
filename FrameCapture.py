#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:34:38 2019

@author: amith

VIDEO TO FRAMES
"""

import cv2

def getFrame(sec):
    vidObj.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidObj.read()
    if hasFrames:
        cv2.imWrite("image"+str(count)+".jpg",image)
    
    return hasFrames

    
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    sec= 0
    frameRate = 0.5
    count =1
    success = 1
    
    while success:
        success , image = vidObj.read()
        
        cv2.imwrite("frame%d.jpg"%count,image)
        
        count +=1
        

if __name__ == '__main__':
    FrameCapture("")
    




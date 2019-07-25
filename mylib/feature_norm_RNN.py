#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:26:23 2019

@author: Amith

FOR NORMALIZATION
"""

import numpy as np
import math

PI = np.pi
Inf = float("inf")

calc_dist = lambda p1,p0: math.sqrt((p1[0] - p0[0])**2 + (p1[1]-p0[0])**2)

def pi2pi(x):
    if x>PI:
        x -= 2*PI
    if x<+ -PI:
        x += 2*PI
    return x

# PROCESS SKELETON
    
NECK = 0
L_ARMS = [1,2,3]
R_ARMS = [4,5,6]
L_LEGS = [8,9]
R_LEGS = [11,12]
ARMS_LEGS = L_ARMS + R_ARMS + L_LEGS + R_LEGS
L_THIGH = 7
R_THIGH = 10
NotANum = 0
    
def get_joint(x,idx):
    px = x[idx]
    py = x[idx+1]
    return px,py

def set_joint(x,idx,px,py):
    x[idx] = px
    x[idx+1] = py
    return

def get_body_height(x):
        if 0:
            px0,py0 = get_joint(x,NECK)
            px_l_thigh , py_l_thigh = get_joint(x, L_THIGH)
            px_r_thigh , py_r_thigh = get_joint(x,R_THIGH)
            
            if px_l_thigh == NotANum and px_r_thigh == NotANum:
                return 1
                
            if px_l_thigh == NotANum:
                px_l_thigh, py_l_thigh = get_joint(x, R_THIGH)

            if px_r_thigh == NotANum:
                px_r_thigh, py_r_thigh = get_joint(x, L_THIGH)

            assert px_r_thigh != NotANum

            px_mid = (px_l_thigh+px_r_thigh)/2
            py_mid = (py_l_thigh+py_r_thigh)/2
            
            body_height = math.sqrt((px0-px_mid)**2 + (py0-py_mid)**2)
            return body_height
        else:
            px = x[0::2]
            py = x[1::2]
            return np.max(py) -  np.min(py)    
    
def remove_body_offset(x):
        x = x.copy()
        if 0:
            px0,py0 = get_joint(x,NECK)
            x[0::2] = x[0::2] - px0
            x[1::2] = x[1::2] - py0
        else:
            x[0::2] -= x[0::2].mean()
            x[1::2] -= x[1::2].mean()
            
        return x
    

def normalise(x):
        height = get_body_height(x)
        x_norm_list = remove_body_offset(x)/ height
        
        return x_norm_list        
        
        
    

            

    
    

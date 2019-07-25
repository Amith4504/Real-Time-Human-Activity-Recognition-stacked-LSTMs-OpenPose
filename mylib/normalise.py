#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:27:35 2019

@author: amith

NORMALIZE X_TEST , X_TRAIN
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATASET_PATH = "data/HAR_pose_activities/database/"
X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"
n_steps = 32

def load_X(X_path):
            
            file = open(X_path, 'r')
            X_ = np.array(
                    [elem for elem in [
                            row.split(',') for row in file
                            ]], 
                            dtype=np.float32
                            )   
            file.close()
            blocks = int(len(X_) / n_steps)
    
            X_ = np.array(np.split(X_,blocks))

            return X_ 
        
X_test= load_X(X_test_path)



X_test_norm= []
k=0
for i in X_test:
    k +=1
    for j in X_test[k-1,:,:]:
        norm = np.interp(j,(0,500),(0,1))
        X_test_norm.append(norm)
blocks = int(len(X_test_norm)/ n_steps)
X_test_norm = np.array(X_test_norm)        
X_test_norm = np.array(np.split(X_test_norm,blocks))
        
        
        
        
for i in X_test[0,:,:]:
    print(i)

scalar = MinMaxScaler(feature_range =(0,1))
scalar = scalar.fit(X_test[0,:,:])

normalized = scalar.transform(X_test[0,:,:])
for i in  ranfe

def printit(X_path):
    file = open(X_path,'r')
    for row in file:
        print(row)
    return

normalise(X_test_path)

norm = np.interp(X_test[0,0,:],(0,500),(0,1))

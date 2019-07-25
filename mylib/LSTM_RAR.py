#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:33:22 2019

@author: amith

LSTM KERAS implementation
"""

import tensorflow as tf
import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import pickle,sys,keras
from keras.models import Sequential
from keras import layers
from keras import regularizers
#expected input data shape:(batch_size,timesteps,data_dim)

class LSTM_classifier():
    
    def __init__(self):
        
        self.init_param()
        self.init_graph()
        
    def init_param(self):
        
        self.timesteps = 32
        self.n_steps = 32
        self.data_dim = 36 # 18*2 points
        self.num_classes = 6 
        self.batch_size = 181
        self.learning_rate = 0.0025
        self.decay_rate = 0.02
        self.lambda_loss_amount = 0.0015
        self.n_hidden = 34 # hidden number of lstms
        self.filepath = 'RNN_RAR.wts.h5'
        self.training_epochs = 100
        self.DATASET_PATH = "data/HAR_pose_activities/database/"
        self.X_train_path = self.DATASET_PATH + "X_train.txt"
        self.X_test_path = self.DATASET_PATH + "X_test.txt"

        self.y_train_path = self.DATASET_PATH + "Y_train.txt"
        self.y_test_path = self.DATASET_PATH + "Y_test.txt"
        
    def init_graph(self):
        
        """ LSTM IMPLEMENTATION  """
        model = models.Sequential()
        model.add(Dense(self.n_hidden,activation='relu',input_shape=(self.timesteps,self.data_dim)))
        model.add(LSTM(self.n_hidden,return_sequences = True))
        model.add(LSTM(self.n_hidden,input_shape=(self.timesteps,self.data_dim)))
        model.add(Dense(self.num_classes,activation = 'softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


    def Train(self):
        
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

        def load_y(y_path):
            file = open(y_path, 'r')
            y_ = np.array(
                    [elem for elem in [
                            row.replace('  ', ' ').strip().split(' ') for row in file
                            ]], 
                            dtype=np.int32
                            )
            file.close()
    
            # for 0-based indexing 
            return y_ - 1
        
        """ TRAINING OVER 100 EPOCHS """
        X_train = load_X(self.X_train_path)
        X_test = load_X(self.X_test_path)

        y_train = load_y(self.y_train_path)
        y_test = load_y(self.y_test_path)

        # DATA LOADED INTO LISTS  x_TRAIN, Y_TRAIN , X_TEST , Y_TEST
        #training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
        #test_data_count = len(X_test)  # 1197 test series
        #n_input = len(X_train[0][0])  # num input parameters per timestep
        
        y_train_one_hot = keras.utils.to_categorical(y_train,6)
        y_test_one_hot = keras.utils.to_categorical(y_test,6)

        history = model.fit(
                X_train,
                y_train_one_hot,
                epochs=self.training_epochs,
                sbatch_size = self.batch_size,
                verbose = 1,
                validation_data=(X_test,y_test_one_hot),
                callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                           keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
        # PLOT a LOSS CURVE and convergence per epoch
        plt.figure()
        plt.title('Training performance')
        plt.plot(history.epoch, history.history['loss'], label='train loss+error')
        plt.plot(history.epoch, history.history['val_loss'], label='val_error')
        plt.legend()


        model.load_weights(filepath)
        
        
        
    def Predict(self,skelton_data):
        #arguement (32,36)
        model.load_weights(filepath)
        k1 = np.expand_dims(skeleton_data,axis=0)
        Y_pred = model.predict(k1)
        return Y_pred
    
    
    
# excecute
        

    
    
        
        
        
        
        
    
"""   
LABELS = [    
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"

]

"""
"""
model = Sequential([
   # relu activation
   layers.Dense(n_hidden, activation='relu', 
       kernel_initializer='random_normal', 
       bias_initializer='random_normal',
       batch_input_shape=(batch_size, n_steps, n_input)
   ),
   layers.LSTM(n_hidden, return_sequences=True,  unit_forget_bias=1.0),
   layers.LSTM(n_hidden,  unit_forget_bias=1.0),

   layers.Dense(n_classes, kernel_initializer='random_normal', 
       bias_initializer='random_normal',
       kernel_regularizer=regularizers.l2(lambda_loss_amount),
       bias_regularizer=regularizers.l2(lambda_loss_amount),
       activation='softmax'
   )
])
"""
#train_size = X_train.shape[0] - X_train.shape[0]
#test_size = X_test.shape[0] - X_test.shape[0]

"""
if __name__ == '__main__':
    
    LSTM1 = LSTM_classifier()
    LSTM1.Train()
    
"""    
    
    #initialize model by creating a instant of class
    
    








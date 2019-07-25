#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:33:22 2019

@author: Amith R

LSTM KERAS implementation
"""

import tensorflow as tf
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
import time
import collections
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
        self.curr_buff_len = 0
        self.y_train_path = self.DATASET_PATH + "Y_train.txt"
        self.y_test_path = self.DATASET_PATH + "Y_test.txt"
        self.LSTM_BUFFER_LEN = 32
        self.de = collections.deque([],32)
        
    def init_graph(self):
        
        """ LSTM IMPLEMENTATION  """
        self.model = models.Sequential()
        self.model.add(Dense(self.n_hidden,activation='relu',input_shape=(self.timesteps,self.data_dim)))
        self.model.add(LSTM(self.n_hidden,return_sequences = True))
        self.model.add(LSTM(self.n_hidden,input_shape=(self.timesteps,self.data_dim)))
        self.model.add(Dense(self.num_classes,activation = 'softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
    
    def load_X(self,X_path):
            
            file = open(X_path, 'r')
            X_ = np.array(
                    [elem for elem in [
                            row.split(',') for row in file
                            ]], 
                            dtype=np.float32
                            )   
            file.close()
            blocks = int(len(X_) / self.n_steps)
    
            X_ = np.array(np.split(X_,blocks))

            return X_ 

    def load_y(self,y_path):
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
    
    def normalise(self,X):
        X_n = []
        n_steps = 32
        k=0
        for i in X:
            k +=1
            for j in X[k-1,:,:]:
                norm = np.interp(j,(0,500),(0,1))
                X_n.append(norm)
        blocks = int(len(X_n)/ n_steps)
        X_n = np.array(X_n)        
        X_n = np.array(np.split(X_n,blocks))
 
        return X_n
    
    
    
    
    
    def Train(self):
        
        # NO ARGUEMENTS REQUIRED WHEN CALLED ,PATHS INITIALIZED WHEN CLASS object is Created
        # PUT THE TRAINING IN THE SPECIFIED PATH 
        """ TRAINING OVER 100 EPOCHS """
        X_train = self.load_X(self.X_train_path)
        X_test = self.load_X(self.X_test_path)
        X_train_norm= self.normalise(X_train)
        X_test_norm = self.normalise(X_test)
        y_train = self.load_y(self.y_train_path)
        y_test = self.load_y(self.y_test_path)

        # DATA LOADED INTO LISTS  x_TRAIN, Y_TRAIN , X_TEST , Y_TEST
        #training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
        #test_data_count = len(X_test)  # 1197 test series
        #n_input = len(X_train[0][0])  # num input parameters per timestep
        
        y_train_one_hot = keras.utils.to_categorical(y_train,6)
        y_test_one_hot = keras.utils.to_categorical(y_test,6)
        
        history = self.model.fit(
                X_train_norm,
                y_train_one_hot,
                epochs=self.training_epochs,
                batch_size = self.batch_size,
                verbose = 1,
                validation_data=(X_test_norm,y_test_one_hot),
                callbacks=[keras.callbacks.ModelCheckpoint(self.filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                           keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
        # PLOT a LOSS CURVE and convergence per epoch
        plt.figure()
        plt.title('Training performance')
        plt.plot(history.epoch, history.history['loss'], label='train loss+error')
        plt.plot(history.epoch, history.history['val_loss'], label='val_error')
        plt.legend()
        self.model.load_weights(self.filepath)
           
        
    def Predict(self,skeleton_data):
        #arguement skeleton_data(32,36) BEFORE
        # arguement skelton_data(36,) NOW !!!!!!!!!!!!!!!!
        self.model.load_weights(self.filepath)
        # CREATE BUFFER
        #print(skeleton_data)
        
        if(self.curr_buff_len <=32):
            self.de.appendleft(skeleton_data)
            self.curr_buff_len += 1
            return None
        elif(self.curr_buff_len > 32):
            self.de.pop()    # changed from popleft to pop 
            self.de.appendleft(skeleton_data)
            skel = list(self.de)
            k1 = np.expand_dims(skel,axis=0)
            print(np.shape(k1))
            Y_pred = self.model.predict(k1)  # shape (1,32,36) that is supposed to be given
            return Y_pred 
        

# excecute 
"""   
LABELS = [    
   

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

if __name__ == '__main__':
    #initialize model by creating a instant of class
    LSTM1 = LSTM_classifier()
    skelton_data_path = LSTM1.X_test_path
    LSTM1.Train()
    X = LSTM1.load_X(skelton_data_path)
    X_norm = LSTM1.normalise(X)
    k = X[0,:,:]
    type(k)
    """
    de = collections.deque(k)
    de.append(X[0,0,:])
    de.appendleft(X[0,0,:])
    de.pop()
    print(de)
    de.popleft()"""
     # SET OF EXPERIMENTS DONE FOR FINDING OUT DIMENSIONS
    j = X_norm[0,0,:]
    print(np.shape(j))
    k1 = np.expand_dims(j,axis=0)
    #pred = self.model.
    t1= time.time()
    for i in range(30):
        pred2=LSTM1.Predict(j)
    
    t1=time.time()
    pred2 = LSTM1.Predict(j)
    print(np.shape(j))
    trans = np.transpose(j,(0,1))
    
    pred=model.predict(k)   
    t2= time.time()
    print(t2-t1)
    #returns a vector of probabilities .
    print(pred2)
    m = np.array([[1],[2],[3]])
    if m is not None:
        print("NONE")
    n = np.array([1,2,3])
    o = np.hstack(m)
    p = np.array([[1,2,3]])
    q = np.hstack(p)
    
    """
    #IMPLEMENT DEQUE FOR POP AND PUSH OPERATIONS AT DIFFERENT ENDS
    de = collections.deque([1,2,3])
    de.append(4)
    
    de.appendleft(6)
  
    de.pop() 
    de.popleft()
     
    print(de)
    """
    j = X_norm[3,0,:]
    skeleton_data = X_norm[0,0,:]
    de = collections.deque([],32)
    curr_buff_len = 0
    
    for i in range(30):
        if(curr_buff_len <=32):
            de.appendleft(j)
            curr_buff_len += 1
            skel = list(de)
            print(skel)
        elif(curr_buff_len > 32):
            de.pop()
            de.appendleft(j)
            skel = list(de)
            k1 = np.expand_dims(skel,axis=0)
            print(np.shape(k1))
        
    
    
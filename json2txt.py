#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:32:16 2019

@author: amith
JSON to TXT 
"""

import os,json
import pandas as pd
import funcs as myfunc
import operator
# SORTED used to sort in the right order
path_to_json = '/home/amith/openpose/examples/media/Captured/JSON/cropped_2'
json_files = [pos_json for pos_json in sorted(os.listdir(path_to_json)) if pos_json.endswith('.json')]
#print(json_files)

# using pandas DataFrame.from_dict to read from json ie python dictionary to a pandas dtaframe
# initialize pandas dataframe with the columns required
jsons_data = []
for index,js in enumerate(json_files):
    #print(index)
    with open(os.path.join(path_to_json , js)) as json_file:
        json_text = json.load(json_file)
        jsons_data.insert(index,([pd.DataFrame(columns =['pose_keypoints_2d']) for j in range(len(json_text['people']))]))
                        
#jsons_data = pd.DataFrame(columns = ['pose_keypoints_2d'])
keypoints_all=[]
for index, js in enumerate(json_files):
    keypoints = []
    with open(os.path.join(path_to_json , js)) as json_file:
        json_text = json.load(json_file)
        #print(len(json_text['people']))
        intermediate_k = []
        for i in range(len(json_text['people'])):
            keypoints.append(json_text['people'][i]['pose_keypoints_2d'])
            jsons_data[index][i].loc[index] = [keypoints[i]]
            intermediate_k.append(keypoints[i])
            
        keypoints_all.append(intermediate_k)
       
# DataFrames now created , the keypoints are shuffled , need to be tracked
# extracting only the keypoints
for i in range(len(keypoints_all)):
    for j in range(len(keypoints_all[i])):
        new_list=[]
        for k in range(len(keypoints_all[i][j])):
            if((k+1)%3 != 0):
                new_list.append(keypoints_all[i][j][k])
            
        keypoints_all[i][j] = new_list
 

tracker  = myfunc.Tracker()
ID = []
for i in range(len(keypoints_all)):
    ID.append( tracker.track(keypoints_all[i]))
   
id_now = tracker.track(keypoints_all[86])
len(ID)
len(keypoints_all)
len(keypoints_all[0])
ID_fine = []
 # list of lists of keypoints without order 
for i in range(len(ID)):
    sorted_ID = []
    sorted_ID = sorted(ID[i].items() , key = operator.itemgetter(0) )
    ID_fine.append(sorted_ID)
ID_unfine = []
   
for i in range(len(ID)):
    unsorted_ID = []
    for j in range(len(ID[i])):
        unsorted_ID.append(ID[i][j+1])
    ID_unfine.append(unsorted_ID)
        
#result = tracker.match_features(keypoints_all[9],keypoints_all[10])       
# Now TRACK AND ALLOCATE 
# keypoints_all contains all lists in a random order, order to be arranged
# ID1 in 1st position
# ID is stored in the order of increasinf X of neck keypoint
# smallest X implies the first position in the first frame
    """
NECK = 3
prev = list.copy(keypoints_all[0][0])

i=10      
for i in range(len(keypoints_all)):
    dist_matrix = []  # distance matrix for all frames
    for j in range(len(keypoints_all[i])):
        column = []
        for k in range(len(keypoints_all[i])):
            column.append(float(0))
        dist_matrix.append(column)  # DISTANCE MATRIX CREATED
            
    for j in range(len(keypoints_all[i])):
        new_list = []
        body_part = 0
        if(i != 0):
            for (present,prev) in zip(keypoints_all[i][j] , keypoints_all[i-1][j]):
                body_part = body_part + 1
                if(body_part-1 == NECK):
                    dist_matrix[j][j] = float("{0:.8f}".format(abs(present - prev)))
        
            for k in range(len(keypoints_all[i])):
                for l in range(len(keypoints_all[i])):
                    present = keypoints_all[i][k][NECK] - keypoints_all[i-1][k][NECK] 
                    dist_matrix[k][l] =  float("{0:.8f}".format(abs(present - prev)))
                    
            # now evaluate distance matrix and change order
            
                    
                
            
        
        for (present,prev) in zip(keypoints_all[i][j] , keypoints_all[i-1][j]):
            if( j == NECK

            
            
        
        

# the definition should return the position to be swapped        
def evaluate_matrix(dist_matrix):
    
    
M = [[1,2],[3,5]]
len(M)
M.index(min([1,2]))
      
     """           
                
           
            
            
        

    
            


            
        
        


        

""" 
keypoints1 = json_text['people'][0]['pose_keypoints_2d']
        keypoints2 = json_text['people'][1]['pose_keypoints_2d']
        # call a function to right assign
        jsons_data1.loc[index] = [keypoints1]
        jsons_data2.loc[index] = [keypoints2]"""
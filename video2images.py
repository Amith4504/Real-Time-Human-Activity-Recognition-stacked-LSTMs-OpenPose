
import cv2
import numpy as np
import sys, os
from time import sleep
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

paths = [
    # (CURR_PATH + "../../data_test/", "exercise",".webm"),
    # (CURR_PATH + "../../data_test/", "apple",".mp4"),
    # (CURR_PATH + "../../data_test/", "exercise2",".webm"),
    # (CURR_PATH + "../../data_test/", "walk-stand",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-1",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-2",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-stand-3",".avi"),
    # (CURR_PATH + "../../data_test/", "walk-1",".avi"),
    # (CURR_PATH + "../../data_test/", "sit-1",".avi"),
    (CURR_PATH, "video_1",".mp4"),
    #(CURR_PATH, "video_2",".mp4"),
    #(CURR_PATH, "video_3",".mp4"),
    #(CURR_PATH, "video_4",".mp4"),
    #(CURR_PATH, "video_5",".mp4"),
    #(CURR_PATH, "video_6",".mp4"),
    #(CURR_PATH, "video_7",".mp4"),
    #(CURR_PATH, "video_8",".mp4"),
    #(CURR_PATH, "video_9",".mp4")
]

# -- Input
idx = 0
s_folder =  paths[idx][0]
s_video_name_only =  paths[idx][1]
s_video = s_video_name_only+ paths[idx][2] 

# -- Output
s_save_to_folder = s_folder + s_video_name_only + "/"
if not os.path.exists(s_save_to_folder):
    os.makedirs(s_save_to_folder)

# -- Functions
int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

# -- Read video
cap = cv2.VideoCapture(s_folder + s_video)
count = 0
def getFrame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    ret,frame = cap.read()
    write_to = s_save_to_folder + int2str(count, 5) + ".png"
    if ret:
        cv2.imwrite(write_to, frame)
    return ret
     
    

sec=0
framerate= 0.5	

#while(cap.isOpened()):
   
ret = getFrame(sec)
    # if ret == False:
    #     break
while ret:

   count += 1
   sec = sec + framerate
   sec = round(sec,2)
   ret = getFrame(sec)    
    
    # if count % 100 ==0:
    #    print('Read a new frame {} of size {} to {}'.format( count, frame.shape, write_to))
    # if count < 1000:
    #     continue
    # if count > 2000:
    #     break

    # Show and save
    # if count % 10 ==0:
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        # save frame as JPEG file      


cap.release()
cv2.destroyAllWindows()

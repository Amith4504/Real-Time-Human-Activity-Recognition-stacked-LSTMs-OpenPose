# Real-Time-Human-Activity-Recognition-stacked-LSTMs-OPENPOSE-
This project is based on work done by felixchenfy ( Real Time Action Recognition ) and stuarteiffert ( RNN for Real Time Action recognition using 2D pose input)  . The Real time interface that is created by the former creator is combined with the classification architecture (stacked LSTMs) done by  latter creator. 

The work i have done is create a new class LSTM_classifier with Train and Predict methods which requires manual changes in the code to excecute in contrast to the work done by felixchenfy . 

## LSTM ARCHITECTURE 
We have used 2 layer Stacked many-to-one LSTM architecture for feature extraction with 32 time steps for capturing time dependency . The intuition behind this architecture is that the network makes a decision (classification) based on the keypoints outputted by [OPENPOSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose) / [TFPOSE](https://github.com/ildoonet/tf-pose-estimation) of the present and previous 31 frames. 32 frames correspond to frames captured in the time interval of 1.5 seconds. Many to one architecture is as given below:

![Alt text](https://i.stack.imgur.com/QCnpU.jpg) 

The network has to be trained to take 32 frame Openpose coordinates as input (   ([0,1] normalised) and give a ouput of scores (softmax activation). Thus during real time activity recognition 32 frames has to be stored in a buffer and updated with every new frame. Thus faster the processing power faster and more accurate the detection. The classification accuracy does depend on the processing power on which it is running .  


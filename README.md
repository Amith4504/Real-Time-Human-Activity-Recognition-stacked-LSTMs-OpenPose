# Real-Time-Human-Activity-Recognition-stacked-LSTMs-OPENPOSE-
This project is based on work done by felixchenfy ( Real Time Action Recognition ) and stuarteiffert ( RNN for Real Time Action recognition using 2D pose input)  . The Real time interface that is created by the former creator is combined with the classification architecture (stacked LSTMs) done by  latter creator. 

The work i have done is create a new class LSTM_classifier with Train and Predict methods which requires manual changes in the code to excecute in contrast to the work done by felixchenfy . 

## LSTM ARCHITECTURE 
We have used 2 layer Stacked LSTM architecture for feature extraction with 32 time steps for capturing time dependency . The intuition behind this architecture is the network makes a decision (classification) based on the keypoints outputted by OPENPOSE of the present and previous 31 frames.


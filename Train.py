
import mylib.feature_proc as myproc
import mylib.action_classifier as myclf
import mylib.displays as mydisp
import importlib

import pandas as pd
import simplejson
import numpy as np
import math
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import simplejson
RAND_SEED = 1

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# load data
def encode_Y(Y_str):
    # One-hot encoder. e.g: Y_one_hot = [1,0,0,0], [0,1,0,0], ...
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(Y_str)
    # labels = enc.categories_[0].tolist()
    labels = enc.categories_[0]
    print("labels: ", labels)
    Y_one_hot = enc.transform(Y_str).toarray()
    
    # Get label index. e.g: Y = 0, 1, 2, ...
    Y = [np.where(yi == 1)[0][0] for yi in Y_one_hot]
    return Y, Y_one_hot, labels

def remove_incomplete_sample(X0, NaN):
    left, right = 0, 14*2 # only check 14 data, which are: head(1) + neck(1) + 2 * (arm(3) + leg(3))
    valid_indices = []
    for i in range(X0.shape[0]):
        if len(np.where(X0[i,left:right] == NaN)[0]) == 0:
            valid_indices.append(i)
    return valid_indices


def load_my_data(filepath):
    with open(filepath, 'r') as f:
        dataset = simplejson.load(f)
        
        # --------------------------->
        # Added in 2019/06/05: If a data is bad, remove it
        tmp = []
        for i, d in enumerate(dataset):
            if d[0] == 0:
                print(f"Bad data at {i}, where data={d}")
            else:
                tmp.append(d)
        dataset = tmp
        # <---------------------------
        
        X = np.array([r[5:5+36] for r in dataset])
        clip_indices = [r[1] for r in dataset]
        Y_str = [[r[3]] for r in dataset]
        Y, Y_one_hot, labels = encode_Y(Y_str)
        print("Num samples = ", len(Y))
        
        if 0:
            valid_indices = remove_incomplete_sample(X, NaN=0)
            X = X[valid_indices, :]
            Y = [Y[i] for i in valid_indices]
            clip_indices = [clip_indices[i] for i in valid_indices]
            print("Num samples after removal = ", len(Y))
        
        return X, Y, clip_indices, labels
    print("my Error: loading skeletons_info.txt failed.")
    return None, None

def split_data(X, Y):
    if 1:
        tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, test_size=0.3, random_state=RAND_SEED)
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    print("Size of X:", tr_X.shape)
    print("Num training: ", len(tr_Y))
    print("Num testing:  ", len(te_Y))
    return tr_X, te_X, tr_Y, te_Y

# Process features
def extract_time_serials_data(X, Y, clip_indices, config_add_noise):
    X_new = []
    Y_new = []

    # Loop through all data
    for i, _ in enumerate(clip_indices):

        # If a new video clip starts, reset the feature generator
        if i == 0 or clip_indices[i] != clip_indices[i-1]:
            fg = myproc.FeatureGenerator(config_add_noise)
        
        # Get features
        success, features = fg.add_curr_skeleton(X[i,:])
        if success: # True if (data length > 5) and (skeleton has enough joints) 
            X_new.append(features)
            Y_new.append(Y[i])

    X_new = np.array(X_new)
    return X_new, Y_new

X0, Y0, clip_indices, classes = load_my_data('skeleton_data/skeletons5_info.txt')

# Get time serials data
importlib.reload(myproc)
X1, Y1 = extract_time_serials_data(X0, Y0, clip_indices, config_add_noise=True)
X2, Y2 = extract_time_serials_data(X0, Y0, clip_indices, config_add_noise=False)
X = np.vstack((X1, X2))
Y = np.concatenate((Y1, Y2))
print("After extract time serials:", "X.shape = ", X.shape, ", len(Y) = ", len(Y))
tr_X, te_X, tr_Y, te_Y = split_data(X, Y)

importlib.reload(myclf)
model = myclf.ClassifierOfflineTrain()
# CHANGE HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#model.train(tr_X, tr_Y)
model.train_RNN()

importlib.reload(mydisp)
import time

t0 = time.time()

tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)

print( "Time cost for predicting one sample:", (time.time() - t0) / len(Y) )

# axis, cf = mydisp.plot_confusion_matrix(tr_Y, tr_Y_predict, classes, normalize=True)
axis, cf = mydisp.plot_confusion_matrix(te_Y, te_Y_predict, classes, normalize=False, size=(10, 6))

from sklearn.metrics import classification_report
print(classification_report(te_Y, te_Y_predict, target_names=classes))

if model.model_name=="Neural Net":
    
    # Save trained model to file
    import pickle
    path_to_save_model = '../model/trained_classifier.pickle'
    
    with open(path_to_save_model, 'wb') as f:
        pickle.dump(model, f)

    if 1:
        # Load and test again to ensure correctly saved to file
        with open(path_to_save_model, 'rb') as f:
            model2 = pickle.load(f)
        print(tr_X.shape)
        model2.predict_and_evaluate(tr_X, tr_Y)
        model2.predict_and_evaluate(te_X, te_Y)
        print("OK, model is saved to disk. I can test it on webcam")







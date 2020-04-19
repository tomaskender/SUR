import numpy as np
#from sklearn.mixture import gmm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from python_speech_features import mfcc

def extract_features(audio, rate):
    mfcc_feature = mfcc(audio,rate, winlen=0.025, winstep=0.01, numcep=20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def build_model(train_x, train_y, test_x, test_y):
    #model = GMM(n_components=16, n_iter=500, convariance_type='diag')
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    print("ACCURACY SCORE: ", accuracy_score(model.predict(test_x), test_y))

def evaluate(verbose = 0):
    pass
import numpy as np
from scipy import stats
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc, delta
import pickle, re, glob
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy.io import wavfile

def extract_features(rate, audio):
    #assert(rate == 16000)
    mfcc_feature = mfcc(audio,rate, winlen=0.025, winstep=0.01, numcep=20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    mfcc_feature = stats.zscore(mfcc_feature)

    delta_f = delta(mfcc_feature, 2)
    d_delta_f = delta(delta_f, 2)
    combined = np.hstack((mfcc_feature,delta_f, d_delta_f)) 
    return combined

def build_model(train_X, train_y, test_X, test_y, verbose=0):
    data_X = np.concatenate((train_X, test_X))
    data_y = np.concatenate((train_y, test_y))
    data_X, data_y = shuffle(data_X, data_y)

    features = []
    features_y = []
    for index in range(len(data_X)):
        vector = extract_features(data_X[index][0], data_X[index][1])
        
        features.append(vector)
        features_y.append(np.array([data_y[index]] * len(vector)))
    features = np.vstack(features)
    features_y = np.hstack(features_y)

    #features = SelectKBest(f_classif, k=20).fit_transform(features, features_y)
    #https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
    model = RandomForestClassifier(n_estimators=96, max_depth=96, verbose=verbose)
    train_X, test_X, train_y, test_y = train_test_split(features, features_y, test_size=0.15)
    model.fit(train_X, train_y)
    print("ACCURACY SCORE: ", accuracy_score(model.predict(test_X), test_y))
    print("Voice model saved as modelRandomForest.pkl")
    
    with open('src/modelRandomForest.pkl', 'wb') as file:
        pickle.dump(model, file)

def evaluate(verbose = 0):
    with open('src/modelRandomForest.pkl', 'rb') as modelfile:
    	model = pickle.load(modelfile)

    results = ''
    for file in glob.glob('data/eval/*.wav'):
        if verbose:
            print('Probing ' + file + '..\n')
        voice = wavfile.read(file)
        feats = extract_features(voice[0], voice[1])
        prob = model.predict_proba(feats)
        mean = np.mean([x[1] for x in prob])
        if verbose:
            print('I\'m ' + str(mean)+' sure this record is the target.\n')
        results += "{} {} {}\n".format(file, mean, int(mean >= 0.5))

    with open('voice_RandomForest.txt', 'w') as output:
	    output.write(results)
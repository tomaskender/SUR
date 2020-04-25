import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle, re, glob
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy.io import wavfile
from librosa.feature import mfcc, delta, spectral_centroid
from librosa.effects import trim
from librosa.core import load as load_audio, piptrack as pitch
from pysndfx import AudioEffectsChain

def reduce_noise_power(y, sr):

    cent = spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean

def extract_features(audio, rate):
    audio = reduce_noise_power(audio, rate)

    audio, indexes = trim(audio)

    mfcc_feature = mfcc(y=audio,sr=rate, n_mfcc=13, n_fft=int(0.025*rate), n_mels=40, fmin=20, hop_length=int(0.03*rate))

    mfcc_feature = preprocessing.scale(mfcc_feature, axis=1)

    mfcc_feature = stats.zscore(mfcc_feature)

    pitches, magnitudes = pitch(y=audio, sr=rate, fmin=50, fmax=400, n_fft=int(0.025*rate), hop_length=int(0.03*rate))

    #delta_f = delta(mfcc_feature)
    #d_delta_f = delta(mfcc_feature, order=2)
    combined = np.hstack((np.transpose(mfcc_feature), np.transpose(pitches)))
    return combined

def build_model(train_X, train_y, test_X, test_y, verbose=0):
    #train model on data in all folders
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

    # output to txt
    results = ''
    # count matches
    cntr = 0
    for file in glob.glob('eval/*.wav'):
        voice = load_audio(file)
        file = file.split('/', 1)[1] # remove eval/ from filename
        if verbose:
            print('Probing ' + file + '..\n')
        feats = extract_features(voice[0], voice[1])
        prob = model.predict_proba(feats)
        # get mean value of score for each frame
        mean = np.mean([x[1] for x in prob])
        if mean > 0.14:
            cntr += 1
        if verbose:
            print('I\'m ' + str(mean)+' sure this record is the target.\n')
        results += "{} {} {}\n".format(file, mean, int(mean > 0.14))

    if verbose:
        print('I\'ve found '+ str(cntr) + ' matches.\n')
    with open('voice_RandomForest.txt', 'w') as output:
	    output.write(results)
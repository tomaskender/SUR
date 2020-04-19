import os, glob, sys  # for managing directory paths and files
import numpy as np
from PIL import Image
from scipy.io import wavfile
import face_detector as face
import voice_detector as voice
import argparse

PARENT_DIR = os.getcwd()
DATA_DIR = PARENT_DIR + "/data/"
USE_CNN = False
PLOTS = False
VERBOSE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--cnn", action="store_true")
parser.add_argument("--evalonly", action="store_true")
parser.add_argument("--trainonly",  action="store_true")
parser.add_argument("--verbose",  action="store_true")
parser.add_argument("--plot",  action="store_true")
parser.add_argument("--system")
arguments = parser.parse_args()

if arguments.evalonly and arguments.trainonly:
	sys.stderr.write("ERROR: use either '--evalonly' or '--trainonly'!\n")
	sys.exit(0)
if arguments.cnn:
	USE_CNN = True
if arguments.plot:
	PLOTS = True
if arguments.verbose:
	VERBOSE = 1


def get_image_file(file):
	return [np.array(Image.open(f).convert('RGB')) for f in glob.glob(file)]

def get_audio_file(file):
    return [wavfile.read(f) for f in glob.glob(file)]

# loads images and concatenates train and test data
def image_data():
	test_x = np.array(get_image_file(DATA_DIR + 'target_dev/*.png'))
	test_y = np.array([1] * len(test_x))  # 1 is our wanted face
	t = np.array(get_image_file(DATA_DIR + 'non_target_dev/*.png'))
	test_y = np.concatenate((test_y, np.array([0] * len(t))), axis=0)
	test_x = np.concatenate((test_x, t), axis=0)

	train_x = np.array(get_image_file(DATA_DIR + 'target_train/*.png'))
	train_y = np.array([1] * len(train_x))  # 1 is our wanted face
	t = np.array(get_image_file(DATA_DIR + 'non_target_train/*.png'))
	train_y = np.concatenate((train_y, np.array([0] * len(t))), axis=0)
	train_x = np.concatenate((train_x, t), axis=0)

	return train_x, train_y, test_x, test_y

# loads voices and concatenates train and test data
def voice_data():
	test_x = np.array(get_audio_file(DATA_DIR + 'target_dev/*.wav'))
	test_y = np.array([1] * len(test_x))  # 1 is our wanted voice
	t = np.array(get_audio_file(DATA_DIR + 'non_target_dev/*.wav'))
	test_y = np.concatenate((test_y, np.array([0] * len(t))), axis=0)
	test_x = np.concatenate((test_x, t), axis=0)

	train_x = np.array(get_audio_file(DATA_DIR + 'target_train/*.wav'))
	train_y = np.array([1] * len(train_x))  # 1 is our wanted voice
	t = np.array(get_audio_file(DATA_DIR + 'non_target_train/*.wav'))
	train_y = np.concatenate((train_y, np.array([0] * len(t))), axis=0)
	train_x = np.concatenate((train_x, t), axis=0)

	return train_x, train_y, test_x, test_y

if arguments.system == 'face':
	if not arguments.evalonly:
		train_x, train_y, test_x, test_y = image_data()
		face.build_model(train_x, train_y, test_x, test_y, USE_CNN=USE_CNN, PLOTS=PLOTS)
	if not arguments.trainonly:
		face.evaluate(cnn=USE_CNN, verbose=VERBOSE)
elif arguments.system == 'voice':
	if not arguments.evalonly:
		train_x, train_y, test_x, test_y = voice_data()
		voice.build_model(train_x, train_y, test_x, test_y)
	if not arguments.trainonly:
		voice.evaluate(verbose=VERBOSE)
else:
	sys.stderr.write("ERROR: Invalid system argument!\n")
	sys.exit(0)

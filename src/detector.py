import os, glob, sys  # for managing directory paths and files
import numpy as np
from PIL import Image
import face_detector as face
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


if not arguments.evalonly:
	train_x, train_y, test_x, test_y = image_data()
	face.build_model(train_x, train_y, test_x, test_y, USE_CNN=USE_CNN, PLOTS=PLOTS)
if not arguments.trainonly:
	face.evaluate(cnn=USE_CNN, verbose=VERBOSE)


def print_result(sample_name, loss, is_target):
	"""Print result of sample analysis to stdout."""
	print("{} {} {}\n".format(sample_name, loss, is_target))

print("test", 1.0, True)

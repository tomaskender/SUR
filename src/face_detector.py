import numpy as np
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from keras import optimizers, initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pickle, re, glob

BATCH_SIZE = 500
EPOCHS = 10
SIZE = 80

# not in use
def extract_face(image_list, required_size=(SIZE, SIZE)):  # size of given examples
	final_image_list = []
	for image in image_list:
		pixels = np.asarray(image)
		detector = MTCNN()  # using default weights
		results = detector.detect_faces(pixels)
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)  # bug fix
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]  # extract the face
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = np.asarray(image)
		final_image_list.append(face_array)
	return final_image_list


def resize_img(images):
	final_image_list = []
	for image in images:
		image = resize(image, (SIZE, SIZE, 3))
		final_image_list.append(np.asarray(image))
	return final_image_list


# does not work unfortunately :(
def build_CNN():
	modelCNN = Sequential()
	modelCNN.add(Conv2D(4, (14, 14),
					 activation='relu',
					 input_shape=(SIZE, SIZE, 3),
					 bias_initializer=initializers.constant(0.1),
					 padding='same'))
	modelCNN.add(MaxPooling2D(pool_size=(2, 2)))  # pooling
	modelCNN.add(Conv2D(8, (8, 8),  # 2nd Convolution layer with 64 channels
					 activation='relu',
					 bias_initializer=initializers.constant(0.1)))
	modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
	# Flattening, turning to 1D
	modelCNN.add(Dropout(0.25))
	modelCNN.add(Flatten())
	modelCNN.add(Dense(256, activation='relu', bias_initializer=initializers.constant(0.1)))
	modelCNN.add(Dropout(0.3))
	modelCNN.add(Dense(1, activation='sigmoid'))
	sgd = optimizers.SGD(lr=0.01, momentum=0.9)
	modelCNN.compile(loss='binary_crossentropy',  # loss function used for classes that are greater than 2)
				  optimizer=sgd,
				  metrics=['accuracy'])
	return modelCNN


def train(train_x, train_y, test_x, test_y, modelCNN, PLOTS):
	hist = modelCNN.fit(train_x, train_y,
					 batch_size=BATCH_SIZE,
					 epochs=EPOCHS,
					 validation_data=(test_x, test_y),
					 shuffle=True, verbose=1)
	score = modelCNN.evaluate(test_x, test_y)
	print('TEST LOSS:', score[0])
	print('ACCURACY SCORE:', score[1])

	if PLOTS:
		pyplot.plot(hist.history['accuracy'])
		pyplot.plot(hist.history['val_accuracy'])
		pyplot.title('modelCNN accuracy')
		pyplot.ylabel('Accuracy')
		pyplot.xlabel('Epoch')
		pyplot.legend(['Train', 'Val'], loc='upper left')
		pyplot.show()

		pyplot.plot(hist.history['loss'])
		pyplot.plot(hist.history['val_loss'])
		pyplot.title('modelCNN loss')
		pyplot.ylabel('Loss')
		pyplot.xlabel('Epoch')
		pyplot.legend(['Train', 'Val'], loc='upper right')
		pyplot.show()
	return modelCNN


def build_model(train_x, train_y, test_x, test_y, USE_CNN = False, PLOTS = False):
	# shuffles data with 10! permutations
	train_x, train_y = shuffle(train_x, train_y, random_state=10)
	test_x, test_y = shuffle(test_x, test_y, random_state=10)
	resized_train = resize_img(train_x)
	resized_test = resize_img(test_x)
	# concatenate all the input data
	all_x = np.concatenate((resized_train, resized_test), axis=0)
	all_y = np.concatenate((train_y, test_y), axis=0)
	# process modification modelCNN, to get more data for training
	train_datagen = ImageDataGenerator(shear_range=0.2,
									   width_shift_range=0.1,
									   height_shift_range=0.1,
									   rotation_range=10,
									   horizontal_flip=True)
	batch_x, batch_y = [], []
	batches = 0
	# apply our generated modifications to data
	for gen_x, gen_y in train_datagen.flow(all_x, all_y, batch_size=2000):
		batch_x.append(gen_x)
		batch_y.append(gen_y)
		batches += 1
		if batches > 30:  # need to manually break the loop
			break
	all_x = np.vstack(batch_x)
	all_y = np.hstack(batch_y)

	if USE_CNN:
		modelCNN = build_CNN()
		train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.2)
		modelCNN = train(train_x, train_y, test_x, test_y, modelCNN, PLOTS=PLOTS)
		modelCNN.save('src/modelCNN.h5')
	else:
		all_x = all_x.reshape(all_x.shape[0], -1)
		modelSGD = SGDClassifier(loss='modified_huber')
		if PLOTS:
			from sklearn.model_selection import KFold
			from sklearn.model_selection import cross_val_score
			cross_val = cross_val_score(modelSGD, all_x, all_y, cv=KFold(n_splits=4,random_state=5))
			print('FOLDS: ', cross_val)
			print('MEAN: ', cross_val.mean(), 'STD: ', cross_val.std())
			cross_val *= 100
			base = min((85, min(cross_val) - 1))
			fig = pyplot.figure()
			ax = fig.add_subplot(111)
			ax.bar(['FOLD ' + str(i + 1) for i in range(len(cross_val))], cross_val - base, bottom=85)
			ax.axhline(y=cross_val.mean())
			ax.set_title('VALIDATION SUMMARY')
			ax.set_ylabel('ACCURACY')
			ax.set_yticks(np.arange(base, 100, 1))
			pyplot.show()
		train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.1)
		modelSGD.fit(train_x, train_y)
		print("ACCURACY SCORE: ", accuracy_score(modelSGD.predict(test_x), test_y))
		print("SGD modelCNN saved as modelSGD.pkl")
		with open('src/modelSGD.pkl', 'wb') as f:
			pickle.dump(modelSGD, f)

def evaluate(cnn = False, verbose = 0):
	if cnn == True:
		if verbose == 1: print("Using CNN model")
		cnn = True
		model = load_model('src/modelCNN.h5')
	else:
		if verbose == 1: print("Using SGD model")
		with open('src/modelSGD.pkl', 'rb') as file:
			model = pickle.load(file)

	eval_images =  []
	results = ''
	for filename in glob.glob('eval/*.png'):
		eval_images.append(filename)
	eval_images.sort(key=lambda f: int(re.sub('\D', '', f)))
	for eval_img in eval_images:
		img = Image.open(eval_img).convert('RGB')
		if cnn == True:
			image = np.array(img)
			image = image.reshape(-1, SIZE, SIZE, 3)
			prob = model.predict_proba(image)
			hard_decision = int(0.5 <= prob[0][0])
			prob = prob[0][0]
		else:
			image = np.array(img).flatten()
			prob = model.predict_proba([image])
			hard_decision = int(prob[0][0] <= prob[0][1])
			prob = prob[0][1]
		result = "{} {} {}\n".format(eval_img[5:-4], prob , hard_decision)
		if verbose == 1: print(result, end='')
		results += result

	if cnn == True: eval_file = "images_CNN.txt"
	else: eval_file = "images_SGD.txt"
	f = open(eval_file, 'w')
	f.write(results)  # python will convert \n to os.linesep
	f.close()

# 	img = Image.open("data/non_target_train/m416_02_r08_i0_0.png").convert('RGB')
# 	image = np.array(img).flatten()
# 	prob = modelSGD.predict_proba([image])
# 	print(prob)
# 	# print(modelSGD.predict([image]))
# 	if cnn:
# 		imCNN = np.array(img)
# 		imCNN = imCNN.reshape(-1, SIZE, SIZE, 3)
# 		prob = modelCNN.predict_proba(imCNN)
# 		print(prob)
# 		print(modelCNN.predict(imCNN))
# 		print(modelCNN.predict_classes(imCNN))
#
# 	img2 = Image.open("data/target_train/m429_01_p02_i0_0.png").convert('RGB')
# 	image2 = np.array(img2).flatten()
# 	proba2 = modelSGD.predict_proba([image2])
# 	print(proba2)
# 	# print(modelSGD.predict([image2]))
# 	if cnn:
# 		imCNN2 = np.array(img2)
# 		imCNN2 = imCNN2.reshape(-1, SIZE, SIZE, 3)
# 		proba2 = modelCNN.predict_proba(imCNN2)
# 		print("DRUHY:", proba2)
# 		print(modelCNN.predict(imCNN2))
# 		print(modelCNN.predict_classes(imCNN2))
# # return modelCNN

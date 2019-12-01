from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
#from imutils import paths


os.environ['KMP_WARNINGS'] = '0'
#disable above if you need to see tensorflow start logs




#creates and organizes paths
#lenTest=len(list(paths.list_images('data/test')))

datagen = ImageDataGenerator(
	rotation_range=180, 
	width_shift_range=0.2, 
	height_shift_range=0.2, 
	shear_range=0.2,
	#rescale = 1/255, 
	zoom_range=0.2, 
	horizontal_flip=True, 
	vertical_flip=True, 
	#validation_split=0.2, 
	fill_mode='nearest')

img = load_img('data/train/cats/cat.1.jpg')

imgArray = img_to_array(img)
imgArray = imgArray.reshape((1,) + imgArray.shape)
print(imgArray.shape)

path = 'imgPreview'
if os.path.exists(path) == False:
	print(f'{path} does not exist')
	try:
		os.mkdir(path)
		print(f'{path} has been made')
	except OSError:
		print(f'Directory {path} could not be made')
else:
	print(f'{path} already exists')


i = 0
for batch in datagen.flow(imgArray, batch_size=1, save_to_dir=path, save_prefix='cats', save_format='jpeg'):
	i += 1

	if i > 20:
		break

model = Sequential()

shape=(150, 150, 3)
channelDim = -1

if K.image_data_format() == 'channels_first':
	shape = (depth, height, width)
	channelDim = 1

model.add(SeparableConv2D(32, (3, 3), padding = 'same', input_shape = shape))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = channelDim))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

model.add(SeparableConv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = channelDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = channelDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


batch_size = 16

train_Datagen = ImageDataGenerator(
	rescale=1/255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

valAug = ImageDataGenerator(rescale=1/255)

train_generator = train_Datagen.flow_from_directory(
	'data/train',
	target_size=(150, 150),
	color_mode='rgb',
	shuffle=False,
	batch_size=batch_size,
	class_mode='binary')

validation_generator = valAug.flow_from_directory(
	'data/validation',
	target_size=(150, 150),
	color_mode='rgb',
	shuffle=False,
	batch_size=batch_size,
	class_mode='binary')

test_generator = valAug.flow_from_directory(
	'data/test',
	target_size=(150, 150),
	color_mode='rgb',
	shuffle=False,
	batch_size=batch_size)


model.fit_generator(
	train_generator,
	steps_per_epoch=1,#2000//batch_size,
	epochs=50,
	validation_data=validation_generator,
	validation_steps=800//batch_size)

predIndices = model.predict_generator(test_generator, steps=125)#((2000//batch_size)+1))

predIndices = np.argmax(predIndices, axis=1)
print(len(test_generator.classes))
print(len(predIndices))
print(len(test_generator.class_indices))
print(len(test_generator.class_indices))

print(classification_report(test_generator.classes, predIndices, target_names=test_generator.class_indices.keys()))

cm = confusion_matrix(test_generator.classes, predIndices)

total=sum(sum(cm))
accuracy=(cm[0, 0]+cm[1, 1])/total
specificity=cm[1, 1]/(cm[1, 0]+cm[1, 1])
sensitvity=cm[0, 0]/(cm[0, 0]+cm[0, 1])
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitvity: {sensitvity}')

model.save_weights('first_try.h5')







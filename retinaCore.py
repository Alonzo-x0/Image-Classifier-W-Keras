from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import History
history = History()

os.environ['KMP_WARNINGS'] = '0'
#disable above if you need to see tensorflow start logs


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

nEpochSteps = 125

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


M = model.fit_generator(
	train_generator,
	steps_per_epoch=nEpochSteps,#2000//batch_size,
	epochs=50,
	validation_data=validation_generator,
	validation_steps=800//batch_size)

predIndices = model.predict_generator(test_generator, steps=nEpochSteps)#((2000//batch_size)+1))

a=test_generator.classes.tolist()
b = predIndices.tolist()


y = []
a = [float(i) for i in a]


for n in b:
	for k in n:
		y.append(k)

z = []
for floats in y:
	z.append(round(floats))
y = z

print(confusion_matrix(a, y))
print(classification_report(a, y,))# target_names=test_generator.class_indices.keys()))





plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 50), M.history['loss'], label='train_loss')
plt.plot(np.arange(0, 50), M.history['accuracy'], label='train_acc')

plt.title('Training loss and accuracy on the provided dataset')
plt.xlabel('Epoch No.')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('plot2.png')


model.save_weights('first_try.h5')

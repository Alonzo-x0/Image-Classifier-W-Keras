from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from keras.models import Sequential
from keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


os.environ['KMP_WARNINGS'] = '0'
#disable above if you need to see tensorflow start logs

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

img = load_img('data/test/cats/cat.4001.jpg')

imgArray = img_to_array(img)


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

#arImage = array_to_img(imgArray)

imgArray = imgArray.reshape((1,) + imgArray.shape)
i = 0
for batch in datagen.flow(imgArray, batch_size=32, save_to_dir=path, save_prefix='cats', save_format='jpeg'):
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

test_Datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_Datagen.flow_from_directory(
	'data/train',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_Datagen.flow_from_directory(
	'data/validation',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')
model.load_weights('first_try.h5')
model.fit_generator(
	train_generator,
	steps_per_epoch=2000//batch_size,
	epochs=50,
	validation_data=validation_generator,
	validation_steps=800//batch_size)

model.save_weights('first_try.h5')







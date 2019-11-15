from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import SeparableConv2D


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

img = load_img('dataset/dataset/test_set/cats/cat.4001.jpg')

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
model.add(SeparableConv2D(32, (3, 3), padding = 'same', input_shape = shape))
model.add(Activation('relu'))		
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(SeparableConv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))		
model.add(SeparableConv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))		
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(SeparableConv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(SeparableConv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(SeparableConv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Activation('softmax'))

opt=Adagrad(lr=INIT_LR, decay=INIT_LR/NUM_EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

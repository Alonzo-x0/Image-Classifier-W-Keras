from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


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



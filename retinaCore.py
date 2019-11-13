from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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

img = load_image('dataset/dataset/test_set/cats/cat.4001')

imgArray = img_to_array(img)

#arImage = array_to_img(imgArray)

imgArray = imgArray.reshape((1,) + imgArray.shape)
i = 0
for batch in datagen.flow(imgArray, batch_size=32, save_to_dir='imgPreview', save_prefix='cats', save_format='jpeg'):
	i += 1

	if i > 20:
		break



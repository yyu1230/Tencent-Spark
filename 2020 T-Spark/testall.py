#encoding:utf-8
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# Load data


os.chdir("G:\\wrk\\data\\432700_821742_bundle_archive\\")

#PATH = os.path.join(os.getcwd(), 'food11')
PATH = os.getcwd()

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')
evaluation_dir = os.path.join(PATH, 'evaluation')


batch_size = 512
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1./255)
# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = train_dir,
    shuffle = True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = validation_dir,
    shuffle = False,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 
val2_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = validation_dir,
    shuffle = False,
    target_size = (224, 224),
    class_mode = 'categorical') 

evl_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = evaluation_dir,
    shuffle = False,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 
evl2_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = evaluation_dir,
    shuffle = False,
    target_size = (224, 224),
    class_mode = 'categorical') 


sample_training_images, _ = next(train_data_gen)
sample_test_images, _ = next(evl_data_gen)
sample_test2_images, _ = next(evl2_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
def plotImages(images_arr):
    fig, axes = plt.subplots(3, 5, figsize=(15,15))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(sample_training_images[:15])


start = 360
num_testjpgs = 15
testcases = sample_test_images[start : start + num_testjpgs]
testcases2 = sample_test2_images[start : start + num_testjpgs]

#num_models = 10
num_models = 2

results = []

model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_100_resnet.h5')
results.append(model.predict(testcases2))

model = tf.keras.models.load_model('G:\\wrk\\data\\432700_821742_bundle_archive\\test_cont.h5')
results.append(model.predict(testcases2))

##model = tf.keras.models.load_model('cat_dog.h5')
#model = tf.keras.models.load_model('food11_3_cnn.h5')
##results = model.predict(sample_training_images[:15])
##results1 = model.predict(testcases)
#results.append(model.predict(testcases))
##print(results)
#
#model = tf.keras.models.load_model('food11_30_cnn.h5')
##results2 = model.predict(testcases)
#results.append(model.predict(testcases))
#
#model = tf.keras.models.load_model('food11_50_cnn.h5')
#results.append(model.predict(testcases))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_2_cnn.h5')
#results.append(model.predict(testcases))
#
#model = tf.keras.models.load_model('food11_3_cnn(vgg).h5')
#results.append(model.predict(testcases2))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_30_cnn_vgg.h5')
#results.append(model.predict(testcases2))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_3_cnn_vgg.h5')
#results.append(model.predict(testcases))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_60_cnn.h5')
#results.append(model.predict(testcases))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_100A_resnet.h5')
#results.append(model.predict(testcases2))
#
#model = tf.keras.models.load_model('G:\\wrk\\tai\\models\\food11_100_resnet.h5')
#results.append(model.predict(testcases2))

#fclass = ["bread面类", "dairy_product乳制品", "dessert甜点", "egg蛋", "fried_food煎制食品", "meat肉类", "noodles/pasta粉面", "rice米饭", "seafood海鲜", "soup汤", "vegetable/fruit蔬菜水果"]
fclass = ["bread", "dairy_product", "dessert", "egg", "fried_food", "meat", "noodles/pasta", "rice", "seafood", "soup", "vegetable/fruit"]

for i in range(num_testjpgs):
    maxindex = [0 for _ in range(num_models)]
    print(i, ':', end=' ')
    for j in range(1, 11):
    	  for k in range(num_models):
    	  	  if results[k][i][j] > results[k][i][maxindex[k]]:
    	  	      maxindex[k] = j
    for k in range(num_models):
    	  print(fclass[maxindex[k]], end=' ')
    print('\n')


#plotImages(testcases)
#plotImages(testcases2)

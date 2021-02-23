#encoding:utf-8
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# Load data


os.chdir("w:\\wrk\\data\\432700_821742_bundle_archive\\")  
#将下载的数据包, 432700_821742_bundle_archive.zip 解压在 w:\wrk\data\432700_821742_bundle_archive 目录下

PATH = os.getcwd()

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')
evaluation_dir = os.path.join(PATH, 'evaluation')


batch_size = 3347
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150



# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)

evl2_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = evaluation_dir,
    shuffle = False,
    target_size = (224, 224),
    class_mode = 'categorical') 


sample_test2_images, _ = next(evl2_data_gen)


start = 360   # start 应小于 batch_size，故 batch_size 设为3347，所有文件的总数
num_testjpgs = 15


testcases2 = sample_test2_images[start : start + num_testjpgs]


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
def plotImages(images_arr):
    fig, axes = plt.subplots(3, 5, figsize=(15,15))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(testcases2)



#num_models = 10
num_models = 1

results = []


model = tf.keras.models.load_model('w:\\wrk\demo\\mn22-283-0.8344.h5')
#自己改model所在目录


results.append(model.predict(testcases2))



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



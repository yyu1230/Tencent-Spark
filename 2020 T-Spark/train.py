#encoding:utf-8
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt




from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation  # ��keras��������㡢���ػ��㡢Dropout��ͼ����
 
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Add
from tensorflow.keras.models import Model

from tensorflow.keras.layers import DepthwiseConv2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler





# Load data

PATH = '/kaggle/input/food11-image-dataset'





train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')
evaluation_dir = os.path.join(PATH, 'evaluation')






# Evaluate the total data

i=0
total_train = 0
for f1 in os.walk(train_dir):
    #print(i,":---")
    #print(f1)
    #print(type(f1),len(f1),len(f1[0]),len(f1[1]),len(f1[2]))
    if i > 0:
        j = len(f1[2])
        total_train += j
#        print(os.path.basename(f1[0]), j)
    i += 1
print('totally', i - 1, 'kinds,', total_train, 'training pictures.' )

i=0
total_validation = 0
for f1 in os.walk(validation_dir):
    if i > 0:
        j = len(f1[2])
        total_validation += j
#        print(os.path.basename(f1[0]), j)
    i += 1
print('totally', i - 1, 'kinds,', total_validation, 'validation pictures.' )

i=0
total_evaluation = 0
for f1 in os.walk(evaluation_dir):
    if i > 0:
        j = len(f1[2])
        total_evaluation += j
#        print(os.path.basename(f1[0]), j)
    i += 1
print('totally', i - 1, 'kinds,', total_evaluation, 'evaluation pictures.' )







batch_size = 16
epochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224








def depthwise_separable_conv(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3,
                        strides=strides,
                        padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters,
               kernel_size=1,
               strides=1,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def create_mobile_net():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(filters=32,
               kernel_size=3,
#               strides=1,
               strides=2,
               padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = depthwise_separable_conv(x, 64, 1)
#    x = depthwise_separable_conv(x, 128, 1)
    x = depthwise_separable_conv(x, 128, 2)
    x = depthwise_separable_conv(x, 128, 1)
#    x = depthwise_separable_conv(x, 256, 1)
    x = depthwise_separable_conv(x, 256, 2)
    x = depthwise_separable_conv(x, 256, 1)
    x = depthwise_separable_conv(x, 512, 2)
    
    x = depthwise_separable_conv(x, 1024, 2)
    x = depthwise_separable_conv(x, 1024, 2)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(11, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="RMSprop",
        metrics=["accuracy"]
    )
    return model


model = create_mobile_net()
model.summary()








def scheduler(epoch):
    lr = 0.001
    if epoch > 30:
        lr = 0.0001
    if epoch > 60:
        lr = 0.00001
    if epoch > 90:
        lr = 0.000001
    if epoch > 120:
        lr = 0.0000001
    if epoch > 150:
        lr = 0.00000001
    if epoch > 180:
        lr = 0.000000001
    if epoch > 210:
        lr = 0.0000000001
    if epoch > 240:
        lr = 0.00000000001
    if epoch > 270:
        lr = 0.000000000001
    if epoch > 300:
        lr = 0.0000000000001
    return lr
change_lr = LearningRateScheduler(scheduler)
 








# Generator for our training data
train_image_datagen = ImageDataGenerator(
     rescale=1./255,
     rotation_range=20,
     horizontal_flip=True,
     width_shift_range=0.1,
     height_shift_range=0.1)

train_image_generator = train_image_datagen.flow_from_directory(
    batch_size = batch_size,
    directory = train_dir,
    shuffle = True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 







# Generator for our validation data
validation_image_generator = ImageDataGenerator(
     rescale=1./255,
     horizontal_flip=True,
     width_shift_range=0.1,
     height_shift_range=0.1)

validation_image_generator = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = validation_dir,
    shuffle = True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 







history = model.fit_generator(
    train_image_generator,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    callbacks=[change_lr],
    validation_data=validation_image_generator,
    validation_steps=total_validation // batch_size
)






model.save('/kaggle/output/food11_mobilenet.h5')



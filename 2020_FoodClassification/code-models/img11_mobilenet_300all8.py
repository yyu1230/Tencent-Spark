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


os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


# Load data

#PATH = os.path.join(os.getcwd(), 'food11')
#PATH = os.getcwd()
#PATH = "/cos_person/food11/"


ENV_ROOT = "/cos_person/"
PATH = os.path.join(ENV_ROOT, 'food11')


#ENV_ROOT = "M:/wrk/tai/models/"
#PATH = "G:/wrk/data/432700_821742_bundle_archive/"


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
#epochs = 103
IMG_HEIGHT = 224
IMG_WIDTH = 224





## Generator for our training data
#train_image_generator = ImageDataGenerator(rescale=1./255)
# Generator for our validation data

#validation_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

#train_data_gen = train_image_generator.flow_from_directory(
#    batch_size = batch_size,
#    directory = train_dir,
#    shuffle = True,
#    target_size = (IMG_HEIGHT, IMG_WIDTH),
#    class_mode = 'categorical') 
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size = batch_size,
    directory = validation_dir,
    shuffle = True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 




#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation  # ��keras��������㡢���ػ��㡢Dropout��ͼ����
 
#from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization
#from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Add
#from tensorflow.keras.models import Model

#from tensorflow.keras.layers import DepthwiseConv2D

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import LearningRateScheduler



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
#model.summary()


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
 
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             horizontal_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

#batch_size = 32
train_image_generator = datagen.flow_from_directory(
    batch_size = batch_size,
    directory = train_dir,
    shuffle = True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = 'categorical') 


#epochs = 10
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size
#)
#model.save('/cos_person/food11_mobilenet_10.h5')
#
#
#epochs = 20
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=10
#)
#model.save('/cos_person/food11_mobilenet_20.h5')
#
#
#epochs = 50
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=20
#)
#model.save('/cos_person/food11_mobilenet_50.h5')

#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_50.h5')
#
#epochs = 60
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=50
#)
#model.save('/cos_person/food11_mobilenet_60.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_60.h5')
#
#epochs = 70
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=60
#)
#model.save('/cos_person/food11_mobilenet_70.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_70.h5')
#
#epochs = 80
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=70
#)
#model.save('/cos_person/food11_mobilenet_80.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_80.h5')
#
#epochs = 90
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=80
#)
#model.save('/cos_person/food11_mobilenet_90.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_90.h5')
#
#epochs = 100
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=90
#)
#model.save('/cos_person/food11_mobilenet_100.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_100.h5')
#
#epochs = 110
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=100
#)
#model.save('/cos_person/food11_mobilenet_110.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_110.h5')
#
#epochs = 120
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=110
#)
#model.save('/cos_person/food11_mobilenet_120.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_120.h5')
#
#epochs = 180
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=120
#)
#model.save('/cos_person/food11_mobilenet_180.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_180.h5')
#
#epochs = 190
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=180
#)
#model.save('/cos_person/food11_mobilenet_190.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_190.h5')
#
#epochs = 200
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=190
#)
#model.save('/cos_person/food11_mobilenet_200.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_200.h5')
#
#epochs = 210
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=200
#)
#model.save('/cos_person/food11_mobilenet_210.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_210.h5')
#
#epochs = 220
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=210
#)
#model.save('/cos_person/food11_mobilenet_220.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_220.h5')
#
#epochs = 225
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=220
#)
#model.save('/cos_person/food11_mobilenet_225.h5')
#
#epochs = 230
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=225
#)
#model.save('/cos_person/food11_mobilenet_230.h5')
#
#epochs = 235
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=230
#)
#model.save('/cos_person/food11_mobilenet_235.h5')
#
#epochs = 240
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=235
#)
#model.save('/cos_person/food11_mobilenet_240.h5')


#model = tf.keras.models.load_model('/cos_person/food11_mobilenet_240.h5')
#
#epochs = 250
#
#history = model.fit_generator(
#    train_image_generator,
#    steps_per_epoch=total_train // batch_size,
#    epochs=epochs,
#    callbacks=[change_lr],
#    validation_data=val_data_gen,
#    validation_steps=total_validation // batch_size,
#    initial_epoch=240
#)
#model.save('/cos_person/food11_mobilenet_250.h5')


def model_fname(epoch_num):
    if epoch_num >= 100:
        fname = ENV_ROOT + "/mn8/mn8_{}.h5"
    elif epoch_num >= 10:
        fname = ENV_ROOT + "/mn8/mn8_0{}.h5"
    else:
        fname = ENV_ROOT + "/mn8/mn8_00{}.h5"
    return fname.format(epoch_num)


start_ep = 0
end_ep = 300
stepsaveper_ep = 50

if start_ep > 0:
    model = tf.keras.models.load_model(model_fname(start_ep))

cur_ep = start_ep
nxt_ep = start_ep + stepsaveper_ep
while nxt_ep <= end_ep:
    history = model.fit_generator(
        train_image_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=nxt_ep,
        callbacks=[change_lr],
        workers=8,
        verbose=2,
        validation_data=val_data_gen,
        validation_steps=total_validation // batch_size,
        initial_epoch=cur_ep
    )
    model.save(model_fname(nxt_ep))
    cur_ep = nxt_ep
    nxt_ep += stepsaveper_ep
    
    

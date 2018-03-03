#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:18:37 2017

"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPool3D, Flatten, Dense, Dropout

input_video = Input(shape=(224,224,16,3))
x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu')(input_video)
x = MaxPool3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu')(x)
x = MaxPool3D(pool_size=(2,2,2))(x)
x = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', activation='relu')(x)
x = MaxPool3D(pool_size=(2,2,2))(x)
x = Conv3D(filters=512, kernel_size=(3,3,3), padding='same', activation='relu')(x)
x = MaxPool3D(pool_size=(2,2,2))(x)
x = Conv3D(filters=512, kernel_size=(3,3,3), padding='same', activation='relu')(x)
x = MaxPool3D(pool_size=(2,2,2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
y = Dense(10, activation='softmax')(x)
model = Model(inputs=input_video, outputs=y)

model.summary()

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

import numpy as np
x_train = np.random.random(size=(100,224,224,16,3))
y_train = np.asarray([np.random.randint(0,10) for i in range(100)])
from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)

model.fit(x_train, y_train, batch_size=1, verbose=1, validation_split=0.1, epochs=10)

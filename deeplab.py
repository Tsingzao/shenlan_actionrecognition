#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import Input, Convolution2D, AtrousConvolution2D, MaxPooling2D, merge, ZeroPadding2D, Dropout, UpSampling2D

img_input = Input(shape=(512,512,1))

# Block 1
h = ZeroPadding2D(padding=(1, 1))(img_input)
h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

# Block 2
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

# Block 3
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

# Block 4
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

# Block 5
h = ZeroPadding2D(padding=(2, 2))(h)
h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
h = ZeroPadding2D(padding=(2, 2))(h)
h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
h = ZeroPadding2D(padding=(2, 2))(h)
h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
h = ZeroPadding2D(padding=(1, 1))(h)
p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

# branching for Atrous Spatial Pyramid Pooling
# hole = 6
b1 = ZeroPadding2D(padding=(6, 6))(p5)
b1 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6), activation='relu', name='fc6_1')(b1)
b1 = Dropout(0.5)(b1)
b1 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
b1 = Dropout(0.5)(b1)
b1 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_1')(b1)

# hole = 12
b2 = ZeroPadding2D(padding=(12, 12))(p5)
b2 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(12, 12), activation='relu', name='fc6_2')(b2)
b2 = Dropout(0.5)(b2)
b2 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
b2 = Dropout(0.5)(b2)
b2 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

# hole = 18
b3 = ZeroPadding2D(padding=(18, 18))(p5)
b3 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(18, 18), activation='relu', name='fc6_3')(b3)
b3 = Dropout(0.5)(b3)
b3 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
b3 = Dropout(0.5)(b3)
b3 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

# hole = 24
b4 = ZeroPadding2D(padding=(24, 24))(p5)
b4 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(24, 24), activation='relu', name='fc6_4')(b4)
b4 = Dropout(0.5)(b4)
b4 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
b4 = Dropout(0.5)(b4)
b4 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

s = merge([b1, b2, b3, b4], mode='sum')
out = UpSampling2D(size=(8, 8))(s)

inputs = img_input

# Create model.
model = Model(inputs, out, name='deeplabV2')
model.summary()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:57:37 2017

"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

img_w = 256
img_h = 256
n_labels = 21

kernel = 3

encoding_layers = [
    Convolution2D(64, kernel, kernel, border_mode='same', input_shape=(img_h, img_w, 1)),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
]

autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [
    UpSampling2D(),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Reshape((n_labels, img_h * img_w)))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))

autoencoder.summary()

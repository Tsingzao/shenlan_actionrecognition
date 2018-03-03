#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:13:31 2017

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
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Input, merge

'''=========================================================================='''
spatial_input = Input((224, 224, 3))
x = Convolution2D(96, 7, 7, activation='relu', border_mode='same', name='spatial_conv1')(spatial_input)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', name='spatial_conv2')(x)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='spatial_conv3')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='spatial_conv4')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='spatial_conv5')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu', name='spatial_fc6')(x)
x = Dropout(0.9)(x)
x = Dense(2048, activation='relu', name='spatial_fc7')(x)
x = Dropout(0.7)(x)
y = Dense(10, activation='softmax')(x)
spatial_model = Model(spatial_input, y)
spatial_model.summary()

'''=========================================================================='''
temporal_input = Input((224, 224, 20))
x = Convolution2D(96, 7, 7, activation='relu', border_mode='same', name='temporal_conv1')(temporal_input)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', name='temporal_conv2')(x)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv3')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv4')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv5')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu', name='temporal_fc6')(x)
x = Dropout(0.9)(x)
x = Dense(2048, activation='relu', name='temporal_fc7')(x)
x = Dropout(0.7)(x)
y = Dense(10, activation='softmax')(x)
temporal_model = Model(temporal_input, y)
temporal_model.summary()

'''=========================================================================='''
spatial_input = Input((224, 224, 3))
x = Convolution2D(96, 7, 7, activation='relu', border_mode='same', name='spatial_conv1')(spatial_input)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', name='spatial_conv2')(x)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='spatial_conv3')(x)
spatial_output = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='spatial_conv4')(x)

temporal_input = Input((224, 224, 20))
x = Convolution2D(96, 7, 7, activation='relu', border_mode='same', name='temporal_conv1')(temporal_input)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', name='temporal_conv2')(x)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv3')(x)
temporal_output = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv4')(x)

fusion_output  = merge([spatial_output, temporal_output], mode='sum')
# fusion_output  = merge([spatial_output, temporal_output], mode='concat')
# fusion_output  = merge([spatial_output, temporal_output], mode='ave')

x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='temporal_conv5')(fusion_output)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu', name='fusion_fc6')(x)
x = Dropout(0.9)(x)
x = Dense(2048, activation='relu', name='fusion_fc7')(x)
x = Dropout(0.7)(x)
y = Dense(101, activation='softmax')(x)
fusion_model = Model([spatial_input, temporal_input], y)
fusion_model.summary()

'''=========================================================================='''
from keras.layers import ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D
from keras.applications.resnet50 import conv_block, identity_block

spatial_input = Input((224, 224, 3))
x = ZeroPadding2D((3, 3))(spatial_input)
x = Convolution2D(64, 7, 7, subsample=(2, 2), name='sconv1')(x)
x = BatchNormalization(axis=3, name='sbn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

temporal_input = Input((224, 224, 20))
y = ZeroPadding2D((3, 3))(temporal_input)
y = Convolution2D(64, 7, 7, subsample=(2, 2), name='tconv1')(y)
y = BatchNormalization(axis=3, name='tbn_conv1')(y)
y = Activation('relu')(y)
y = MaxPooling2D((3, 3), strides=(2, 2))(y)

x = conv_block(x, 3, [64, 64, 256], stage=2, block='sa', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=2, block='sb')
x = identity_block(x, 3, [64, 64, 256], stage=2, block='sc')

y = conv_block(y, 3, [64, 64, 256], stage=2, block='ta', strides=(1, 1))
y = identity_block(y, 3, [64, 64, 256], stage=2, block='tb')
y = identity_block(y, 3, [64, 64, 256], stage=2, block='tc')

x = merge([x, y], mode='sum')
x = conv_block(x, 3, [128, 128, 512], stage=3, block='sa')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='sb')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='sc')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='sd')

y = conv_block(y, 3, [128, 128, 512], stage=3, block='ta')
y = identity_block(y, 3, [128, 128, 512], stage=3, block='tb')
y = identity_block(y, 3, [128, 128, 512], stage=3, block='tc')
y = identity_block(y, 3, [128, 128, 512], stage=3, block='td')

x = merge([x, y], mode='sum')
x = conv_block(x, 3, [256, 256, 1024], stage=4, block='sa')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='sb')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='sc')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='sd')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='se')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='sf')

y = conv_block(y, 3, [256, 256, 1024], stage=4, block='ta')
y = identity_block(y, 3, [256, 256, 1024], stage=4, block='tb')
y = identity_block(y, 3, [256, 256, 1024], stage=4, block='tc')
y = identity_block(y, 3, [256, 256, 1024], stage=4, block='td')
y = identity_block(y, 3, [256, 256, 1024], stage=4, block='te')
y = identity_block(y, 3, [256, 256, 1024], stage=4, block='tf')

x = merge([x, y], mode='sum')
x = conv_block(x, 3, [512, 512, 2048], stage=5, block='sa')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='sb')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='sc')

y = conv_block(y, 3, [512, 512, 2048], stage=5, block='ta')
y = identity_block(y, 3, [512, 512, 2048], stage=5, block='tb')
y = identity_block(y, 3, [512, 512, 2048], stage=5, block='tc')

x = AveragePooling2D((7, 7), name='savg_pool')(x)
x = Flatten()(x)
x = Dense(101, activation='softmax', name='sfc101')(x)

y = AveragePooling2D((7, 7), name='tavg_pool')(y)
y = Flatten()(y)
y = Dense(10, activation='softmax', name='tfc101')(y)

residual_model = Model([spatial_input, temporal_input], [x, y])
residual_model.summary()

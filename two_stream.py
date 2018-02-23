#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created by Tsingzao

"""

'''=========================================================================='''
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

'''prepare your video data'''

'''=========================================================================='''

'''=========================================================================='''
spatial_input = Input((224, 224, 3))

'''write your spatial code here'''

ys = Dense(2, activation='softmax')(x)
spatial_model = Model(spatial_input, ys)
spatial_model.summary()

'''=========================================================================='''
temporal_input = Input((224, 224, 20))

'''write your temporal code here'''

yt = Dense(2, activation='softmax')(x)
temporal_model = Model(temporal_input, yt)
temporal_model.summary()

'''========================================================================='''

'''write your fusion code here'''

y = Dense(2, activation='softmax')(x)
fusion_model = Model([spatial_input, temporal_input], y)
fusion_model.summary()

'''=========================================================================='''

'''write your compile code; set the learning rate and method; and run your code'''

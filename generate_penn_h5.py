#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 08:59:54 2018

"""

import h5py
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'#1,2,3,4'
os.system('echo $CUDA_VISIBLE_DEVICES')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.preprocessing import image   # to read video frames

penn_path = "/home/***/penn/"

X_train = []    # to store videos
y_train = []    # to store labels
for video in os.listdir(penn_path):     #penn_path is the full path you store your videos

    video_frames = os.listdir(penn_path+'/'+video)     #list all video frames within a video, note videos are stored in frames
    frame_number = len(video_frames)                    #total frame numbers
    skip_length  = frame_number // 7                   #number of video clips

    for i in range(skip_length):                        

        frame_select = tuple(range(i, skip_length*7, skip_length)) #get the frame index to read
        video_cube   = np.zeros((270, 480, 7, 3), dtype='uint8')   #to store video clips
        frame_count  = 0
        for frame in frame_select:                                  #frames
            temp = 'frame-'+str(frame+1).zfill(4)+'.jpg'             #frame name
            frame_path = penn_path + '/'+ video + '/' + temp        #frame path
            frame_image = image.load_img(frame_path, target_size=(270, 480))    #read frame
            frame_array = image.img_to_array(frame_image)           #frame to numpy array
            video_cube[:,:,frame_count,:] = frame_array             #video cubes
            frame_count += 1                                        
            print('Read ' + video + ': ' + str(i+1) + '/' + str(skip_length) + ' - frame ' + str(frame_count))
        X_train.append(video_cube)                                  # stored data
        if video in ['0051','0052','0053','0058','0059']:
            video_class = 0
        else:
            video_class = 1
        y_train.append(video_class)                                 # stored label
        
X_train = np.asarray(X_train)                   # Transform list to numpy array
y_train = np.asarray(y_train)
h5file = h5py.File('/home/***/penn.h5','w')     # create h5file to store video frames
h5file.create_dataset(name='data',data=X_train)    # push the data into h5file
h5file.create_dataset(name='label',data=y_train)
h5file.close()                                  # close the file
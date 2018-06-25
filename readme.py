#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 

This is a demo for video preprocessing

including: 
    0\ video to frame
    1\ generate video list
    2\ video augmentation
    3\ generate video to h5file
    
Note: 
    You'd better download the original video files
    1\ UCF-101: http://crcv.ucf.edu/data/UCF101.php
        Both the videos and file list are given
    2\ Penn: https://dreamdragon.github.io/PennAction/
        Only the videos and annotations, you need generate the list by yourself

Warning:
    In the following, we take one class as example        
        
@author: 
"""

'''0\ video to frame'''
# You may use FFmpeg to extract video frames

'''1\ frame to list'''
import os
penn_path = '/home/***/penn/'
fw = open('/home/***/penn.lst','w')
for video in ['0051','0052','0053','0058','0059']:      # the list is the file you'd read
    video_path = penn_path+video        # the full path of videos
    fw.writelines(video_path+' 0\n')    # 0/1 represents the video label
fw.close()

'''2\ data augmentation'''
import h5py
import numpy as np
from keras.preprocessing import image   # to read video frames

X_train = []    # to store videos
y_train = []    # to store labels
video_class = 0 # sample class
for video in os.listdir(penn_path):     #penn_path is the full path you store your videos

    video_frames = os.listdir(video_path+'/'+video)     #list all video frames within a video, note videos are stored in frames
    frame_number = len(video_frames)                    #total frame numbers
    skip_length  = frame_number // 16                   #number of video clips

    for i in range(skip_length):                        

        frame_select = tuple(range(i, skip_length*16, skip_length)) #get the frame index to read
        video_cube   = np.zeros((112, 112, 16, 3), dtype='uint8')   #to store video clips
        frame_count  = 0
        for frame in frame_select:                                  #frames
            temp = 'frame-'+str(frame).zfill(4)+'.jpg'             #frame name
            frame_path = video_path + '/'+ video + '/' + temp       #frame path
            frame_image = image.load_img(frame_path, target_size=(112, 112))    #read frame
            frame_array = image.img_to_array(frame_image)           #frame to numpy array
            video_cube[:,:,frame_count,:] = frame_array             #video cubes
            frame_count += 1                                        
            print('Read ' + video + ': ' + str(i+1) + '/' + str(skip_length) + ' - frame ' + str(frame_count))
        X_train.append(video_cube)                                  # stored data
        y_train.append(video_class)                                 # stored label

'''3\ to h5file'''
X_train = np.asarray(X_train)                   # Transform list to numpy array
y_train = np.asarray(y_train)
h5file = h5py.File('/home/***/penn.h5','w')     # create h5file to store video frames
h5file.create_dataset(name='X',data=X_train)    # push the data into h5file
h5file.create_dataset(name='y',data=y_train)
h5file.close()                                  # close the file

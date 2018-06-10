#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 
"""

import os
video_path = '/home/***/penn/'              #Your Video Path (stored in frames)
fw = open('/home/***/filelist.lst','w')     #Your Destination filelist path
for video in os.listdir(video_path):
    line = video_path+video+'\n'            #Video Path
    fw.writelines(line)
fw.close()
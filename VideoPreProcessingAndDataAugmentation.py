import os
import h5py
import numpy as np
from keras.preprocessing import image
video_path = '/home/VideoData/UCF-RGB/UCF-frame/'
video_file = os.listdir(video_path)[1:]
h5file_path= '/home/VideoData/UCF-h5/UCF-Augmentation/'

video_class_key_value = {}
with open('/home/VideoData/ucfTrainTestlist/classInd.txt', 'r') as fp:
    for line in fp.readlines():
        key = line.strip().split(' ')[1].upper()
        value = line.strip().split(' ')[0]
        video_class_key_value[key] = value
fp.close()

fp = open('/home/VideoData/UCF-h5/UCF-Augmentation/ucflist-part1.txt','w')

X_train = []
y_train = []
video_count = 1
original_part = 0
for video in video_file:

    video_class  = video.split('_')[1].upper()
    video_frames = os.listdir(video_path+'/'+video)
    frame_number = len(video_frames)
    skip_length  = frame_number // 16

    for i in range(skip_length):
        current_part = video_count // 50000

        frame_select = tuple(range(1, skip_length*16, skip_length))
        video_cube   = np.zeros((112, 112, 16, 3), dtype='uint8')
        frame_count  = 0
        for frame in frame_select:
            if frame < 10:
                temp = 'frame-000'+str(frame)+'.jpg'
            elif frame < 100:
                temp = 'frame-00'+str(frame)+'.jpg'
            elif frame < 1000:
                temp = 'frame-0'+str(frame)+'.jpg'
            else:
                temp = 'frame-'+str(frame)+'.jpg'
            frame_path = video_path + '/'+ video + '/' + temp
            frame_image = image.load_img(frame_path, target_size=(112, 112))
            frame_array = image.img_to_array(frame_image)
            video_cube[:,:,frame_count,:] = frame_array
            frame_count += 1
            print('Read ' + video + ': ' + str(i+1) + '/' + str(skip_length) + ' - frame ' + str(frame_count))
        X_train.append(video_cube)
        y_train.append(int(video_class_key_value[video_class]) - 1)
        fp.write(video)
        fp.write(' ')
        fp.write(video_class_key_value[video_class])
        fp.write('\n')
        video_count += 1

        if not current_part == original_part:
            print('CurrentPart %d, OriginalPart %d' % (current_part, original_part))
            print('VideoCount ', video_count-1)
            h5file = h5py.File(h5file_path+'UCF-part'+str(current_part)+'.h5','w')
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            print('Create X_train ...')
            h5file.create_dataset('X_train', data=X_train)
            print('Create y_train ...')
            h5file.create_dataset('y_train', data=y_train)
            print('Create h5 file Done!')
            h5file.close()
            X_train = []
            y_train = []
            fp.close()
            fp = open('/home/VideoData/UCF-h5/UCF-Augmentation/ucflist-part'+str(current_part+1)+'.txt','w')

        if video_count == 148933:
            print('VideoCount ', video_count-1)
            h5file = h5py.File(h5file_path+'UCF-part'+str(current_part+1)+'.h5','w')
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            print('Create X_train ...')
            h5file.create_dataset('X_train', data=X_train)
            print('Create y_train ...')
            h5file.create_dataset('y_train', data=y_train)
            print('Create h5 file Done!')
            h5file.close()
            X_train = []
            y_train = []
            fp.close()

        original_part = current_part





index = []
with open('/home/VideoData/ucfTrainTestlist/testlist03.txt','r') as fp:
    for line in fp.readlines():
        index.append(line.strip().split('/')[1].split('.avi')[0])
fp.close()

fw = open('/home/VideoData/ucfTrainTestlist/testlist03-aug.txt','w')
for i in range(15):
    content = []
    with open('/home/VideoData/UCF-h5/UCF-Augmentation-224/ucflist-224-part'+str(i+1)+'.txt') as fp:
        for line in fp.readlines():
            content.append(line.strip().split(' ')[0])
    fp.close()
    for ind in index:
        if ind in content:
            cnt = content.count(ind)
            beg = content.index(ind)
            for c in range(cnt):
                print(beg+c+i*10000)
                num = str(beg+c+i*10000)
                fw.write(num)
fw.close()
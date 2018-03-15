#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:15:59 2018

@author: Wonjoong Cheon
"""
#%%
import tensorflow as tf
import skimage.io as io 
import numpy as np
import sys
# For figure
import matplotlib
import matplotlib.pyplot as plt
# For Python verions check 
import platform
#
platform_name = platform.system()
if(platform_name is 'Windows'):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
else:
    # python 2.7
    python_version = sys.version_info
    if (python_version[0] == 2):
        import tkFileDialog
        # ROI file
        img_fullfilename_single_full = tkFileDialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
        img_pathname_single, img_filename_single = os.path.split(img_fullfilename_single_full)
        img_extension_filename = img_filename_single .split('.')[1]
        img_file_lists = sorted(glob.glob(os.path.join(img_pathname_single,'*.'+img_extension_filename)))    
        #
        #
        annotation_fullfilename_single_full = tkFileDialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
        annotation_pathname_single, annotation_filename_single = os.path.split(annotation_fullfilename_single_full)
        annotation_extension_filename = annotation_filename_single .split('.')[1]
        annotation_file_lists = sorted(glob.glob(os.path.join(annotation_pathname_single,'*.'+annotation_extension_filename)))    
    else:
        #
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        tfrecords_fullname  = filedialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
        
tfrecords_filename = tfrecords_fullname

#%%

def read_and_decode(filename_queue_, IMAGE_HEIGHT_, IMAGE_WIDTH_, IMAGE_CHANNEL_):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue_)
    
    features = tf.parse_single_example(serialized_example,
                                       # Defaults are not specified since both keys are required)
                                       features = {
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64), 
                                               'channel': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'mask_raw': tf.FixedLenFeature([], tf.string)
                                               })
    
    # Convert from a scalar string tensor to a uint8 tensor with shape 
    image = tf.decode_raw(features['image_raw'], tf.float64)
    annotation = tf.decode_raw(features['mask_raw'], tf.float64)
    #
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)
    #
#    image_shape = tf.pack([height, width, channel])
#    annotation_shape = tf.pack([height, width, channel])
    image_shape = tf.stack([height, width, channel])
    annotation_shape = tf.stack([height, width, channel])
    #
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    #
#    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype= tf.int32)
#    annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype= tf.int32)
    #
    #
    # Random tranformation can be put here: right before you crop image image to predifined size. 
    # To get more information look at the stackoverflow question linked above. 
    #
    resized_image = tf.image.resize_image_with_crop_or_pad(image= image, target_height= IMAGE_HEIGHT_, target_width= IMAGE_WIDTH_)
    #resized_image_shape = tf.shape(resized_image)
    resized_image = tf.reshape(resized_image , [IMAGE_HEIGHT_, IMAGE_WIDTH_, IMAGE_CHANNEL_])
    #
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image= annotation, target_height= IMAGE_HEIGHT_, target_width= IMAGE_WIDTH_)
    #resized_annotation_shape  = tf.shape(resized_annotation)
    resized_annotation = tf.reshape(resized_annotation, [IMAGE_HEIGHT_, IMAGE_WIDTH_, IMAGE_CHANNEL_])
    #
    images, annotations= tf.train.shuffle_batch([resized_image, resized_annotation], 
                                                batch_size= 2,
                                                capacity= 30, 
                                                num_threads= 1,
                                                min_after_dequeue= 10)
    return images, annotations
    

#%%
# Parameter for Crop and Padding (Original shape of image is 512 X 512)
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 1
#
filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs= None )
# Even when reading in multiple threads, share the filename queue 
# (분산 컴퓨팅이 가능함, 그때 옵션을 신경써줘야함! 나는 아직은..ㅇㅅㅇ)

image, annotation = read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
#%%
#
# The op for initializaing the variables. 
IMAGE_DUBUG = True
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord= coord)

#%
    for iter1 in range(200):
        img, anno = sess.run([image, annotation])
        print(img[0, :, :, :].shape)
        print('Current batch')
        
        # We selected the batch size of "two:2"
        # So we should get two image pairs in each batch 
        # Lets's make sure it is random 
        if(len(np.unique(anno))>1):
            fig = plt.figure(figsize=(10, 5))
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            #
            ax1.imshow(img[0,:,:,0], cmap = plt.get_cmap('gray'))
            ax1.set_title('DICOM')
            ax1.axis('off')
            #
            ax2.imshow(anno[0,:,:,0], cmap = plt.get_cmap('gray'))
#            ax2.imshow(anno[0,:,:,0])
            ax2.set_title('Annotation')
            ax2.axis('off')
            
            fig = plt.figure(figsize=(10, 5))
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            #
            ax1.imshow(img[1,:,:,0], cmap = plt.get_cmap('gray'))
            ax1.set_title('DICOM')
            ax1.axis('off')
            #
            ax2.imshow(anno[1,:,:,0], cmap = plt.get_cmap('gray'))
#            ax2.imshow(anno[1,:,:,0])
            ax2.set_title('Annotation')
            ax2.axis('off')
    coord.request_stop()
    coord.join(threads=threads)
print('Done!!')

# Fix to bug digging Stackoverflow
# https://stackoverflow.com/questions/34050071/tensorflow-random-shuffle-queue-is-closed-and-has-insufficient-elements/43370673 

#%%
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#%
for iter1 in range(30):
    img, anno = sess.run([image, annotation])
    print(img[0, :, :, :].shape)
    print('Current batch')
    
    # We selected the batch size of "two:2"
    # So we should get two image pairs in each batch 
    # Lets's make sure it is random 
    io.imshow(img[0,:,:,:])
    io.show()
    
    io.imshow(anno[0,:,:,:])
    io.show()
    
    io.imshow(img[1,:,:,:])
    io.show()
    
    io.imshow(anno[1,:,:,:])
    io.show()  

coord.request_stop()
coord.join(threads=threads)

print('Done!!')
 
    
#%%


   
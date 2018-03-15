#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 00:04:08 2018

@author: Wonjoong Cheon
"""

#%%
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
#
import os
import glob
import sys
import nibabel as nib
import numpy as np
from skimage.transform import rotate
from skimage.transform import resize

import matplotlib.pyplot as plt

#%% function
def print_shape(string, x):
    print ("Shape of '%s' is %s" % (string, x.shape,))

#%% DATA LIST-UP
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
        img_fullfilename_single_full  = filedialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
        img_pathname_single, img_filename_single = os.path.split(img_fullfilename_single_full)
        img_extension_filename = img_filename_single .split('.')[1]
        img_file_lists = sorted(glob.glob(os.path.join(img_pathname_single,'*.'+img_extension_filename)))    
        #
        #
        annotation_fullfilename_single_full = filedialog.askopenfilename(initialdir="./", title='Select imgs file for tfrecord!')
        annotation_pathname_single, annotation_filename_single = os.path.split(annotation_fullfilename_single_full)
        annotation_extension_filename = annotation_filename_single .split('.')[1]
        annotation_file_lists = sorted(glob.glob(os.path.join(annotation_pathname_single,'*.'+annotation_extension_filename)))    

filename_pairs = zip(img_file_lists, annotation_file_lists)
filename_pairs = zip(img_file_lists[:2], annotation_file_lists[:2])

#%%
def _bytes_feature(value_):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value_]))

def _int64_feature(value_):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value_]))

def _float_feature(value_):
    tf.train.Feature(FloatList = tf.train.FloatList(value=[value_]))

tfrecords_filename = 'dicom_nii_LITS_Liver.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

#%%
original_images = []
#
#img_path = img_file_lists[0]
#annotation_path = annotation_file_lists[0]
patient_counter = 0 
for img_path, annotation_path in filename_pairs:
#    
#    img = np.array(Image.open(img_path))
#    annotation = np.array(Image.open(annotation_path))
 
#filepath_roi = '/home/ubuntu/wjcheon/chsodam/segmentation-120.nii'
#filepath_img = '/home/ubuntu/wjcheon/chsodam/volume-120.nii'
 
    train_instance_nii = nib.load(img_path)
    data3D_img = train_instance_nii.get_data()  #float64
    #
    # Standardization
    data3D_img_mean = np.mean(data3D_img)
    data3D_img_std = np.std(data3D_img)
    data3D_img = (data3D_img - data3D_img_mean) / data3D_img_std
    #
    # roi
    train_roi_instance_nii = nib.load(annotation_path)
    data3D_roi = train_roi_instance_nii.get_data()    #uint8
    [sx, sy, sz] = np.shape(data3D_roi)
    # Select Liver 
    data3D_roi[data3D_roi>0] = 1
    data3D_roi = np.float64(data3D_roi)
    print(np.unique(data3D_roi))
    #
    for iter1 in range(sz):
        img2d = data3D_img[:,:,iter1]
        roi2d = data3D_roi[:,:,iter1]
        # Expand dimension
        img2d = np.expand_dims(img2d, axis= -1)
        roi2d = np.expand_dims(roi2d, axis= -1)
 
        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img2d.shape[0]
        width = img2d.shape[1]
        channel_num = img2d.shape[2]
#        print("height: {},   width: {},  channel_num: {}".format(height, width, channel_num)) 
        # Put in the original images into array
        # Just for future check for correctness
        original_images.append((img2d, roi2d))
        
        img_raw = img2d.tostring()   # binary
        annotation_raw = roi2d.tostring() # binary
        # generate one example 
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channel': _int64_feature(channel_num),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))
        
        writer.write(example.SerializeToString())

    print("{} th *.nii file was serialized !!".format(patient_counter+1))
    patient_counter += 1 

writer.close()
del img2d, roi2d

#%%
reconstructed_images = []
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
#    print(string_record)
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    channel_num = int(example.features.feature['channel']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    annotation_string = (example.features.feature['mask_raw']
                                .bytes_list
                                .value[0])
    #
    #
    img_1d = np.fromstring(img_string, dtype=np.float64)
    reconstructed_img = img_1d.reshape((height, width, channel_num))
    #
    annotation_1d = np.fromstring(annotation_string, dtype=np.float64)
    reconstructed_annotation = annotation_1d.reshape((height, width, channel_num))
    #
    reconstructed_images.append((reconstructed_img, reconstructed_annotation))

#%%
# Let's check if the reconstructed images match
# the original images
for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))
    
#%%
    

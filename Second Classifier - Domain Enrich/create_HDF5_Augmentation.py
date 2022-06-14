import cv2
import numpy as np
import os, glob
import h5py
import argparse
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from numpy import ndarray
from scipy import ndimage

parser = argparse.ArgumentParser(description='Generate H5 Augmentation file')

parser.add_argument("--path_IMG",type=str,default = 'Path to data image folder')
parser.add_argument("--path_write_h5",type=str, default = 'Path to save H5 file')
parser.add_argument("--num_slices", type=str, default=20)
parser.add_argument("--num_Munition", type=int, default=15)
parser.add_argument("--num_Shotput", type=int, default=13)
parser.add_argument("--num_Stell_Alu", type=int, default=18)

opt = parser.parse_args()

path_dataset = opt.path_IMG
num_Munition = opt.num_Munition
num_Shotput = opt.num_Shotput
num_Stell_Alu = opt.num_Stell_Alu

dim_height_img = 98
dim_width_img = 98
resized_height_img = 128
resized_width_img = 128
dim_channel_img = 3

dir_data = glob.glob(path_dataset + '/*')
num_slices = opt.num_slices
num_total_subvolumes = len(dir_data)

# Multi Slice data
dataset_multislice_original = np.zeros([num_total_subvolumes, num_slices, dim_channel_img, resized_height_img, resized_width_img])
dataset_multislice_90 = np.zeros([num_total_subvolumes, num_slices, dim_channel_img, resized_height_img, resized_width_img])
dataset_multislice_180 = np.zeros([num_total_subvolumes, num_slices, dim_channel_img, resized_height_img, resized_width_img])
dataset_multislice_270 = np.zeros([num_total_subvolumes, num_slices, dim_channel_img, resized_height_img, resized_width_img])

# Alongtrack data
dataset_alongtrack_original = np.zeros([num_total_subvolumes, dim_channel_img, dim_height_img, dim_width_img])
dataset_alongtrack_augmented = np.zeros([num_total_subvolumes, dim_channel_img, dim_height_img, dim_width_img])

# Crosstrack data
dataset_crosstrack_original = np.zeros([num_total_subvolumes, dim_channel_img, dim_height_img, dim_width_img])
dataset_crosstrack_augmented = np.zeros([num_total_subvolumes, dim_channel_img, dim_height_img, dim_width_img])

# label
dataset_label = np.zeros([num_total_subvolumes, 1])

ID_slice = []
ID_subvolume = []

subvolume = 0

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    ### process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

for path_folder in dir_data:
    ID = int(path_folder.split('/')[-1][:5])
    ID_subvolume.append(ID)

    # read and save image
    # Multi Slices
    dir_img_MS = sorted(glob.glob(path_folder + '/MultiSlices' + '/*.png'))
    for i in range (0, num_slices, 1):
        img = cv2.imread(dir_img_MS[i])
        img = img[:, :, :].astype(np.float32)/255
        img = center_crop(img, (dim_height_img, dim_width_img))
        img_original = cv2.resize(img, (resized_height_img, resized_width_img), interpolation=cv2.INTER_AREA)

        # Data Augmentation
        img_90 = ndimage.rotate(img_original, 90)
        img_180 = ndimage.rotate(img_original, 180)
        img_270 = ndimage.rotate(img_original, 270)

        img_original = np.einsum('kli->ikl', img_original)
        img_90 = np.einsum('kli->ikl', img_90)
        img_180 = np.einsum('kli->ikl', img_180)
        img_270 = np.einsum('kli->ikl', img_270)

        dataset_multislice_original[subvolume, i, :, :, :] = img_original
        dataset_multislice_90[subvolume, i, :, :, :] = img_90
        dataset_multislice_180[subvolume, i, :, :, :] = img_180
        dataset_multislice_270[subvolume, i, :, :, :] = img_270

        ID_slice.append(ID)

    # Along-track
    dir_img_AT = glob.glob(path_folder + '/AlongTrack' + '/*.png')
    img = cv2.imread(dir_img_AT[0])
    img = img[:, :, :].astype(np.float32) / 255
    img = center_crop(img, (dim_height_img, dim_width_img))
    img = np.einsum('kli->ikl', img)
    dataset_alongtrack_original[subvolume, :, :, :] = img
    dataset_crosstrack_augmented[subvolume, :, :, :] = img

    # Cross-track
    dir_img_CT = glob.glob(path_folder + '/CrossTrack' + '/*.png')
    img = cv2.imread(dir_img_CT[0])
    img = img[:, :, :].astype(np.float32) / 255
    img = center_crop(img, (dim_height_img, dim_width_img))
    img = np.einsum('kli->ikl', img)
    dataset_crosstrack_original[subvolume, :, :, :] = img
    dataset_alongtrack_augmented[subvolume, :, :, :] = img

    subvolume += 1

# label
for k in ID_subvolume:
    if k < num_Munition + 1:
        dataset_label[ID_subvolume.index(k)] = 0
    elif k > num_Munition and k < num_Munition + num_Shotput + 1:
        dataset_label[ID_subvolume.index(k)] = 1
    else:
        dataset_label[ID_subvolume.index(k)] = 2

dataset_multislice_original = dataset_multislice_original[:num_total_subvolumes]
dataset_multislice_90 = dataset_multislice_90[:num_total_subvolumes]
dataset_multislice_180 = dataset_multislice_180[:num_total_subvolumes]
dataset_multislice_270 = dataset_multislice_270[:num_total_subvolumes]

dataset_alongtrack_original = dataset_alongtrack_original[:num_total_subvolumes]
dataset_alongtrack_augmented = dataset_alongtrack_augmented[:num_total_subvolumes]
dataset_crosstrack_original = dataset_crosstrack_original[:num_total_subvolumes]
dataset_crosstrack_augmented = dataset_crosstrack_augmented[:num_total_subvolumes]

dataset_label = dataset_label[:num_total_subvolumes]

dataset_multislice_original = np.array(dataset_multislice_original)
dataset_multislice_90 = np.array(dataset_multislice_90)
dataset_multislice_180 = np.array(dataset_multislice_180)
dataset_multislice_270 = np.array(dataset_multislice_270)

dataset_alongtrack_original = np.array(dataset_alongtrack_original)
dataset_alongtrack_augmented = np.array(dataset_alongtrack_augmented)
dataset_crosstrack_original = np.array(dataset_crosstrack_original)
dataset_crosstrack_augmented = np.array(dataset_crosstrack_augmented)

dataset_label = np.array(dataset_label)

# Concatenation
dataset_multislice = np.concatenate((dataset_multislice_original, dataset_multislice_90, dataset_multislice_180, dataset_multislice_270), axis= 0)
dataset_alongtrack = np.concatenate((dataset_alongtrack_original, dataset_alongtrack_augmented, dataset_alongtrack_original, dataset_alongtrack_augmented), axis= 0)
dataset_crosstrack = np.concatenate((dataset_crosstrack_original, dataset_crosstrack_augmented, dataset_crosstrack_original, dataset_crosstrack_augmented), axis= 0)
label = np.concatenate((dataset_label, dataset_label, dataset_label, dataset_label), axis = 0)

# Check
print(sum(label))
print(label.shape)
print(dataset_multislice.shape)
print(dataset_alongtrack.shape)
print(dataset_crosstrack.shape)


# save all the data into H5py
h5w = h5py.File(opt.path_write_h5, 'w')
h5w.create_dataset(name='dataset_multislice', dtype=np.float32, shape=dataset_multislice.shape, data=dataset_multislice)
h5w.create_dataset(name='dataset_alongtrack', dtype=np.float32, shape=dataset_alongtrack.shape, data=dataset_alongtrack)
h5w.create_dataset(name='dataset_crosstrack', dtype=np.float32, shape=dataset_crosstrack.shape, data=dataset_crosstrack)
h5w.create_dataset(name='dataset_label', dtype=int, shape=label.shape, data=label)
h5w.close()

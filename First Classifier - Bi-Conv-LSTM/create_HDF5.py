import cv2
import numpy as np
import os, glob
import h5py
import argparse
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from numpy import ndarray
import random

parser = argparse.ArgumentParser(description='Generate H5 file')

parser.add_argument("--path_IMG",type=str,default = 'Path to Image Folder')
parser.add_argument("--path_write_h5_test",type=str, default = 'Path to H5 Folder')
parser.add_argument("--path_write_h5_train_val",type=str, default = 'Path to H5 Folder')
parser.add_argument("--num_slices", type=str, default=20)
parser.add_argument("--num_non_resonant_samples", type=int, default=245)

opt = parser.parse_args()

path_dataset = opt.path_IMG
num_non_resonant_samples = opt.num_non_resonant_samples

# In Pytorch, image shape [channels, height, width]
# Note from 1 to 245: Non-Resonant Class, from 246 to 475: Resonant Class
dim_height_img = 102
dim_width_img = 102
dim_channel_img = 3

dir_data = glob.glob(path_dataset + '/*')
num_slices = opt.num_slices
num_total_subvolumes = len(dir_data)

# data
# F x S x C x H x W
dataset = np.zeros([num_total_subvolumes, num_slices, dim_channel_img, dim_height_img, dim_width_img])

### label
Resonant_NonResonant = np.ones([num_total_subvolumes, 1])

ID_slice = []
ID_subvolume = []

subvolume = 0

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
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
    dir_img = sorted(glob.glob(path_folder + '/MultiSlices' + '/*.png'))
    for i in range (0, num_slices, 1):
        img = cv2.imread(dir_img[i])
        img = img[:, :, :].astype(np.float32)/255
        img = center_crop(img, (dim_height_img, dim_width_img))
        img = np.einsum('kli->ikl', img)
        dataset[subvolume, i, :, :, :] = img
        ID_slice.append(ID)
    subvolume += 1

# label
    Resonant_NonResonant = [0 if k < (num_non_resonant_samples + 1) else 1 for k in ID_subvolume]

dataset = dataset[:num_total_subvolumes]
Resonant_NonResonant = Resonant_NonResonant[:num_total_subvolumes]
Resonant_NonResonant = np.array(Resonant_NonResonant)
ID_slice = np.array(ID_slice)
ID_subvolume = np.array(ID_subvolume)


### Divide the test and Train data
list_index = list(range(0,475))
random.shuffle(list_index)

train_index = list_index[0:95]
test_index = list_index[95:475]

dataset_train = []
Resonant_NonResonant_train = []
dataset_test = []
Resonant_NonResonant_test = []

for index in train_index:
    dataset_train.append(dataset[index, :, :, :, :])
    Resonant_NonResonant_train.append(Resonant_NonResonant[index])

for index in test_index:
    dataset_test.append(dataset[index, :, :, :, :])
    Resonant_NonResonant_test.append(Resonant_NonResonant[index])

k = 0
for i in train_index:
  for j in test_index:
    if i == j:
      k += 1
print("Check the value", k)

dataset_train = np.array(dataset_train)
Resonant_NonResonant_train = np.array(Resonant_NonResonant_train)
dataset_test = np.array(dataset_test)
Resonant_NonResonant_test = np.array(Resonant_NonResonant_test)

# Check
print(sum(Resonant_NonResonant_train))
print(dataset_train.shape)
print(Resonant_NonResonant_train.shape)

print(sum(Resonant_NonResonant_test))
print(dataset_test.shape)
print(Resonant_NonResonant_test.shape)


# Save all the data

h5w = h5py.File(opt.path_write_h5_test, 'w')
h5w.create_dataset(name='dataset', dtype=np.float32, shape=dataset_test.shape, data=dataset_test)
h5w.create_dataset(name='Resonant_NonResonant', dtype=int, shape=Resonant_NonResonant_test.shape, data=Resonant_NonResonant_test)
h5w.close()

h5w = h5py.File(opt.path_write_h5_train_val, 'w')
h5w.create_dataset(name='dataset', dtype=np.float32, shape=dataset_train.shape, data=dataset_train)
h5w.create_dataset(name='Resonant_NonResonant', dtype=int, shape=Resonant_NonResonant_train.shape, data=Resonant_NonResonant_train)
h5w.close()

# Check
print(Resonant_NonResonant.shape)
print(dataset.shape)
print(Resonant_NonResonant.tolist())
print(sum(Resonant_NonResonant))

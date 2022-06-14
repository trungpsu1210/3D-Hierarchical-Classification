import cv2
import numpy as np
import os, glob
import h5py
import argparse
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser(description='Preprocessing training data for LSTM approach')

parser.add_argument("--path_read_h5_test",type=str, default = '/cvdata2/trung/Sonar Project/Second Stage/Implementation/First Module/H5/low_scenario_test.h5')
parser.add_argument("--path_read_h5_train",type=str, default = '/cvdata2/trung/Sonar Project/Second Stage/Implementation/First Module/H5/low_scenario_train_val.h5')

opt = parser.parse_args()
path_read_h5_test = opt.path_read_h5_test
path_read_h5_train = opt.path_read_h5_train

class DatasetFromHdf5():
  def __init__(self, file_path):
    super(DatasetFromHdf5, self).__init__()
    hf = h5py.File(file_path, 'r')
    self.data = hf.get('dataset') # F x S x C x H x W
    self.label = hf.get('Resonant_NonResonant') # (F, 1)
    self.num_subvolume = self.data.shape[0]
    self.num_img_per_subvolume = self.data.shape[1]

  def __getitem__(self, index):
    index_sub = index
    img = self.data[index_sub, :, :, :, :]
    label = [self.label[index_sub]]
    label = np.array(label)
    return torch.from_numpy(img.copy()).float(), torch.from_numpy(label).long()

  def __len__(self):
    return self.data.shape[0]

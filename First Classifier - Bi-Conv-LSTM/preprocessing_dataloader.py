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

parser = argparse.ArgumentParser(description='Preprocessing for DataLoader')

opt = parser.parse_args()
path_read_h5_test = opt.path_read_h5_test
path_read_h5_train = opt.path_read_h5_train

class DatasetFromHdf5():
  def __init__(self, file_path):
    super(DatasetFromHdf5, self).__init__()
    hf = h5py.File(file_path, 'r')
    self.data = hf.get('dataset') # B x F x C x H x W
    self.label = hf.get('Resonant_NonResonant') # (B, 1)
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

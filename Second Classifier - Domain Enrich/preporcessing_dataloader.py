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
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Preprocessing for dataloader')

class DatasetFromHdf5():
  def __init__(self, file_path):
    super(DatasetFromHdf5, self).__init__()
    hf = h5py.File(file_path, 'r')
    self.data_multislice = hf.get('dataset_multislice')
    self.data_alongtrack = hf.get('dataset_alongtrack')
    self.data_crosstrack = hf.get('dataset_crosstrack')
    self.data_label = hf.get('dataset_label')
    self.num_subvolume = self.data_multislice.shape[0]
    self.num_img_per_subvolume = self.data_multislice.shape[1]

  def __getitem__(self, index):
    index_sub = index
    img_multislice = self.data_multislice[index_sub, :, :, :, :]
    img_alongtrack = self.data_alongtrack[index_sub, :, :, :]
    img_crosstrack = self.data_crosstrack[index_sub, :, :, :]
    data_label = [self.data_label[index_sub]]
    data_label = np.array(data_label)
    return torch.from_numpy(img_multislice.copy()).float(), \
           torch.from_numpy(img_alongtrack.copy()).float(), \
           torch.from_numpy(img_crosstrack.copy()).float(), \
           torch.from_numpy(data_label).long()

  def __len__(self):
    return self.data_multislice.shape[0]

import cv2
import numpy as np
import os, glob
import h5py
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from model import *
import pytorch_msssim
from Preporcessing_fromH5 import DatasetFromHdf5
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Preprocessing data for model')

parser.add_argument("--path_read_h5_data",type=str, default = './H5/High - Aug/Train_80_pct_184_samples_Ag.h5')
parser.add_argument("--batchsize", type=int, default=4, help="Training batch size")
parser.add_argument("--save_path_result", default="./Results/High - Aug/Metrics", type=str, help="Result folder")
parser.add_argument("--save_path_csv", default="./Results/High - Aug/Metrics/results_statistic.csv", type=str, help="Save CSV result")
parser.add_argument("--save_model_path", default="./Checkpoint/High - Aug/Metrics", type=str, help="Save model path")

parser.add_argument("--lr", type=float, default=0.00016, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.75, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--num_iter_toprint", type=int, default=4, help="Training patch size")
parser.add_argument("--block_config", type=int, default=(8,12,8,8), help="Training patch size")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--resume", default="./model/dense_cbam_cmv_BloodOrCSF_onlyPIH_ct_2D3D_32_fold5of5/model_epoch_40000.pth" , type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--ID", default="", type=str, help='ID for training')

def main():

  global opt, model
  opt = parser.parse_args()

  path_read_h5_data = opt.path_read_h5_data
  batchsize = opt.batchsize
  lr = opt.lr
  momentum = opt.momentum
  threads = opt.threads
  weight_decay = opt.weight_decay
  num_epochs = opt.num_epochs
  save_model_path = opt.save_model_path
  start_epoch = opt.start_epoch
  writer = SummaryWriter('runs/Experiment')


  print('Load data')
  dataset = DatasetFromHdf5(path_read_h5_data)
  print(dataset.data_multislice)
  train_set, validation_set = torch.utils.data.random_split(dataset, [644, 92])
  train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=batchsize, shuffle=True)
  validation_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=batchsize, shuffle=True)

  device = get_default_device()

  print('Load model')

  model = Fusion_model()
  model = to_device(model, device)
  model = torch.nn.DataParallel(model).cuda()

  train_loader = DeviceDataLoader(train_loader, device)
  validation_loader = DeviceDataLoader(validation_loader, device)

  cls_criterion = nn.CrossEntropyLoss()
  re_criterion = nn.MSELoss()
  MSSSIM_criterion = pytorch_msssim.MSSSIM()

  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  total_loss = []
  val_acc_epoch = []
  train_acc_epoch = []

  for epoch in range(start_epoch, num_epochs + 1, 1):
    val_acc = check_accuracy(validation_loader, model)
    val_acc_epoch.append(val_acc)
    train_acc = check_accuracy(train_loader, model)
    train_acc_epoch.append(train_acc)
    print("==> Accuracy for validation in epoch {} is: {:.4f}".format(epoch - 1, val_acc))
    print("==> Accuracy for train in epoch {} is: {:.4f}".format(epoch - 1, train_acc))
    total_train_loss = train(train_loader, optimizer, model, epoch, cls_criterion,
                         re_criterion, MSSSIM_criterion, opt, writer)
    total_loss.append(total_train_loss)
    plot_check(val_acc_epoch, train_acc_epoch, total_loss, epoch)

def check_accuracy(dataloader, model):
  model.eval()
  accuracy = 0.0
  total = 0.0


  with torch.no_grad():
    for data in dataloader:
      MS_images, AT_images, CT_images, label = data
      outputs, _ = model(AT_images, CT_images, MS_images)
      _, predicted = torch.max(outputs.data, 1)
      total += label.size(0)
      label = label.view(-1)
      accuracy += (predicted == label).sum().item()

    # compute the metrics over all images
  accuracy = (100 * accuracy / total)
  return accuracy

def plot_check(val_acc_epoch, train_acc_epoch, average_loss, epoch):
  plt.figure()

  plt.subplot(2, 1, 1)
  plt.plot(val_acc_epoch, color='tab:red', label='Val Accuracy')
  plt.plot(train_acc_epoch, color='tab:blue', label='Train Accuracy')
  plt.ylim([0, 100])

  plt.subplot(2, 1, 2)
  plt.plot(average_loss)
  plt.ylim([0, 0.9])

  plt.savefig(os.path.join(opt.save_path_result, 'Result_after_epoch{}'.format(epoch)), dpi=200)
  plt.close()

  df = pd.DataFrame({'epoch': epoch, 'Val Accuracy': val_acc_epoch, 'Train Accuracy': train_acc_epoch,
                       'Loss': average_loss})
  df.to_csv(opt.save_path_csv)
  if epoch % 1 == 0 and epoch > 40:
    saveModel_per_epoch(epoch)

def saveModel_per_epoch(epoch):
  model_out_path = os.path.join(opt.save_model_path, "model_epoch_{}.pth".format(epoch))
  torch.save(model.state_dict(), model_out_path)


def adjust_learning_rate(epoch):
  lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
  return lr

def train(train_loader, optimizer, model, epoch, cls_criterion, re_criterion, MSSSIM_criterion, opt, writer):

    lr = adjust_learning_rate(epoch - 1)
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    ### Stage 1: Only train the reconstruction branch
    if epoch < 26:
      Freeze = False

    ### Stage 2: Train all
    else:
      Freeze = True

    for param in model.module.classifier.parameters():
      param.requires_grad = Freeze
    for param in model.module.AT_model.parameters():
      param.requires_grad = Freeze
    for param in model.module.CT_model.parameters():
      param.requires_grad = Freeze
    for param in model.module.En_DE_LSTM_model.ConvBLSTM.parameters():
      param.requires_grad = Freeze
    for param in model.module.En_DE_LSTM_model.Classifier_before.parameters():
      param.requires_grad = Freeze
    for param in model.module.En_DE_LSTM_model.Classifier_after.parameters():
      param.requires_grad = Freeze

    model.train()

    for iteration, batch in enumerate(train_loader, 1):
      loss_train = []
      MS_input_data, AT_input_data, CT_input_data, label = batch
      label = label.view(-1)

      cls_output, re_output = model(AT_input_data, CT_input_data, MS_input_data)
      cls_loss = cls_criterion(cls_output, label)

      re_loss = 0
      MSSSIM_loss = 0
      for i in range(0, MS_input_data.shape[1], 1):
        tem_MSSSIM_loss = 1.0 - MSSSIM_criterion(re_output[:, i, :, :, :], MS_input_data[:, i, :, :, :])
        tem_re_loss = re_criterion(re_output[:, i, :, :, :], MS_input_data[:, i, :, :, :])
        MSSSIM_loss += tem_MSSSIM_loss
        re_loss += tem_re_loss

      re_loss = re_loss/20
      MSSSIM_loss = MSSSIM_loss/20

      if epoch < 26:
        total_loss = re_loss + MSSSIM_loss
      else:
        total_loss = cls_loss + 0.01*re_loss + 0.005*MSSSIM_loss

      writer.add_scalar('Total_loss', total_loss, epoch * len(train_loader) + iteration)
      writer.add_scalar('Cls_loss', cls_loss, epoch * len(train_loader) + iteration)
      writer.add_scalar('Re_loss', re_loss, epoch * len(train_loader) + iteration)
      writer.add_scalar('MSSSIM_loss', MSSSIM_loss, epoch * len(train_loader) + iteration)

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      loss_train.append(total_loss.cpu().detach().numpy())
      if iteration % opt.num_iter_toprint == 0:
        print("===> Epoch[{}]({}/{},lr:{:.8f}): total_loss:{:.6f}, cls_loss:{:.6f}, re_loss:{:.6f}, MSSSIM_loss:{:.6f}".format(epoch, iteration,
                                                                    len(train_loader), lr, total_loss, cls_loss, re_loss, MSSSIM_loss))
      average_loss = np.average(loss_train)
    return average_loss


if __name__ == "__main__":
  main()
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from preprocessing_dataloader import DatasetFromHdf5
import os, glob
from model import Densenet_LSTM
import numpy as np
from utils import *
from tqdm import tqdm
import pandas as pd
import h5py
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Train the model")

parser.add_argument("--batchsize", type=int, default=8, help="Training batch size")
parser.add_argument("--num_iter_toprint", type=int, default=8, help="Print the value in training")
parser.add_argument("--path_read_data", default="Path to read data", type=str, help="Training datapath")
parser.add_argument("--save_path_result", default="Path to save results", type=str, help="Result folder")
parser.add_argument("--save_path_csv", default="Path to save result file", type=str, help="Save CSV result")
parser.add_argument("--save_model_path", default="Path to save check point file", type=str, help="Save model path")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--num_out", type=int, default=2, help="how many classes in outputs")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')

def main():
    
    global opt, model
    opt = parser.parse_args()

    batchsize = opt.batchsize
    lr = opt.lr
    momentum = opt.momentum
    threads = opt.threads
    weight_decay = opt.weight_decay
    path_read_data = opt.path_read_data
    num_epochs = opt.num_epochs
    save_model_path = opt.save_model_path
    start_epoch = opt.start_epoch
    writer = SummaryWriter('runs/Experiment') #Tensorboard
    device = get_default_device()
    
    print('Load Data')
    dataset = DatasetFromHdf5(path_read_data)
    train_set, validation_set = torch.utils.data.random_split(dataset, [72, 23])
    train_loader = DataLoader(dataset=train_set, num_workers= opt.threads, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=batchsize, shuffle=True)

    print('Load Model')
    model = Densenet_LSTM()
    model = to_device(model, device)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DeviceDataLoader(train_loader, device)
    validation_loader = DeviceDataLoader(validation_loader, device)

    val_acc_epoch = []
    train_acc_epoch = []
    ave_loss = []

    for epoch in range(start_epoch, num_epochs + 1):
        val_acc, val_sens, val_spec = check_accuracy(validation_loader, model)
        val_acc_epoch.append(val_acc)
        train_acc, train_sens, train_spec = check_accuracy(train_loader, model)
        train_acc_epoch.append(train_acc)
        print("==> Accuracy/Sensitivity/Specificity for validation in epoch {} is: {:.4f}/{:.4f}/{:.4f}".format(epoch-1,
                                                val_acc, val_sens, val_spec))
        print("==> Accuracy/Sensitivity/Specificity for train in epoch {} is: {:.4f}/{:.4f}/{:.4f}".format(epoch-1,
                                                train_acc, train_sens, train_spec))
        average_loss = train(train_loader, optimizer, model, epoch, criterion, opt, writer)
        ave_loss.append(average_loss)
        plot_check(val_acc_epoch, train_acc_epoch, ave_loss, epoch)

def plot_check(val_acc_epoch, train_acc_epoch, average_loss, epoch):
    
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(val_acc_epoch, color = 'tab:red', label = 'Val Accuracy')
    plt.plot(train_acc_epoch, color = 'tab:blue', label = 'Train Accuracy')
    plt.ylim([0, 100])

    plt.subplot(2,1,2)
    plt.plot(average_loss)
    plt.ylim([0, 0.9])

    plt.savefig(os.path.join(opt.save_path_result, 'Result_after_epoch{}'.format(epoch)), dpi=200)
    plt.close()

    df = pd.DataFrame({'epoch': epoch, 'Val Accuracy': val_acc_epoch, 'Train Accuracy': train_acc_epoch,
                       'Loss': average_loss})
    df.to_csv(opt.save_path_csv)
    if epoch%10 == 0:
        saveModel_per_epoch(epoch)

def saveModel_per_epoch(epoch):
    model_out_path = os.path.join(opt.save_model_path, "model_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)

def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return  lr

def check_accuracy(dataloader, model):
    model.eval()
    accuracy = 0.0
    total = 0.0
    sens = []
    spec = []

    with torch.no_grad():
        for data in dataloader:
            images, label = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            label = label.view(-1)
            accuracy += (predicted == label).sum().item()
            sensitivity = ((torch.sum(((predicted == 1) & (label == 1))) / torch.sum(label == 1))).float().cpu().numpy()
            specificity = ((torch.sum(((predicted == 0) & (label == 0))) / torch.sum(label == 0))).float().cpu().numpy()
            sens = np.append(sens, sensitivity)
            spec = np.append(spec, specificity)

    # compute the metrics over all images
    accuracy = (100 * accuracy / total)
    average_sens = np.average(sens)
    average_spec = np.average(spec)
    return accuracy, average_sens, average_spec

def train(training_data_loader, optimizer, model, epoch,  criterion, opt, writer):

    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        loss_tr = []
        input_data, label = batch
        label = label.view(-1)
        out = model(input_data.cuda())
        loss = criterion(out, label.cuda())
        writer.add_scalar('Training_loss_data', loss, epoch * len(training_data_loader) + iteration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr.append(loss.cpu().detach().numpy())
        if iteration%opt.num_iter_toprint == 0:
             print("===> Epoch[{}]({}/{},lr:{:.8f}): loss:{:.6f}".format(epoch, iteration,
                                                    len(training_data_loader), lr, loss))
        average_loss = np.average(loss_tr)
    return average_loss

if __name__ == "__main__":
    main()

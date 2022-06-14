import numpy as np
import math, os
import pdb
import torch
from numpy import *
import matplotlib.pyplot as plt
import itertools

def Draw_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = 100*(np.trace(cm) / float(np.sum(cm)))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names, rotation =0)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    # plt.xlabel('Predicted label\nAccuracy={:0.3f}; Misclass={:0.3f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.show()

def convert_to_0_1_scale(array):
    new_array = np.zeros((2, 2))
    new_array[0] = array[0]/array[0].sum()
    new_array[1] = array[1]/array[1].sum()
    return new_array

def checkdirctexist(dirct):
	if not os.path.exists(dirct):
		os.makedirs(dirct)

def sensitivity_specificify(correct, labels):
    sensitivity = torch.sum((correct == 1) & (labels == 1)).float()/torch.sum(labels == 1).float()
    specificity = torch.sum((correct == 1) & (labels == 0)).float()/torch.sum(labels == 0).float()
    return sensitivity, specificity

def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    pred_np = pred.cpu().data.numpy()
    gt_np = pred.cpu().data.numpy()
    rmse = math.sqrt(np.mean(imdff.cpu().data.numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1/rmse)

def adjust_learning_rate(epoch, opt):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr

def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model_files
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device

  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.dl)

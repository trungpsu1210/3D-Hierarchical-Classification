import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from preporcessing_dataloader import DatasetFromHdf5
import os, glob
from model import *
import numpy as np
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import h5py
import sklearn.metrics
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix

import pytorch_msssim

parser = argparse.ArgumentParser(description="Test the performance of model")

parser.add_argument("--batchsize", type=int, default=184, help="Training batch size")

### input NoConvLSTM
# parser.add_argument("--path_data", default="./H5/Test/288_samples_Test_32_slices.h5", type=str, help="Training datapath")
# parser.add_argument("--save_model_path", default="./Checkpoint/NoConvLSTM/288 samples/32 Slices", type=str, help="Save model path")
# parser.add_argument("--confusion_matrix", default="./Result/NoConvLSTM/288 samples/32 Slices", type=str, help="Confusion matrix")

### input ConvLSTM
# parser.add_argument("--path_data", default="./H5/Test/288_samples_Test_32_slices.h5", type=str, help="Training datapath")
# parser.add_argument("--save_model_path", default="./Checkpoint/ConvLSTM/288 samples/32 Slices", type=str, help="Save model path")
# parser.add_argument("--confusion_matrix", default="./Result/ConvLSTM/288 samples/32 Slices", type=str, help="Confusion matrix")

### input Bidirectional ConvLSTM
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--path_data", default="./H5/High - Aug/Test_20_pct_46_samples.h5", type=str, help="Training datapath")
parser.add_argument("--save_model_path", default="./Checkpoint/High - Aug/0.002-0.001-0.00016-0.75", type=str, help="Save model path")
#parser.add_argument("--save_model_path", default="./Checkpoint/High - No Aug/Metrics", type=str, help="Save model path")
parser.add_argument("--confusion_matrix", default="./Results/High - Aug/0.002-0.001-0.00016-0.75", type=str, help="Confusion matrix")
parser.add_argument("--cuda", type=str, default='0')

def main():

    global opt, model
    opt = parser.parse_args()

    save_model_path = opt.save_model_path
    path_data = opt.path_data
    batchsize = opt.batchsize

    device = get_default_device()

    ## Load model
    print('Load model')
    model = Fusion_model()
    model = to_device(model, device)
    model = torch.nn.DataParallel(model).cuda()

    ## Load data
    print('Load data')
    dataset = DatasetFromHdf5(path_data)
    dataloader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=batchsize, shuffle=False)
    dataloader = DeviceDataLoader(dataloader, device)

    i = 0
    for checkpoint_save_path in glob.glob(save_model_path + '/*'):
        model.load_state_dict(torch.load(checkpoint_save_path))
        model.eval()

        save_name = checkpoint_save_path.split('/')[-1]

        accuracy = check_accuracy(dataloader, model)
        print("==>Test, Accuracy: {:.3f}".format(
                accuracy))
        print(checkpoint_save_path)

        # Draw_confusion_matrix(cm= Array_CM, normalize=True,
        #                   target_names=['Resonant', 'NonResonant'],
        #                   title='Confusion Matrix for epoch {}'.format(save_name), path=i)
        i += 1


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def check_accuracy(dataloader, model):
  model.eval()
  accuracy = 0.0
  total = 0.0


  with torch.no_grad():
    for data in dataloader:
      MS_images, AT_images, CT_images, label = data
      outputs, re_outputs = model(AT_images, CT_images, MS_images)
      _, predicted = torch.max(outputs.data, 1)
      total += label.size(0)
      label = label.view(-1)
      accuracy += (predicted == label).sum().item()
      cf_matrix = confusion_matrix(label.cpu().numpy(), predicted.cpu().numpy())
      print(cf_matrix)

    # compute the metrics over all images
  accuracy = (100 * accuracy / total)

  re_criterion = nn.MSELoss()
  MSSSIM_criterion = pytorch_msssim.MSSSIM()

  print(re_criterion(MS_images[10, 10, :, :, :].cpu(), re_outputs[10, 10, :, :, :].cpu()))
  print(MSSSIM_criterion(MS_images[:, 10, :, :, :].cpu(), re_outputs[:, 10, :, :, :].cpu()))


  # image = MS_images[10, 10, :, :, :].cpu()
  # image = image.permute(1, 2, 0)
  # image = np.array(image)
  # image1 = rgb2gray(image)
  # plt.imshow((image1 * 255).astype(np.uint8), cmap="gray")
  # plt.show()
  #
  # image = re_outputs[10, 10, :, :, :].cpu()
  # image = image.permute(1, 2, 0)
  # image = np.array(image)
  # image2 = rgb2gray(image)
  # plt.imshow((image2 * 255).astype(np.uint8), cmap="gray")
  # plt.show()

  return accuracy

def Draw_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, path = 0):
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
    plt.savefig(os.path.join(opt.confusion_matrix, 'Confusion Matrix Result {}'.format(path)), dpi=200)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()

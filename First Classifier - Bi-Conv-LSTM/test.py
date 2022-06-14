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
from Bi_Conv_LSTM_model import Densenet_LSTM
import numpy as np
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import h5py
import sklearn.metrics
from sklearn.metrics import plot_precision_recall_curve

parser = argparse.ArgumentParser(description="Test the performance of model")

parser.add_argument("--batchsize", type=int, default=380, help="Testing batch size")
parser.add_argument("--num_iter_toprint", type=int, default=30, help="Print the loss metric in training")
parser.add_argument("--path_data", default="Path to H5py", type=str, help="Testing datapath")
parser.add_argument("--save_model_path", default="Path to save the model results", type=str, help="Save model path")
parser.add_argument("--confusion_matrix", default="Path to save the model results, type=str, help="Confusion matrix")
parser.add_argument("--num_out", type=int, default=2, help="how many classes in outputs?")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
                    
def main():
                    
    global opt, model
    opt = parser.parse_args()
                    
    save_model_path = opt.save_model_path
    path_data = opt.path_data
    batchsize = opt.batchsize

    print('Load model')
    device = get_default_device()
    model = Densenet_LSTM()
    model = to_device(model, device)
    model = torch.nn.DataParallel(model).cuda()

    print('Load data')
    dataset = DatasetFromHdf5(path_data)
    dataloader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=batchsize, shuffle=False)
    dataloader = DeviceDataLoader(dataloader, device)

    i = 0
    for checkpoint_save_path in glob.glob(save_model_path + '/*'):
                    
        model.load_state_dict(torch.load(checkpoint_save_path))
        model.eval()
        save_name = checkpoint_save_path.split('/')[-1]

        accuracy, sensitivity, specificity, Array_CM, AUCPR = check_accuracy(dataloader, model)
        print("==>Test, Accuracy: {:.3f}/Sensitivity: {:.3f}/Specificity: {:.3f}/AUCPR: {:.3f}".format(
                accuracy, sensitivity, specificity, AUCPR))
        print(checkpoint_save_path)

        Draw_confusion_matrix(cm= Array_CM, normalize=True,
                          target_names=['Resonant', 'NonResonant'],
                          title='Confusion Matrix for epoch {}'.format(save_name), path=i)
        i += 1

def check_accuracy(dataloader, model):
    model.eval()
    accuracy = 0.0
    total = 0.0
    sens = []
    spec = []
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    Array_confusion_matrix = np.zeros((2,2))

    with torch.no_grad():
        for batch in dataloader:
            images, label = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            label = label.view(-1)
            accuracy += (predicted == label).sum().item()
            TP = torch.sum((predicted == 1) & (label == 1)).float().cpu().numpy()
            TN = torch.sum((predicted == 0) & (label == 0)).float().cpu().numpy()
            FP = torch.sum((predicted == 1) & (label == 0)).float().cpu().numpy()
            FN = torch.sum((predicted == 0) & (label == 1)).float().cpu().numpy()
            Array_confusion_matrix[0] = TP, FN
            Array_confusion_matrix[1] = FP, TN
            Array_confusion_matrix = convert_to_0_1_scale(Array_confusion_matrix)
            AUPRC = sklearn.metrics.average_precision_score(label.float().cpu().numpy(), outputs[:,1].float().cpu().numpy())
            print(AUPRC)
            sensitivity = ((torch.sum(((predicted == 1) & (label == 1))) / torch.sum(label == 1))).float().cpu().numpy()
            specificity = ((torch.sum(((predicted == 0) & (label == 0))) / torch.sum(label == 0))).float().cpu().numpy()
            sens = np.append(sens, sensitivity)
            spec = np.append(spec, specificity)

    # compute the metrics over all images
    accuracy = (100 * accuracy / total)
    average_sens = np.average(sens)
    average_spec = np.average(spec)

    return accuracy, average_sens, average_spec, Array_confusion_matrix, AUPRC

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

# Hierarchical classification for 3D image data #

This code originally is proposed for hierarchical classification for 3D sonar imagery. The two models are developed based on Bidirectional Convolutional LSTM for sequence data (multi 2D slices/frames). Besides, to enhance the performance of the feature extraction network, I proposed a new auxiliary task, hence the training data is learned via cross-entropy, msssim, and l2 loss function. Two side views (along-track and cross-track) are also corporated by the fusion technique 

## Flow chart of proposed idea is here ##

![alt text](https://github.com/trungpsu1210/3D-Hierarchical-Classification/blob/main/FlowChart.png)

## Structure of the code ##

1. First Classifier - Bi-Conv-LSTM and Second Classifier - Domain Enrich: Two proposed methods, each folders will have
* model.py: proposed model
* create_HDF5.py: convert all the data and label to H5py files
* preprocessing_dataloader.py: preprocessed data
* train.py: train the model
* test.py: test the performance
2. Checkpoint, H5, Results: save all the corresponding files to here
3. pytorch-msssim: designed the msssim loss function
4. Visualization.ipynb: code for drawing the figures with latex fonts (bar, chart, line, confusion matrix,...) used for papers
5. utils.py: useful function using in the code

## Requirments ##

Pytorch 1.9.0

Python 3.8

Deep learning libraries/frameworks: OpenCV, HDF5, TensorBoard,Pandas,...

To run the code, make sure all the files are in the corresponding folders

 


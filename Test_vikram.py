#!/usr/bin/env python


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import os
import sys
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import CIFAR10Model_ResNet
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = np.float32(cv2.imread(ImageName))
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    #I1S = iu.StandardizeInputs(np.float32(I1))
    I1_mean = np.mean(I1,axis=(1,2),keepdims=True)
    I1_std = np.std(I1,axis=(1,2),keepdims=True)
    I1S = (I1-I1_mean)/(I1_std+np.exp(-5))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, total_epochs):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = CIFAR10Model_ResNet(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()
    # print("\ncalled\n")

    
    with tf.Session() as sess:
        for epoch in range(total_epochs):
            ModelPath_ = ModelPath + str(epoch) + "model.ckpt"
            Saver.restore(sess, ModelPath_)
            print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            
            OutSaveT1 = open(LabelsPathPred+str(epoch)+'.txt', 'w')
            OutSaveT2 = open(LabelsPathPred+'.txt', 'a')

            for count in tqdm(range(np.size(DataPath))):            
                DataPathNow = DataPath[count]
                Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
                FeedDict = {ImgPH: Img}
                PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
                # print(PredT)

                OutSaveT1.write(str(PredT)+'\n')
                OutSaveT2.write(str(PredT)+'\n')
                
            OutSaveT1.close()
            OutSaveT2.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, plot_path, operation):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

    cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.arange(10))
    cm_plot.plot()
    if operation==0:
        plt.savefig(plot_path + 'train_cm.png')
    else:
        plt.savefig(plot_path + 'test_cm.png')

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='checkpoints/resnet/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../CIFAR10', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPathTrain', dest='LabelsPathTrain', default='./TxtFiles/LabelsTrain.txt', help='Path of labels file, Default:./TxtFiles/LabelsTrain.txt')
    Parser.add_argument('--LabelsPathTest', dest='LabelsPathTest', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--total_epochs', type = int, default=20, help='Number of iterations of training/testing')
    Parser.add_argument('--plot_path', default='plots/resnet/', help='Path to save training loss and accuracy plots, Default=plots/')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPathTrain = Args.LabelsPathTrain
    LabelsPathTest = Args.LabelsPathTest
    plot_path = Args.plot_path
    total_epochs = Args.total_epochs

    # Setup all needed parameters including file reading
    ImageSize, DataPath_train = SetupAll(BasePath + '/Train/')
    ImageSize, DataPath_test = SetupAll(BasePath + '/Test/')

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred_train = './TxtFiles/training_labels_pred/resnet/pred' # Path to save predicted labels
    LabelsPathPred_test = './TxtFiles/testing_labels_pred/resnet/pred' # Path to save predicted labels

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath_train, LabelsPathPred_train, total_epochs)
    tf.reset_default_graph()
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    TestOperation(ImgPH, ImageSize, ModelPath, DataPath_test, LabelsPathPred_test, total_epochs)

    testing_epochs = []
    testing_accuracy = []
    for i in range(total_epochs):
        testing_epochs.append(i)
        true_labels_train, pred_labels_train = ReadLabels(LabelsPathTrain,LabelsPathPred_train+str(i)+'.txt')
        true_labels_test, pred_labels_test = ReadLabels(LabelsPathTest,LabelsPathPred_test+str(i)+'.txt')
        testing_accuracy.append(Accuracy(list(true_labels_test),list(pred_labels_test)))
        if i==total_epochs-1:
            true_labels_train, pred_labels_train = ReadLabels(LabelsPathTrain,LabelsPathPred_train+str(i)+'.txt')
            true_labels_test, pred_labels_test = ReadLabels(LabelsPathTest,LabelsPathPred_test+str(i)+'.txt')
            ConfusionMatrix(list(true_labels_train),list(pred_labels_train),plot_path,0)
            ConfusionMatrix(list(true_labels_test),list(pred_labels_test),plot_path,1)

    plt.figure()
    plt.title('Variation of Testing Accuracy Through Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim((0,total_epochs))
    plt.plot(testing_epochs,testing_accuracy)
    plt.savefig(plot_path+'testing_cm.png')

    # Plot Confusion Matrix
    #LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    #ConfusionMatrix(LabelsTrue, LabelsPred, plot_path)
     
if __name__ == '__main__':
    main()
 

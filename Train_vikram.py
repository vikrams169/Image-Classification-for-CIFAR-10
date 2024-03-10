#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import os
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import CIFAR10Model_ResNet
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    BasePath - Path to CIFAR10 folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.png'   
        ImageNum += 1
    	
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################

        I1 = np.float32(cv2.imread(RandImageName))
        I1_mean = np.mean(I1,axis=(1,2),keepdims=True)
        I1_std = np.std(I1,axis=(1,2),keepdims=True)
        I1S = (I1-I1_mean)/(I1_std+np.exp(-5))
        Label = convertToOneHot(TrainLabels[RandIdx], 10)

        # Append All Images and Mask
        I1Batch.append(I1S)
        LabelBatch.append(Label)
        
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, plot_path):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to CIFAR10 folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    prLogits, prSoftMax = CIFAR10Model_ResNet(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prLogits,labels=LabelPH))

    with tf.name_scope('Accuracy'):
        prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        LabelDecoded = tf.argmax(LabelPH, axis=1)
        Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
        
    with tf.name_scope('Adam'):
    	###############################################
    	# Fill your optimizer of choice here!
    	###############################################
        Optimizer = Optimizer = tf.train.AdamOptimizer(learning_rate = 0.0015).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.scalar('Accuracy', Acc)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver(max_to_keep=50)
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        training_epochs = []
        training_loss = []
        training_accuracy = []
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            loss_epoch = []
            accuracy_epoch = []
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                loss_epoch.append(LossThisBatch)
                accuracy_epoch.append(sess.run(Acc,feed_dict=FeedDict)*100)
                
                # # Save checkpoint every some SaveCheckPoint's iterations
                # if PerEpochCounter % SaveCheckPoint == 0:
                #     # Save the Model learnt in this epoch
                #     SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                #     Saver.save(sess,  save_path=SaveName)
                #     print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
            training_epochs.append(Epochs)
            training_loss.append(np.mean(np.array(loss_epoch)))
            training_accuracy.append(np.mean(np.array(accuracy_epoch)))
            print("\n")
            print("Epoch:",Epochs)
            print("Loss:",training_loss[-1])
            print("Accuracy:",training_accuracy[-1])

            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')

        plt.subplots(1,2,figsize=(20,20))
        plt.subplot(1,2,1)
        plt.title('Variation of Training Loss Through Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim((0,NumEpochs))
        plt.plot(training_epochs,training_loss)
        plt.subplot(1,2,2)
        plt.title('Variation of Training Accuracy Through Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.xlim((0,NumEpochs))
        plt.plot(training_epochs,training_accuracy)
        plt.savefig(plot_path + 'training_plots.png')

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


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../CIFAR10', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/CIFAR10')
    Parser.add_argument('--CheckPointPath', default='checkpoints/resnet/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=20, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='logs/resnet/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--plot_path', default='plots/resnet/', help='Path to save training loss and accuracy plots')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    plot_path = Args.plot_path

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
    TrainLabels = list(TrainLabels)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, plot_path)   
    
if __name__ == '__main__':
    main()
 

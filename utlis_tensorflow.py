import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.image as mpimg
from imgaug import augmenters as iaa  # Fixed typo

import random

##--------------------------------- Autonimous Car ----------------------------------------------------------------------------##

## Step 1 _ Initialize Data _ Nuaral Network
def getName(filePath):
    myImagePathL =filePath.split('/')[-2:]
    myImagePath = os.path.join(myImagePathL[0], myImagePathL[1])
    return myImagePath


def importDataInfo(path):
    columns = ['Center', 'Steering']
    noOfFolders = len(os.listdir(path))//2  
    data = pd.DataFrame()
    for x in range(17, 22):
        dataNew = pd.read_csv(os.path.join(path, f'log_{x}.csv'), frames_ = columns)
        print(f'{x}:{dataNew.shpae[0]} ',end='')
        #Remove File Path and Get File Name
        dataNew['Center']=dataNew['Center'].apply(getName)
        data =data.append(dataNew, True  )
    print(' ')
    print('Total Images Imported', data.shape[0])
    return data

# Step 2 - Visualize and Balance Data
def balanceData(data, display = True):
    nBin = 31
    samplesPerBin = 300
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width = 0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title("Data Visualisation")
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)


    print('Removed images: ', len(removeindexList))
    data.drop(data.index[removeindexList], inplace = True)
    print("Remaining images: ",len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist, width=0.03)

        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title("Data Visualisation")
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data

# Step 3 - Prepare for Proccesing
def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append( os.path.join(path, indexed_data[0]))
        steering.append(float(indexed_data[1]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

#Step 5 - Augment Data

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering

    return img, steering

# imgRe, st = augmentImage('test.jpg',0)
# mpimg.imsave('Results.jpg', imgRe)
# plt.imshow(imgRe)
# plt.show


# Step 6 - Pre Process
def preProcess(img):
    img = img[54:120,:,:]
    img = cv2.cvtColor(img, cv2.Color_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# imgRe = preProcess(mpimg.imread('test.jpg'))
# mpimg.imsave('Results.jpg', imgRe)
# plt.imshow(imgRe)
# plt.show

# step 7 - Create Model
def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64 ,(3, 3), activation='elu'))
    model.add(Convolution2D(64 ,(3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))

    model.compile(Adam(lr = 0.0001), loss = 'mse')
    return model

# Step 8 - Training
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Convolution2D, Flatten, Dense
#from tensorflow.keras.optimizers import Adam

import matplotlib.image as mpimg
#from imgaug import augmenters as iaa  # Fixed typo

import random


def initializeTrackbars(initialTrackbarVals, wT = 480, hT = 240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, nothing)

def valTrackbars(wT = 480, hT = 240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([
        (widthTop, heightTop),
        (wT - widthTop, heightTop),
        (widthBottom, heightBottom),
        (wT - widthBottom, heightBottom)
    ])
    return points

def warpImg(img, points, w, h):
    try:
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warpedImg = cv2.warpPerspective(img, matrix, (w, h))
        return warpedImg
    except cv2.error as e:
        print(f"Error in warpImg: {e}")
        return img

def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 5, (0, 0, 255), cv2.FILLED)
    return img

def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,0])
    upper_white = np.array([179,255,189])
    mask = cv2.inRange(imgHsv, lower_white, upper_white)
    return mask

def draw_perspective_lines(img, pts1, pts2):
    """
    Draw the perspective lines on the image.
    Args:
        img: The image where lines will be drawn.
        pts1: Source points for the perspective.
        pts2: Destination points for the warped perspective.
    """
    for i in range(len(pts1)):
        # Ensure points are integers for cv2.line
        pt1 = tuple(np.int32(pts1[i]))
        pt2 = tuple(np.int32(pts2[i]))

        # Draw lines between the perspective points in the source image
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Draw lines between the corresponding points in pts1 and pts2
        cv2.line(img, pt1, tuple(np.int32(pts1[(i+1)%4])), (0, 255, 0), 2)  # Connecting the source points
        cv2.line(img, pt2, tuple(np.int32(pts2[(i+1)%4])), (0, 0, 255), 2)  # Connecting the destination points
    
    return img

def nothing(a):
    pass

def getHistogram(img , minPer=0, display= False):
    histvalues = np.sum(img, axis=0)
    #print(histvalues)
    maxValue = np.max(histvalues)
    minValue = minPer*maxValue

    indexArray = np.where(histvalues >= minValue)
    basePoint = int(np.average(indexArray))
    print(basePoint)

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histvalues):
            cv2.line(imgHist,(x,img.shape[0]), (x,img.shape[0] - intensity//255), (255,0,255), 1)

        return basePoint, imgHist
    return basePoint

##--------------------------------- Autonimous Car ----------------------------------------------------------------------------##

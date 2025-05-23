import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
import logging  # Added for better error handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== Lane Detection Functions ====================
def nothing(a): pass

def initializeTrackbars(initialTrackbarVals, wT=480, hT=240):
    """Initialize trackbars for perspective transform adjustment"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    """Get current trackbar values"""
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    
    # Validate points
    points = np.float32([
        (max(0, min(widthTop, wT)), max(0, min(heightTop, hT))),
        (max(0, min(wT - widthTop, wT)), max(0, min(heightTop, hT))),
        (max(0, min(widthBottom, wT)), max(0, min(heightBottom, hT))),
        (max(0, min(wT - widthBottom, wT)), max(0, min(heightBottom, hT)))
    ])
    return points

def warpImg(img, points, w, h):
    """Perform perspective warp with error handling"""
    try:
        if len(points) != 4:
            raise ValueError("Exactly 4 points required for perspective transform")
            
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (w, h))
    except Exception as e:
        logging.error(f"Error in warpImg: {str(e)}")
        return img
    
def showWarpPreview(img, wT=480, hT=240):
    points = valTrackbars(wT, hT)
    imgWarp = warpImg(img, points, wT, hT)
    imgPoints = drawPoints(img.copy(), points)
    stacked = np.hstack((imgPoints, imgWarp))
    cv2.imshow("Warp Preview", stacked)




def drawPoints(img, points, color=(0, 0, 255), size=5):
    """Draw perspective points on image"""
    for pt in points:
        x, y = map(int, pt)
        cv2.circle(img, (x, y), size, color, cv2.FILLED)
    return img

def thresholding(img):
    """Improved thresholding with morphological operations"""
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])  # Adjusted for better detection
    upper_white = np.array([179, 30, 255])
    
    mask = cv2.inRange(imgHsv, lower_white, upper_white)
    
    # Add morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    return mask

def getHistogram(img, minPer=0.1, display=False):
    """Calculate histogram with smoothing"""
    histValues = np.sum(img, axis=0)
    
    # Smooth histogram
    histValues = cv2.GaussianBlur(histValues, (5, 5), 0)
    
    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]), 
                     (x, img.shape[0] - int(intensity//255)), 
                     (255, 0, 255), 1)
        return basePoint, imgHist
    return basePoint

# ==================== Autonomous Car Functions ====================

def getName(filePath):
    """Improved path handling using os.path"""
    return os.path.join(os.path.basename(os.path.dirname(filePath)), 
                        os.path.basename(filePath))

def importDataInfo(path):
    """Load CSV data with error handling"""
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    
    try:
        # Get all CSV files in path
        csv_files = [f for f in os.listdir(path) if f.startswith('log_') and f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(path, csv_file)
            dataNew = pd.read_csv(file_path, names=columns)  # Fixed parameter name
            
            # Validate required columns
            if not all(col in dataNew.columns for col in columns):
                raise ValueError(f"Missing columns in {csv_file}")
                
            dataNew['Center'] = dataNew['Center'].apply(getName)
            data = pd.concat([data, dataNew], ignore_index=True)  # Fixed deprecated append
            
        logging.info(f"Total images imported: {data.shape[0]}")
        return data
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def balanceData(data, nBin=31, samplesPerBin=300, display=True):
    """Balanced data with configurable parameters"""
    if data.empty:
        return data
        
    hist, bins = np.histogram(data['Steering'], nBin)
    removeIndexList = []

    # Iterate through each bin and select the samples
    for j in range(nBin):
        # Select the indices within the bin range
        binDataList = data[(data['Steering'] >= bins[j]) & (data['Steering'] <= bins[j+1])].index.tolist()

        # Shuffle indices within the bin
        binDataList = shuffle(binDataList)

        # Keep only the first `samplesPerBin` samples, remove the rest
        removeIndexList.extend(binDataList[samplesPerBin:])

    # Drop the excess samples to balance the data
    dataBalanced = data.drop(data.index[removeIndexList])
    
    if display:
        # Plot histograms of original and balanced data distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['Steering'], bins=nBin)
        plt.title("Original Data Distribution")
        
        plt.subplot(1, 2, 2)
        plt.hist(dataBalanced['Steering'], bins=nBin)
        plt.title("Balanced Data Distribution")
        plt.show()
        
    return dataBalanced

def loadData(path, data):
    """Load data with path validation"""
    imagesPath = []
    steering = []
    
    for i in range(len(data)):
        indexedData = data.iloc[i]
        fullPath = os.path.join(path, indexedData[0])
        
        if os.path.exists(fullPath):
            imagesPath.append(fullPath)
            steering.append(float(indexedData[1]))
        else:
            logging.warning(f"Missing file: {fullPath}")
            
    return np.array(imagesPath), np.array(steering)

def augmentImage(imgPath, steering):
    """Improved augmentation with OpenCV"""
    img = cv2.imread(imgPath)
    if img is None:
        logging.warning(f"Failed to read {imgPath}")
        return np.zeros((66, 200, 3)), 0  # Or skip
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Random transformations
    if np.random.rand() < 0.5:
        # Pan
        tx = 100 * np.random.uniform(-0.1, 0.1)
        ty = 100 * np.random.uniform(-0.1, 0.1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
    if np.random.rand() < 0.5:
        # Zoom
        zoom = np.random.uniform(1, 1.2)
        h, w = img.shape[:2]
        newW, newH = int(w/zoom), int(h/zoom)
        img = cv2.resize(img, (newW, newH))
        img = cv2.copyMakeBorder(img, 
                               (h-newH)//2, h-newH-(h-newH)//2,
                               (w-newW)//2, w-newW-(w-newW)//2,
                               cv2.BORDER_REPLICATE)
                               
    if np.random.rand() < 0.5:
        # Flip
        img = cv2.flip(img, 1)
        steering = -steering
        
    return img, steering

def preProcess(img):
    """Fixed color conversion typo and improved preprocessing"""
    img = img[54:120, :, :]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Fixed typo
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = (img / 127.5) - 1.0  # Better normalization
    return img

def create_model(input_shape):
    model = Sequential()

    # Convolutional Layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))  # Drop 50% during training

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    # Output Layer (for regression)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    """Improved generator with shuffling"""
    while True:
        # Shuffle at start of each epoch
        if trainFlag:
            imagesPath, steeringList = shuffle(imagesPath, steeringList)
            
        imgBatch = []
        steeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = cv2.imread(imagesPath[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steeringList[index]
                
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
            
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def nothing(a):
    pass

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Example usage
    data = importDataInfo("data")
    if not data.empty:
        data = balanceData(data)
        imagesPath, steering = loadData("data", data)
        
        model = createModel()
        model.summary()
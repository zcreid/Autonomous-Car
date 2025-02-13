import pandas as pd
import os 
import numpy as np
import cv2 
from datetime import datetime

global imgList, steeringList
countFolder = 0
count = 0
imgList = []    
steeringList = []

#Get current Dir Path
myDiectory = os.path.join(os.getcwd(), 'DataCollected') #Get Current Working Directory      
#print(myDiectory)

#Get a New Folder Based on Previous Folder count
while os.path.exists(os.path.join(myDiectory, f'IMG{countFolder}')):
    countFolder += 1
newPath = os.path.join(myDiectory, f'IMG{countFolder}')
os.makedirs(newPath)


# Save Images in Folder
def saveData(img, steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print('Timestamp =', timestamp)
    filename = os.path.join(newPath, f'Image_{timestamp}.jpg')
    cv2.imwrite(filename, img)
    imgList.append(filename)
    steeringList.append(steering)



# Save Log File When Seshion Ends
def saveLog(img, steering):
    global imgList, steeringList
    rawData = {'Image': imgList, 'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDiectory, f'log_{countFolder}.csv'), index = False, header = False)
    print('Log Saved')
    print('Total Images: ', len(imgList))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    for x in range(30):
        _, img = cap.read()
        cv2.imshow('Image', img)
        saveData(img, 0.5)
        cv2.waitKey(1)
    saveLog()


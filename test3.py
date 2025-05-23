import pandas as pd
import os
import cv2
from datetime import datetime

# Global variables
imgList = []
steeringList = []

# Get current directory path
DATA_DIR = os.path.join(os.getcwd(), "DataCollected")
os.makedirs(DATA_DIR, exist_ok=True)

# Get a new folder based on the previous folder count
countFolder = 0
while os.path.exists(os.path.join(DATA_DIR, f"IMG{countFolder}")):
    countFolder += 1

newPath = os.path.join(DATA_DIR, f"IMG{countFolder}")
os.makedirs(newPath)

def saveData(img, steering):
    """ Saves images and steering data """
    global imgList, steeringList
    timestamp = str(datetime.timestamp(datetime.now())).replace(".", "")
    filename = os.path.join(newPath, f"Image_{timestamp}.jpg")
    
    cv2.imwrite(filename, img)
    imgList.append(filename)
    steeringList.append(steering)

def saveLog():
    """ Saves the log file when the session ends """
    global imgList, steeringList
    df = pd.DataFrame({"Image": imgList, "Steering": steeringList})
    log_file = os.path.join(DATA_DIR, f"log_{countFolder}.csv")
    df.to_csv(log_file, index=False, header=False)
    
    print(f"Log Saved: {log_file}")
    print(f"Total Images: {len(imgList)}")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    for _ in range(30):
        ret, img = cap.read()
        if ret:
            cv2.imshow("Image", img)
            saveData(img, 0.5)
            cv2.waitKey(1)
    saveLog()

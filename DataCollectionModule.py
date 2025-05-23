import pandas as pd
import os 
import cv2 
from datetime import datetime
from UltrasonicModule import setup_ultrasonic, get_front_distance, get_rear_distance, cleanup_ultrasonic

global imgList, steeringList, frontDistList, rearDistList
countFolder = 0
count = 0
imgList = []    
steeringList = []
frontDistList = []  # Stores front ultrasonic readings (mm)
rearDistList = []   # Stores rear ultrasonic readings (mm)

# Directory setup (unchanged)
myDirectory = os.path.join(os.getcwd(), 'DataCollected')
while os.path.exists(os.path.join(myDirectory, f'IMG{countFolder}')):
    countFolder += 1
newPath = os.path.join(myDirectory, f'IMG{countFolder}')
os.makedirs(newPath)

def saveData(img, steering):
    global imgList, steeringList, frontDistList, rearDistList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    filename = os.path.join(newPath, f'Image_{timestamp}.jpg')
    cv2.imwrite(filename, img)
    
    # Get distances from both sensors
    front_mm = get_front_distance()
    rear_mm = get_rear_distance()
    
    imgList.append(filename)
    steeringList.append(steering)
    frontDistList.append(front_mm)
    rearDistList.append(rear_mm)

def saveLog():
    global imgList, steeringList, frontDistList, rearDistList
    rawData = {
        'Image': imgList,
        'Steering': steeringList,
        'FrontDistance_mm': frontDistList,
        'RearDistance_mm': rearDistList
    }
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory, f'log_{countFolder}.csv'), index=False, header=False)
    print('Log Saved')
    print(f'Total Entries: {len(imgList)}')
    #print(f'Front Distances: {len(frontDistList)}')
    #print(f'Rear Distances: {len(rearDistList)}')

if __name__ == '__main__':
    setup_ultrasonic()  # Initialize both sensors
    cap = cv2.VideoCapture(0)
    
    try:
        for x in range(30):  # Example: Capture 30 frames
            _, img = cap.read()
            cv2.imshow('Image', img)
            saveData(img, 0.5)  # Replace 0.5 with actual steering input
            cv2.waitKey(1)
    finally:
        saveLog()
        cleanup_ultrasonic()
        cap.release()
        cv2.destroyAllWindows()

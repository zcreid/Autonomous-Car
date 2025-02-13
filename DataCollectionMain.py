import WebcamModule as wM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import MotorModule as mM
import cv2
from time import sleep

maxThrottle = 0.25
motor = mM.Motor(17, 27, 22, 10, 11, 9)

record = 0
while True:
    joyVal = jsM.getJS()
    #print(joyVal)
    steering = joyVal['axis1']
    throttle = joyVal['o']*maxThrottle
    if joyVal['share'] == 1:
        if record ==0: print('Recording Started ...')
        record += 1
        sleep(0.3)

    if record == 1:
        img = wM.getImg(display=True, size=(240, 120))
        dcM.saveData(img, joyVal['axis1'])
    elif record == 2:
        dcM.saveLog()
        record = 0


    mortor.move(throttle, -steering)
    cv2
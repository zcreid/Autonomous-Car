import cv2
import utlis 

def getLaneCurve(img):

    imgCopy = img.copy()

    #Step 1
    imgThres = utlis.thresholding(img)

    #Step 2
    h, w, c, = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, w, h)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    # Step 3 
    basePoint, imgHist = utlis.getHistogram(imgWarp, display =True)

    cv2.imshow('thres', imgThres)
    cv2.imshow('warp', imgWarp)
    cv2.imshow('warp Points', imgWarpPoints)
    cv2.imshow('Histogram', imgHist)

    return None

feed = r'C:\Users\Cloud\Desktop\Lane_Detect\Images\vid1.mp4'

if __name__ == '__main__':
    cap = cv2.VideoCapture(feed)
    intialTrackBarVals = [0,0,0,117]
    utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        succes, img = cap.read()
        img = cv2.resize(img,(480,240))
        getLaneCurve(img)

        cv2.imshow('vid', img)
        cv2.waitKey(1)
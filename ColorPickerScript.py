import cv2
import numpy as np

# Set frame dimensions
frameWidth = 640
frameHeight = 480


# Initialize video capture for a video file
video_path = r'C:\Users\Cloud\Desktop\\Images\vid1.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(3, frameWidth)  # Set width
cap.set(4, frameHeight)  # Set height

# Check if video file is loaded
if not cap.isOpened():
    print(f"Error: Cannot open video file at {video_path}")
    exit()

# Trackbar callback function
def nothing(a):
    pass

# Create HSV adjustment trackbars
cv2.namedWindow("HSV Trackbars")
cv2.resizeWindow("HSV Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "HSV Trackbars", 0, 179, nothing)
cv2.createTrackbar("Hue Max", "HSV Trackbars", 179, 179, nothing)
cv2.createTrackbar("Sat Min", "HSV Trackbars" , 0, 255, nothing)
cv2.createTrackbar("Sat Max", "HSV Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Min", "HSV Trackbars", 0, 255, nothing)
cv2.createTrackbar("Val Max", "HSV Trackbars", 255, 255, nothing)

# Frame counter for looping the video
frameCounter = 0

while True:
    # Restart video if it reaches the end
    frameCounter += 1
    if frameCounter >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    # Read a frame from the video
    _, img = cap.read()
    # Convert the frame to HSV
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    h_min = cv2.getTrackbarPos("Hue Min", "HSV Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "HSV Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "HSV Trackbars")

    # Print HSV values for debugging
    print(h_min)

    # Create mask based on HSV range
        # Create mask based on HSV range
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)

    # Apply mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert mask to BGR for stacking
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize images to ensure consistent dimensions for stacking
    img_resized = cv2.resize(img, (frameWidth, frameHeight))
    mask_bgr_resized = cv2.resize(mask_bgr, (frameWidth, frameHeight))
    result_resized = cv2.resize(result, (frameWidth, frameHeight))

    # Horizontally stack the images
    hStack = np.hstack([img_resized, mask_bgr_resized, result_resized])

    # Display the stacked frames
    cv2.imshow("Horizontal Stacking", hStack)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

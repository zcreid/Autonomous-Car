import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Camera failed to open")
    exit()

os.makedirs("video", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"video/{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
    cap.release()
    exit()

height, width = frame.shape[:2]
out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

print("Recording for 10 seconds...")

for _ in range(200):  # ~10 seconds at 20 fps
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved: {filename}")

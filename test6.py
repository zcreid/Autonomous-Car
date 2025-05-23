import cv2
import pygame
import RPi.GPIO as GPIO
from time import sleep
from JoyStickModule import getJS
from WebcamModule import getImg
from DataCollectionModule import saveData, saveLog
from MotorModule import Motor  # Assuming Motor class is in MotorModule.py
import os
from datetime import datetime

# GPIO Setup
LED_PIN = 17  # LED connected to GPIO 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED is off initially

# Initialize Motor
motor = Motor(12, 1, 7, 13, 6, 5)

def main():
    recording = False  # Track recording state
    video_writer = None
    video_folder = "video"
    os.makedirs(video_folder, exist_ok=True)
    prev_x_button = False  # Track previous button state

    print("Waiting to Start Recording...")

    try:
        while True:
            # Get joystick values
            js_data = getJS()
            dpad_x, dpad_y = js_data['axis1'], js_data['axis2']
            x_button = js_data['x']  # Check if X button is pressed

            # Edge-detect X button (pressed now but not in previous loop)
            if x_button and not prev_x_button:
                recording = not recording  # Toggle state

                if recording:
                    print("Recording Started...")
                    GPIO.output(LED_PIN, GPIO.HIGH)

                    # Setup video writer
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(video_folder, f"{timestamp}.mp4")
                    frame = getImg(display=False)
                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                else:
                    print("Recording Stopped.")
                    saveLog()
                    GPIO.output(LED_PIN, GPIO.LOW)
                    if video_writer:
                        video_writer.release()
                        video_writer = None

            # Update previous button state
            prev_x_button = x_button

            # Motor control using D-pad
            if dpad_y > 0.5:
                motor.set_motors(0, 1, 0, 1, 60)  # Forward
            elif dpad_y < -0.5:
                motor.set_motors(1, 0, 1, 0, 60)  # Backward
            elif dpad_x < -0.5:
                motor.set_motors(1, 0, 0, 1, 50)  # Left
            elif dpad_x > 0.5:
                motor.set_motors(0, 1, 1, 0, 50)  # Right
            else:
                motor.stop()

            # Capture image and write video when recording
            if recording:
                img = getImg(display=True)  # Show camera feed
                saveData(img, dpad_x)  # Save image with steering input
                if video_writer:
                    video_writer.write(img)

            # Stop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping Data Collection...")

    finally:
        if video_writer:
            video_writer.release()
        motor.cleanup()
        GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED is off before exiting
        GPIO.cleanup()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()



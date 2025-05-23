import cv2
import pygame
from time import sleep
from JoyStickModule import getJS
from WebcamModule import getImg
from DataCollectionModule import saveData, saveLog
from MotorModule import Motor  # Assuming Motor class is in MotorModule.py

# Initialize Motor
motor = Motor(12, 1, 7, 13, 6, 5)
recording = False  # Start with recording OFF

def main():
    global recording
    print("Starting Data Collection...")
    
    try:
        while True:

            js_data = getJS()

            # Read D-pad inputs
            dpad_up = js_data['dpad_up']
            dpad_down = js_data['dpad_down']
            dpad_left = js_data['dpad_left']
            dpad_right = js_data['dpad_right']

            # Motor control using D-pad buttons
            if dpad_up:
                motor.set_motors(1, 0, 1, 0, 60)  # Forward
            elif dpad_down:
                motor.set_motors(0, 1, 0, 1, 60)  # Reverse
            elif dpad_left:
                motor.set_motors(1, 0, 0, 1, 50)  # Left
            elif dpad_right:
                motor.set_motors(0, 1, 1, 0, 50)  # Right
            else:
                motor.stop()

            # Toggle recording with "X" button
            if js_data['x'] == 1:
                recording = not recording
                print(f"Recording {'Started' if recording else 'Stopped'}")
                sleep(0.3)  # Debounce to prevent multiple toggles

            # Capture and save data if recording
            img = getImg(display=True)  
            if recording:
                saveData(img, dpad_right - dpad_left)  # Save D-pad steering

            # Exit if "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping Data Collection...")

    finally:
        motor.cleanup()
        saveLog()  # Save log data
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()

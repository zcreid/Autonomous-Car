import cv2
import pygame
from time import sleep
from JoyStickModule import getJS
from WebcamModule import getImg
from DataCollectionModule import saveData, saveLog
from MotorModule import Motor  # Assuming you saved the motor class as MotorModule.py

# Initialize Motor
motor = Motor(12, 1, 7, 13, 6, 5)

def main():
    print("Starting Data Collection...")
    recording = False  # Flag to track recording state
    
    try:
        while True:
            # Get joystick values
            js_data = getJS()
            dpad_x, dpad_y = js_data['axis1'], js_data['axis2']
            x_button = js_data['x']  # X button state
            
            # Toggle recording on X button press
            if x_button:  
                recording = not recording
                print(f"Recording {'started' if recording else 'stopped'}")
                sleep(0.3)  # Small delay to prevent rapid toggling
            
            # Control motors (add deadzone for better response)
            deadzone = 0.2
            if abs(dpad_y) > deadzone:
                if dpad_y > 0:
                    motor.set_motors(1, 0, 1, 0, 60)  # Move forward
                else:
                    motor.set_motors(0, 1, 0, 1, 60)  # Move backward
            elif abs(dpad_x) > deadzone:
                if dpad_x < 0:
                    motor.set_motors(0, 1, 1, 0, 50)  # Turn left
                else:
                    motor.set_motors(1, 0, 0, 1, 50)  # Turn right
            else:
                motor.stop()

            # Capture and save data only if recording is enabled
            if recording:
                img = getImg(display=True)  # Show camera feed
                saveData(img, dpad_x)  # Save image with steering input

            # Stop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping Data Collection...")

    finally:
        motor.cleanup()
        saveLog()  # Save all logged data
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()


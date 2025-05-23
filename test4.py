import cv2
import pygame
import RPi.GPIO as GPIO
from time import sleep
from JoyStickModule import getJS
from WebcamModule import getImg
from DataCollectionModule import saveData, saveLog
from MotorModule import Motor  # Assuming Motor class is in MotorModule.py

# GPIO Setup
LED_PIN = 17  # LED connected to GPIO 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED is off initially

# Initialize Motor
motor = Motor(12, 1, 7, 13, 6, 5)

def main():
    recording = False  # Track recording state
    print("Waiting to Start Recording...")

    try:
        while True:
            # Get joystick values
            js_data = getJS()
            dpad_x, dpad_y = js_data['axis1'], js_data['axis2']
            x_button = js_data['x']  # Check if X button is pressed

            # Toggle recording state when X is pressed
            if x_button and not recording:
                recording = True
                print("Recording Started...")
                GPIO.output(LED_PIN, GPIO.HIGH)  # Turn LED on

            elif not x_button and recording:
                recording = False
                print("Recording Stopped.")
                saveLog()  # Save log when stopping
                GPIO.output(LED_PIN, GPIO.LOW)  # Turn LED off

            # Motor control using D-pad
            if dpad_y > 0.5:
                motor.set_motors(1, 0, 1, 0, 60)  # Forward
            elif dpad_y < -0.5:
                motor.set_motors(0, 1, 0, 1, 60)  # Backward
            elif dpad_x < -0.5:
                motor.set_motors(0, 1, 1, 0, 50)  # Left
            elif dpad_x > 0.5:
                motor.set_motors(1, 0, 0, 1, 50)  # Right
            else:
                motor.stop()

            # Capture image only when recording
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
        GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED is off before exiting
        GPIO.cleanup()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()

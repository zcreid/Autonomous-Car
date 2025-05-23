import RPi.GPIO as GPIO
from time import sleep
import pygame
from JoyStickModule import getJS

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    def __init__(self, EnaA, In1A, In2A, EnaB, In1B, In2B, freq=1000):
        self.In1A, self.In2A, self.EnaA = In1A, In2A, EnaA
        self.In1B, self.In2B, self.EnaB = In1B, In2B, EnaB

        GPIO.setup([self.EnaA, self.In1A, self.In2A, self.EnaB, self.In1B, self.In2B], GPIO.OUT)
        
        self.pwmA = GPIO.PWM(self.EnaA, freq)
        self.pwmB = GPIO.PWM(self.EnaB, freq)
        self.pwmA.start(0)
        self.pwmB.start(0)

    def set_motors(self, m1_f, m1_b, m2_f, m2_b, speed=50):
        GPIO.output(self.In1A, m1_f)
        GPIO.output(self.In2A, m1_b)
        GPIO.output(self.In1B, m2_f)
        GPIO.output(self.In2B, m2_b)
        self.pwmA.ChangeDutyCycle(speed)
        self.pwmB.ChangeDutyCycle(speed)

    def stop(self):
        self.set_motors(0, 0, 0, 0, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()

def main():
    # Initialize pygame for joystick handling
    pygame.init()
    motor = Motor(12, 1, 7, 13, 6, 5)
    try:
        while True:
            js_data = getJS()
            # Check for joystick input and make sure it's returning expected data
            print(f"Joystick Data: {js_data}")  # Debugging output
            
            # Getting joystick values
            dpad_x, dpad_y = js_data.get('axis1', 0), js_data.get('axis2', 0)
            print(f"Joystick X: {dpad_x}, Y: {dpad_y}")  # Debugging output

            # Control logic based on joystick values
            if dpad_y > 0.5:  # Move forward
                motor.set_motors(1, 0, 1, 0, 60)
            elif dpad_y < -0.5:  # Move backward
                motor.set_motors(0, 1, 0, 1, 60)
            elif dpad_x < -0.5:  # Turn left
                motor.set_motors(0, 1, 1, 0, 50)
            elif dpad_x > 0.5:  # Turn right
                motor.set_motors(1, 0, 0, 1, 50)
            else:  # Stop if joystick is centered
                motor.stop()

            sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        motor.cleanup()
        pygame.quit()

if __name__ == "__main__":
    main()


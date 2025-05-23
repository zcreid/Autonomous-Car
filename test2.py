import RPi.GPIO as GPIO
from time import sleep
import pygame  # Library for joystick input

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor 1 (Diagonal Pair 1)
In1A, In2A, EnaA = 1, 7, 12

# Motor 2 (Diagonal Pair 2)
In3A, In4A, EnaB = 6, 5, 13

# Setup Motor GPIOs
for pin in [In1A, In2A, EnaA, In3A, In4A, EnaB]:
    GPIO.setup(pin, GPIO.OUT)

# PWM Setup
pwmA = GPIO.PWM(EnaA, 1000)
pwmB = GPIO.PWM(EnaB, 1000)
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)

# Initialize Joystick
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

def set_motors(m1_forward, m1_backward, m2_forward, m2_backward, speed=50):
    """ Controls both motors based on direction flags. """
    print(f"Motor 1 -> Forward: {m1_forward}, Backward: {m1_backward}")
    print(f"Motor 2 -> Forward: {m2_forward}, Backward: {m2_backward}")
    print(f"Speed: {speed}%\n")

    GPIO.output(In1A, m1_forward)
    GPIO.output(In2A, m1_backward)
    GPIO.output(In3A, m2_forward)
    GPIO.output(In4A, m2_backward)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

try:
    while True:
        pygame.event.pump()
        
        dpad_x, dpad_y = joystick.get_hat(0)  # Read D-Pad values

        print(f"D-Pad X: {dpad_x}, D-Pad Y: {dpad_y}")  # Debug output

        if dpad_y == 1:  # Up on D-Pad → Forward
            print("Moving Forward")
            set_motors(1, 0, 1, 0, 60)  # 60% speed
        
        elif dpad_y == -1:  # Down on D-Pad → Backward
            print("Moving Backward")
            set_motors(0, 1, 0, 1, 60)
        
        elif dpad_x == -1:  # Left on D-Pad → Turn Left
            print("Turning Left")
            set_motors(0, 1, 1, 0, 50)  # Reduce speed slightly for turns
        
        elif dpad_x == 1:  # Right on D-Pad → Turn Right
            print("Turning Right")
            set_motors(1, 0, 0, 1, 50)
        
        else:  # No input → Stop
            print("Stopping")
            set_motors(0, 0, 0, 0, 0)

        sleep(0.1)  # Small delay to prevent excessive CPU usage

finally:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    pygame.quit()


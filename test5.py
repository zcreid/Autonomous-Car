import RPi.GPIO as GPIO
import time
import pygame
from JoyStickModule import getJS
from MotorModule import Motor  # Assuming Motor class is in MotorModule.py

# GPIO Pin Configuration
TRIG = 11  # GPIO pin for Trigger
ECHO = 8   # GPIO pin for Echo
GPIO.setmode(GPIO.BCM)

# Initialize Motor
motor = Motor(12, 1, 7, 13, 6, 5)

# Setup GPIO
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    """Measures and returns the distance from the ultrasonic sensor."""
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time, stop_time = None, None

    timeout = time.time() + 0.1  # Avoid infinite loop
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
        if time.time() > timeout:
            return None  # Return None if no pulse detected

    timeout = time.time() + 0.1
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
        if time.time() > timeout:
            return None  # Return None if no pulse detected

    if start_time is None or stop_time is None:
        return None  # Ensure valid values

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Speed of sound = 343 m/s
    return distance

def main():
    """Main function for joystick-controlled motors with ultrasonic obstacle detection."""
    try:
        while True:
            distance = get_distance()
            if distance is None:
                print("Ultrasonic sensor error: No reading")
                continue  # Skip this loop iteration

            print(f"Distance: {distance:.2f} cm")  # Debug output

            if distance <= 15:
                motor.stop()
                print("🚨 Obstacle detected! Motor stopped.")
            else:
                js_data = getJS()
                dpad_x, dpad_y = js_data['axis1'], js_data['axis2']
                print(f"D-pad X: {dpad_x}, D-pad Y: {dpad_y}")  # Debug output

                if dpad_y > 0.5:
                    print("➡ Moving Forward")
                    motor.set_motors(1, 0, 1, 0, 60)
                elif dpad_y < -0.5:
                    print("⬅ Moving Backward")
                    motor.set_motors(0, 1, 0, 1, 60)
                elif dpad_x < -0.5:
                    print("↺ Turning Left")
                    motor.set_motors(0, 1, 1, 0, 50)
                elif dpad_x > 0.5:
                    print("↻ Turning Right")
                    motor.set_motors(1, 0, 0, 1, 50)
                else:
                    print("🛑 Stopping motors")
                    motor.stop()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("🛑 Stopping program...")

    finally:
        motor.cleanup()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

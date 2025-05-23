import RPi.GPIO as GPIO
import time

# Pin Definitions
TRIG = 11
ECHO = 8

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    # Send 10µs pulse to trigger
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    stop = time.time()

    # Wait for echo to go high
    while GPIO.input(ECHO) == 0:
        start = time.time()

    # Wait for echo to go low
    while GPIO.input(ECHO) == 1:
        stop = time.time()

    # Time difference
    time_elapsed = stop - start

    # Distance in cm (speed of sound = 34300 cm/s)
    distance = (time_elapsed * 34300) / 2

    return distance

try:
    while True:
        dist = get_distance()
        print(f"Distance: {dist:.2f} cm")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    GPIO.cleanup()

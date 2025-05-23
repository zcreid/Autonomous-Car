import RPi.GPIO as GPIO
import time

# Front Sensor Pins (adjust as needed)
FRONT_TRIG = 23
FRONT_ECHO = 24

# Rear Sensor Pins (adjust as needed)
REAR_TRIG = 17
REAR_ECHO = 27

def setup_ultrasonic():
    GPIO.setmode(GPIO.BCM)
    # Front Sensor Setup
    GPIO.setup(FRONT_TRIG, GPIO.OUT)
    GPIO.setup(FRONT_ECHO, GPIO.IN)
    # Rear Sensor Setup
    GPIO.setup(REAR_TRIG, GPIO.OUT)
    GPIO.setup(REAR_ECHO, GPIO.IN)

def get_distance(trig_pin, echo_pin):
    """Generic function to read distance from any sensor"""
    GPIO.output(trig_pin, False)
    time.sleep(0.0001)
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance_mm = (pulse_duration * 34300) / 2  # Convert to mm
    return distance_mm

def get_front_distance():
    return get_distance(FRONT_TRIG, FRONT_ECHO)

def get_rear_distance():
    return get_distance(REAR_TRIG, REAR_ECHO)

def cleanup_ultrasonic():
    GPIO.cleanup()
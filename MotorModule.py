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
        
    def move(self,speed=0.5,turn=0,t=0):
        speed *=100
        turn *=70
        leftSpeed = speed-turn
        rightSpeed = speed+turn

        if leftSpeed>100: leftSpeed =100
        elif leftSpeed<-100: leftSpeed = -100
        if rightSpeed>100: rightSpeed =100
        elif rightSpeed<-100: rightSpeed = -100
        #print(leftSpeed,rightSpeed)
        self.pwmA.ChangeDutyCycle(abs(leftSpeed))
        self.pwmB.ChangeDutyCycle(abs(rightSpeed))
        if leftSpeed>0:GPIO.output(self.In1A,GPIO.HIGH);GPIO.output(self.In2A,GPIO.LOW)
        else:GPIO.output(self.In1A,GPIO.LOW);GPIO.output(self.In2A,GPIO.HIGH)
        if rightSpeed>0:GPIO.output(self.In1B,GPIO.HIGH);GPIO.output(self.In2B,GPIO.LOW)
        else:GPIO.output(self.In1B,GPIO.LOW);GPIO.output(self.In2B,GPIO.HIGH)
        sleep(t)

    def stop(self):
        self.set_motors(0, 0, 0, 0, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()


def main():
    motor = Motor(12, 1, 7, 13, 6, 5)
    try:
        while True:
            js_data = getJS()
            dpad_x, dpad_y = js_data['axis1'], js_data['axis2']

            if dpad_y > 0.5:
                motor.set_motors(1, 0, 1, 0, 60)
            elif dpad_y < -0.5:
                motor.set_motors(0, 1, 0, 1, 60)
            elif dpad_x < -0.5:
                motor.set_motors(0, 1, 1, 0, 50)
            elif dpad_x > 0.5:
                motor.set_motors(1, 0, 0, 1, 50)
            else:
                motor.stop()

            sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        motor.cleanup()
        pygame.quit()

if __name__ == "__main__":
    main()


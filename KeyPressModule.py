import pygame
from MotorModule import Motor


def init():
    pygame.init()
    win = pygame.display.set_mode((100,100))

def getKey(keyName):
    ans = False
    for eve in pygame.event.get():pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame,'K_{}'.format(keyName))
    if keyInput [myKey]:
        ans = True
    pygame.display.update()

    return ans
def main():
    if getKey('UP'):
        motor.move(0.5, 0, 0.1)
    elif getKey('DOWN'):
        motor.move(-0.5, 0, 0.1)
    elif getKey('LEFT'):
        motor.move(0, -0.9,0.1)
    elif getKey('RIGHT'):
        motor.move(0, 0.9,0.1)
    else:
        motor.stop(0.1)
if __name__ == '__main__':
    init()
    motor = Motor(17, 27, 22, 10, 11, 9)
    while True:
        main()
import pygame
from time import sleep

pygame.init()
controller = pygame.joystick.Joystick(0)
controller.init()

buttons = {
    'x': 0, 'o': 0, 't': 0, 's': 0,
    'L1': 0, 'R1': 0, 'L2': 0, 'R2': 0,
    'share': 0, 'options': 0,
    'axis1': 0., 'axis2': 0., 'axis3': 0., 'axis4': 0.,
    'dpad_up': 0, 'dpad_down': 0, 'dpad_left': 0, 'dpad_right': 0  # D-pad buttons
}

axiss = [0., 0., 0., 0., 0., 0.]

def getJS(name=''):
    global buttons
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axiss[event.axis] = round(event.value, 2)
        elif event.type == pygame.JOYBUTTONDOWN:
            for x, (key, val) in enumerate(buttons.items()):
                if x < 10:
                    if controller.get_button(x): buttons[key] = 1
        elif event.type == pygame.JOYBUTTONUP:
            for x, (key, val) in enumerate(buttons.items()):
                if x < 10:
                    if event.button == x: buttons[key] = 0
        elif event.type == pygame.JOYHATMOTION:  # Detect D-pad movement
            hat_x, hat_y = controller.get_hat(0)
            buttons['dpad_up'] = 1 if hat_y == 1 else 0
            buttons['dpad_down'] = 1 if hat_y == -1 else 0
            buttons['dpad_left'] = 1 if hat_x == -1 else 0
            buttons['dpad_right'] = 1 if hat_x == 1 else 0

    buttons['axis1'], buttons['axis2'], buttons['axis3'], buttons['axis4'] = [axiss[0], axiss[1], axiss[3], axiss[4]]
    
    if name == '':
        return buttons
    else:
        return buttons[name]

if __name__ == '__main__':
    while True:
        print(getJS())  # Test output
        sleep(0.05)


from pynput.keyboard import Key, Controller
import time
class ActionExecutor:
    def __init__(self):
        self.keyboard = Controller()

    def restart_level(self):
        # restart the level by pressing the spacebar 
        time.sleep(3) # wait 3 seconds to restart the level
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

    def act(self, action):
        if action == 1:
            self.keyboard.press(Key.space)
        elif action == 0:
            self.keyboard.release(Key.space)


   
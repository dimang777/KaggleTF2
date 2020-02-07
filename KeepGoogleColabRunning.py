# Google colab keep it open

import pynput.mouse as mouse
from pynput.mouse import Button
import time

while True:
    mouse.click(Button.left, 1)
    time.sleep(30)
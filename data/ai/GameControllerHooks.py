# GameControllerHooks.py
# Intercepts and sends keystrokes to the game window for Sarah's AI learning and control

import pyautogui
import time

ACTION_MAP = {
    'up': 'up',
    'down': 'down',
    'left': 'left',
    'right': 'right',
    'space': 'space'
}


def press_key(action):
    if action in ACTION_MAP:
        pyautogui.press(ACTION_MAP[action])
        print(f"[GameController] Executed: {action}")
    else:
        print(f"[GameController] Invalid action: {action}")


def tap_sequence(actions, delay=0.2):
    for action in actions:
        press_key(action)
        time.sleep(delay)


if __name__ == '__main__':
    print("[GameController] Testing sequence...")
    tap_sequence(['up', 'up', 'left', 'right', 'space'])

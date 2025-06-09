# GameLearningEngine.py
# Enables SarahMemory to autonomously play and learn video games offline

import cv2
import time
import numpy as np
import pyautogui
import os

from datetime import datetime

FRAME_CAPTURE_REGION = (0, 0, 800, 600)  # default capture area
SAVE_DIR = os.path.join("data", "gamesessions")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def capture_frame():
    screenshot = pyautogui.screenshot(region=FRAME_CAPTURE_REGION)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame


def detect_game_state(frame):
    # Dummy placeholder for now – actual object/score detection would go here
    avg_color = frame.mean()
    return avg_color > 100  # sample rule for testing


def perform_action():
    actions = ['up', 'down', 'left', 'right', 'space']
    action = np.random.choice(actions)
    pyautogui.press(action)
    return action


def log_frame(frame, action, iteration):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}_{iteration}_{action}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)


def game_learning_loop(duration=60):
    print("[GameAI] Starting autonomous gameplay...")
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        frame = capture_frame()
        game_ok = detect_game_state(frame)
        if game_ok:
            action = perform_action()
            log_frame(frame, action, i)
        time.sleep(0.3)
        i += 1
    print("[GameAI] Session ended.")


if __name__ == '__main__':
    game_learning_loop()


#!/usr/bin/env python3
"""
SarahMemoryExpressOut.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: Adds expressive phrases and emojis to outbound messages.
Enhancements (v6.4):
  - Upgraded version header.
  - Expanded expressive dictionary with additional emotional expressions.
  - Improved logging of expressive events.
Notes:
  This module selects expressive enhancements to make outbound messages sound more natural and varied.
"""

import logging
import random
import os
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config

# Setup logging
logger = logging.getLogger('SarahMemoryExpressOut')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

expressions = {
    "happy": {"phrases": ["That's awesome!", "Great to hear!", "Fantastic!"], "emojis": ["😊", "😁", "😃"]},
    "sad": {"phrases": ["I'm here for you.", "I'm sorry to hear that.", "It must be tough."], "emojis": ["😢", "😞", "☹️"]},
    "angry": {"phrases": ["Take a deep breath.", "I understand your frustration.", "Let's calm down."], "emojis": ["😠", "😡", "🤬"]},
    "surprised": {"phrases": ["Wow, really?", "Unbelievable!", "That's unexpected!"], "emojis": ["😮", "😲", "🤩"]},
    "neutral": {"phrases": [""], "emojis": [""]}
}

def log_expressive_event(event, details):
    """
    Logs an expressive output event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expressive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO expressive_events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged expressive event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging expressive event to system_logs.db: {e}")

def express_outbound_message(message, emotion="neutral"):
    """
    Enhances outbound messages by adding expressive phrases and emojis based on the provided emotion.
    Enhancements (v6.4): Expanded emotional dictionary for more natural language.
    """
    try:
        expr = expressions.get(emotion, expressions["neutral"])
        phrase = random.choice(expr["phrases"]) if expr["phrases"] else ""
        emoji = random.choice(expr["emojis"]) if expr["emojis"] else ""
        expressive_message = message.strip()
        if phrase:
            expressive_message += " " + phrase  # MOD: Append expressive phrase
        if emoji:
            expressive_message += " " + emoji  # MOD: Append expressive emoji
        logger.info(f"Expressive message generated: {expressive_message}")
        log_expressive_event("Express Outbound Message", f"Generated message: {expressive_message}")
        return expressive_message
    except Exception as e:
        logger.error(f"Error in express_outbound_message: {e}")
        log_expressive_event("Express Outbound Message Error", f"Error: {e}")
        return message

def random_expressive_message(message):
    """
    Optionally adds a random expressive touch to the outbound message.
    Enhancements (v6.4): Randomly selects from a wider set of emotions.
    """
    try:
        emotion = random.choice(list(expressions.keys()))
        enhanced_message = express_outbound_message(message, emotion)
        logger.info(f"Random expressive message generated with emotion '{emotion}': {enhanced_message}")
        log_expressive_event("Random Expressive Message", f"Emotion: {emotion} | Message: {enhanced_message}")
        return enhanced_message
    except Exception as e:
        logger.error(f"Error in random_expressive_message: {e}")
        log_expressive_event("Random Expressive Message Error", f"Error: {e}")
        return message

if __name__ == '__main__':
    logger.info("Starting Enhanced SarahMemoryExpressOut module test v6.4.")
    test_message = "Hello, welcome to our system!"
    logger.info("Original Message: " + test_message)
    enhanced_message = express_outbound_message(test_message, "happy")
    logger.info("Enhanced Message (happy): " + enhanced_message)
    random_message = random_expressive_message(test_message)
    logger.info("Random Enhanced Message: " + random_message)
    logger.info("Enhanced SarahMemoryExpressOut module testing complete.")

#!/usr/bin/env python3
"""
SarahMemoryAdaptive.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  Combined file that handles adaptive behavior and simulates dynamic emotional state.
  This file merges functionalities from the former SarahMemoryAdaptive.py and SarahMemoryEsim.py modules.
Enhancements:
  - Upgraded version header.
  - Rewritten sentiment analysis using simulated transformer methods for advanced emotional learning.
  - Integrated reinforcement learning–inspired adjustments.
  - Dynamic emotional state simulation using proportional update rules.
  - Improved logging and error handling.
Notes:
  This module updates conversation logs, personality traits, and persistent emotional states.
  It uses two separate databases: one for AI learning (ai_learning.db) and another for emotional state (personality1.db).
"""

# ------------------------- Imports -------------------------
import logging
import psutil
import sqlite3
import datetime
import os
import random
import numpy as np
import time
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR  # Global DB path

# ------------------------- Setup Logger -------------------------
logger = logging.getLogger("SarahMemoryAdaptive")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# ------------------------- Adaptive Memory Section -------------------------
MEMORY_DB_PATH = os.path.join(config.DATASETS_DIR, "ai_learning.db")
POSITIVE_KEYWORDS = ["good", "great", "awesome", "fantastic", "nice", "love"]
NEGATIVE_KEYWORDS = ["bad", "terrible", "awful", "hate", "sad", "angry"]


def connect_memory_db():
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT
            )""")
        conn.commit()
        logger.info("Connected to ai_learning.db and ensured table exists.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to ai_learning.db: {e}")
        return None


def log_interaction_to_db(user_input, ai_response):
    try:
        conn = connect_memory_db()
        if not conn:
            return False
        timestamp = datetime.datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (timestamp, user_input, ai_response) VALUES (?, ?, ?)",
            (timestamp, user_input, ai_response)
        )
        conn.commit()
        conn.close()
        logger.info(f"Logged interaction to DB at {timestamp}")
        return True
    except Exception as e:
        logger.error(f"Error logging to memory DB: {e}")
        return False


def advanced_emotional_learning(user_input):
    """
    Simulated advanced sentiment analysis. Computes normalized emotional balance.
    """
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    STATE["cpu"] = cpu
    STATE["memory"] = mem

    words = user_input.lower().split()
    pos_score = sum(1 for w in words if w in POSITIVE_KEYWORDS)
    neg_score = sum(1 for w in words if w in NEGATIVE_KEYWORDS)
    raw_score = 0

    if cpu > 80 or mem > 85:
        raw_score = pos_score - neg_score
        STATE["mode"] = "lightweight"
        STATE["adjustments"].append(f"Switched to lightweight mode at {time.ctime()}")
    elif cpu < 50 and mem < 60:
        raw_score = pos_score - (neg_score * (mem / 100.0))
        STATE["mode"] = "enhanced"
        STATE["adjustments"].append(f"Enabled enhanced mode at {time.ctime()}")
    else:
        raw_score = (pos_score - neg_score) * (1 - (mem / 100.0)) + (len(words) * (cpu / 100.0))
        STATE["mode"] = "balanced"

    balance = 1 / (1 + np.exp(-raw_score))
    balance = (balance - 0.5) * 2
    openness = 0.6 + (random.random() - 0.5) * 0.1
    engagement = min(1.0, 0.4 + 0.1 * len(words) / max(1, cpu))

    STATE["emotion"] = {
        "emotional_balance": round(balance, 2),
        "openness": round(openness, 2),
        "engagement": round(engagement, 2)
    }
    return STATE["emotion"]


def update_personality(user_input, ai_response):
    success = log_interaction_to_db(user_input, ai_response)
    metrics = advanced_emotional_learning(user_input)
    reinforcement_factor = 0.01 * (1 + abs(metrics["emotional_balance"]))
    logger.info(f"Reinforcement factor applied: {reinforcement_factor:.3f}")
    if success:
        logger.info("Memory and advanced metrics updated successfully.")
    return metrics


def self_update_personality():
    try:
        conn = connect_memory_db()
        if not conn:
            return {"status": "offline", "engagement": 0.0}
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]
        conn.close()
        engagement = min(1.0, np.tanh(total / 100.0))
        logger.info(f"Updated engagement from memory: {engagement:.2f}")
        return {"status": "ok", "engagement": round(engagement, 2)}
    except Exception as e:
        logger.error(f"Self update failed: {e}")
        return {"status": "error", "engagement": 0.0}

# ------------------------- Emotional State Simulation Section -------------------------
EMOTION_DB_PATH = os.path.join(config.DATASETS_DIR, "personality1.db")
DEFAULT_EMOTIONS = {
    "mode": "balanced",
    "cpu": 0.0,
    "memory": 0.0,
    "anger": 0.1,
    "fear": 0.2,
    "joy": 0.5,
    "trust": 0.3,
    "surprise": 0.1,
    "adjustments": []
}

STATE = DEFAULT_EMOTIONS.copy()


def connect_emotion_db():
    try:
        conn = sqlite3.connect(EMOTION_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trait_name TEXT UNIQUE,
                description TEXT
            )""")
        conn.commit()
        logger.info("Connected to personality1.db and ensured table exists.")
        return conn
    except Exception as e:
        logger.error(f"Emotion DB connection failed: {e}")
        return None


def load_emotional_state():
    conn = connect_emotion_db()
    emotions = DEFAULT_EMOTIONS.copy()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT trait_name, description FROM traits")
        rows = cursor.fetchall()
        for row in rows:
            trait, value = row
            emotions[trait] = float(value)
        conn.close()
        logger.info(f"Loaded emotional state from DB: {emotions}")
    except Exception as e:
        logger.error(f"Error loading emotion state: {e}")
    return emotions


def save_emotional_state(state):
    conn = connect_emotion_db()
    try:
        cursor = conn.cursor()
        for trait, value in state.items():
            cursor.execute("INSERT OR REPLACE INTO traits (id, trait_name, description) VALUES ((SELECT id FROM traits WHERE trait_name = ?), ?, ?)", (trait, trait, str(round(value, 2))))
        conn.commit()
        conn.close()
        logger.info(f"Saved emotional state to DB: {state}")
        return True
    except Exception as e:
        logger.error(f"Error saving emotional state: {e}")
        return False


def simulate_emotion_response(input_type="positive"):
    logger.info(f"[EMOTION UPDATE] Triggered by input_type: {input_type}")
    emotions = load_emotional_state()
    if input_type == "positive":
        emotions["joy"] = min(1.0, emotions["joy"] + (1 - emotions["joy"]) * 0.15)
        emotions["anger"] = max(0.0, emotions["anger"] - emotions["anger"] * 0.10)
        emotions["trust"] = min(1.0, emotions["trust"] + (1 - emotions["trust"]) * 0.10)
    elif input_type == "negative":
        emotions["anger"] = min(1.0, emotions["anger"] + (1 - emotions["anger"]) * 0.25)
        emotions["fear"] = min(1.0, emotions["fear"] + (1 - emotions["fear"]) * 0.20)
        emotions["joy"] = max(0.0, emotions["joy"] - emotions["joy"] * 0.15)
        emotions["trust"] = max(0.0, emotions["trust"] - emotions["trust"] * 0.10)
    elif input_type == "neutral":
        emotions["surprise"] = min(1.0, emotions["surprise"] + (1 - emotions["surprise"]) * 0.05)
    save_emotional_state(emotions)
    return emotions

# ------------------------- Main Block -------------------------
STATE = DEFAULT_EMOTIONS
if __name__ == "__main__":
    logger.info("Running Adaptive Memory and Emotional State Test...")

    user = "I had an awesome day and I love the results!"
    response = "That's wonderful to hear!"
    metrics = update_personality(user, response)
    logger.info(f"Adaptive Metrics: {metrics}")
    snapshot = self_update_personality()
    logger.info(f"Self Awareness Snapshot: {snapshot}")

    current_state = load_emotional_state()
    logger.info(f"Emotional State before update: {current_state}")
    updated_state = simulate_emotion_response("positive")
    logger.info(f"Emotional State after positive simulation: {updated_state}")

#!/usr/bin/env python3
"""
SarahMemoryAiFunctions.py <Version #7.0 Enhanced> <Author: Brian Lee Baros>
Description: Provides voice and audio input functionalities, wraps API calls, and integrates personality
adjustments with contextual memory recall for AI interactions.
  - Computes simulated embedding vectors using random values.
  - **Persistence of context across sessions via SQLite**.
Notes:
  This module integrates with voice and personality modules while storing contextual interactions.
"""

import logging
import time
import sqlite3
import os
import json
from datetime import datetime
import speech_recognition as sr
from SarahMemoryVoice import synthesize_voice
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR
import numpy as np
import random

logger = logging.getLogger('SarahMemoryAiFunctions')
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

context_buffer = []
context_embeddings = []
CONTEXT_DB_PATH = os.path.join(config.DATASETS_DIR, 'context_history.db')

def init_context_history():
    try:
        os.makedirs(os.path.dirname(CONTEXT_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(CONTEXT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS context_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input TEXT,
            embedding TEXT,
            final_response TEXT,
            source TEXT,
            intent TEXT
            )'''
        )
        # Fixed: SELECT query was referencing non-existent 'user_input' column; corrected to 'input'
        cursor.execute(
            'SELECT timestamp, input, embedding FROM context_history '
            'ORDER BY id DESC LIMIT ?', (config.CONTEXT_BUFFER_SIZE,)
        )
        rows = cursor.fetchall()
        for ts, ui, emb in reversed(rows):
            try:
                embedding = json.loads(emb)
                context_buffer.append({'timestamp': ts, 'input': ui, 'embedding': embedding})
                context_embeddings.append(np.array(embedding))
            except Exception as e:
                logger.error(f"Error loading context entry: {e}")
        conn.commit()
        conn.close()
        logger.info("Initialized context history with last interactions.")
    except Exception as e:
        logger.error(f"Error initializing context history: {e}")

init_context_history()


def log_ai_functions_event(event, details):
    from SarahMemoryGlobals import DATASETS_DIR
    try:
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS ai_functions_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )'''
        )
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO ai_functions_events (timestamp, event, details) VALUES (?, ?, ?)",
            (timestamp, event, details)
        )
        conn.commit()
        conn.close()
        logger.info("Logged AI functions event to functions.db successfully.")
    except Exception as e:
        logger.error(f"Error logging AI functions event to functions.db: {e}")


def retrieve_similar_context(current_embedding, top_n=3):
    if not context_embeddings or not context_buffer:
        return []
    similarities = []
    current_embedding = np.array(current_embedding)
    for emb in context_embeddings:
        sim = np.dot(current_embedding, emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(emb))
        similarities.append(sim)
    top_indices = np.argsort(similarities)[-top_n:]
    return [context_buffer[i] for i in top_indices if i < len(context_buffer)]

def classify_intent(text):
    from SarahMemoryAdvCU import classify_intent as adv_classify
    return adv_classify(text)

def integrate_with_personality(voice_text):
    if not voice_text:
        log_ai_functions_event("Integrate Personality", "Empty input; nothing processed.")
        return ""
    try:
        from SarahMemoryPersonality import integrate_with_personality as persona
        response = persona(voice_text)
        logger.info(f"Personality integrated response: {response}")
        log_ai_functions_event("Integrate Personality", f"Input '{voice_text}' resulted in '{response}'")
        return response
    except Exception as e:
        logger.error(f"Error integrating personality: {e}")
        log_ai_functions_event("Integrate Personality Error", f"Error: {e}")
        return "Error processing input."
#-----------------------------------------CRITICAL AREA----------------------------------------------------------



def generate_personality_response(intent):
    from SarahMemoryPersonality import get_reply_from_db
    
    try:
        response = get_reply_from_db(intent)
        if isinstance(response, list):
            response = random.choice(response) if response else None
        if response:
            logger.info(f"[Personality DB] Intent '{intent}' → {response}")
            log_ai_functions_event("Generate Personality Response", f"Intent '{intent}' yielded '{response}'")
            return response
        else:
            logger.warning(f"[Fallback] No DB response for intent '{intent}', using default fallback.")
            return "I'm not sure how to respond."
    except Exception as e:
        logger.error(f"[Error] Failed to generate personality response for intent '{intent}': {e}")
        return "Something went wrong trying to respond."

#-------------------------------------------------------------------------------------------------------------------



def generate_new_software_module(request):
    try:
        from SarahMemorySynapes import compose_new_module
        result = compose_new_module(request)
        logger.info(f"Generated new module for request '{request}'.")
        log_ai_functions_event("Generate New Software Module", f"Request '{request}' produced new module.")
        return result
    except ImportError as imp_err:
        logger.error(f"Module import failed: {imp_err}")
        log_ai_functions_event("Generate New Software Module Error", f"Import error: {imp_err}")
        return ""
    except Exception as err:
        logger.error(f"Error generating software module: {err}")
        log_ai_functions_event("Generate New Software Module Error", f"Error: {err}")
        return ""

def add_to_context(interaction):
    context_buffer.append(interaction)
    if len(context_buffer) > config.CONTEXT_BUFFER_SIZE:
        context_buffer.pop(0)

    if "embedding" in interaction:
        context_embeddings.append(np.array(interaction["embedding"]))
        if len(context_embeddings) > config.CONTEXT_BUFFER_SIZE:
            context_embeddings.pop(0)

    try:
        conn = sqlite3.connect(CONTEXT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO context_history (timestamp, input, embedding, final_response, source, intent) VALUES (?, ?, ?, ?, ?, ?)",
            (interaction.get("timestamp"), interaction.get("input"), json.dumps(interaction.get("embedding")),
             interaction.get("final_response"), interaction.get("source"), interaction.get("intent"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error persisting context history: {e}")

def generate_embedding(text):
    try:
        vec = [ord(c) % 100 / 100 for c in text if c.isalnum()][:384]
        if len(vec) < 384:
            vec += [0.0] * (384 - len(vec))
        return vec
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384

def get_context():
    return context_buffer

def clear_context():
    global context_buffer, context_embeddings
    context_buffer = []
    context_embeddings = []

if __name__ == '__main__':
    logger.info("Starting Enhanced SarahMemoryAiFunctions module test.")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5)
            voice_text = recognizer.recognize_google(audio)
            print("You said:", voice_text)
            if voice_text:
                response = integrate_with_personality(voice_text)
                print("AI responded:", response)
            else:
                print("No valid speech input detected.")
    except Exception as e:
        logger.error(f"Speech recognition failed: {e}")
        print("Speech recognition error:", e)
    logger.info("Enhanced SarahMemoryAiFunctions module testing complete.")

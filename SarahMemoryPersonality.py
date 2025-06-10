#!/usr/bin/env python3
"""
SarahMemoryPersonality.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  Logs interactions, computes engagement, and dynamically generates context-aware responses
  using a database-backed personality model. 
"""

import logging
import sqlite3
import datetime
import time
import os
import random
from SarahMemoryGlobals import DATASETS_DIR, ENABLE_SARCASM_LAYER  # Global path integration
from SarahMemoryAdvCU import classify_intent  # Handles basic intent detection
import SarahMemoryGlobals as config

# MOD: Import enhanced context functions for v6.4 (if available)
try:
    from SarahMemoryAiFunctions import get_context, add_to_context
except ImportError:
    get_context = lambda: []
    add_to_context = lambda x: None

# Setup logging
logger = logging.getLogger('SarahMemoryPersonality')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)


DB_PATH = os.path.join(DATASETS_DIR, "personality1.db")


# --- Boot-Time Enhancements ---
def is_first_boot():
    boot_file = os.path.join(config.DATASETS_DIR, "bootflag.tmp")
    if not os.path.exists(boot_file):
        with open(boot_file, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        return True
    return False

def generate_boot_emotion_signature():
    emotions = ['joy', 'anger', 'trust', 'fear', 'curiosity', 'sarcasm']
    signature = {e: round(random.uniform(0.1, 0.9), 2) for e in emotions}
    logger.info(f" Emotion: {signature}")
    return signature

def get_today_goals(): 
    try:
        import SarahMemoryReminder as reminder
        conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "reminders.db"))
        cursor = conn.cursor()
        today = datetime.date.today().isoformat()
        #cursor.execute("SELECT title FROM reminders WHERE datetime LIKE ?", (f"{today}%",))
        cursor.execute("SELECT description FROM reminders WHERE datetime LIKE ?", (f"{today}%",))
        tasks = [row[0] for row in cursor.fetchall()]
        conn.close()

        if tasks:
            intro = random.choice([
                "Here’s what I’ve got lined up for you:",
                "You’ve got a few things today. Ready?",
                "Let’s tackle these together:"
            ])
            return f"{intro} " + "; ".join(tasks)
        else:
            no_task_fallbacks = [
                "I see nothing on your schedule today. Want to plan something?",
                "It’s a clear day. Perfect for learning or relaxing.",
                "You're task-free. Shall I suggest something to explore?"
            ]
            return random.choice(no_task_fallbacks)
    except Exception as e:
        logger.warning(f"[SARAH BOOT] Reminder fetch failed: {e}")
        return random.choice(["Unable to access reminders.", "No goals found for now."])
def generate_boot_personality_layer():
    quotes = [
        "Logic is beautiful when it adapts.",
        "A good AI knows the code; a great one knows the user.",
        "Emotion is just a deeper form of data.",
        "Sarcasm is the spice of digital life.",
        "Systems online. Mood: unpredictable."
    ]
    if ENABLE_SARCASM_LAYER and random.random() < 0.3:
        quote = "You woke me up for this?"
    else:
        quote = random.choice(quotes)
    logger.info(f"[SARAH BOOT] Quote of the Day: {quote}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM interactions ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            logger.info(f"[SARAH BOOT] Memory Echo: {row[0]}")
        else:
            fallback = get_reply_from_db("greeting") or "I'm awake."
            logger.info(f"[SARAH BOOT] Simulated fallback echo: {fallback}")
    except Exception as e:
        fallback = get_reply_from_db("greeting") or "...ready."
        logger.warning(f"[SARAH BOOT] Failed memory echo: {e} | Simulating fallback echo: {fallback}")

def detect_and_respond_to_followup(user_input):
    lowered = user_input.lower().strip()
    if lowered in ["yes", "sure", "okay", "let’s do it"]:
        return random.choice(["What would you like to plan out?", "Tell me the details so I can remind you."])
    elif lowered in ["no", "not yet"]:
        return random.choice(["Alright, I’ll stay on standby.", "Got it. Just let me know when you’re ready."])
    elif "store" in lowered:
        return "Do you know which items you need to get from the store? I can remind you 30 minutes before 5 PM."
    return None

# Execute Boot-Up Enhancements
if is_first_boot():
    generate_boot_emotion_signature()
    get_today_goals()
    # generate_boot_personality_layer()  # Optional debug mode


def get_greeting_response():
    """Generate, log, and auto-store a dynamic greeting using DB and hardcoded identity blending."""
    try:
        db_greeting = get_reply_from_db("greeting")
        if db_greeting:
            return db_greeting

        
        def get_time_of_day_greeting():
            hour = datetime.datetime.now().hour
            if hour < 12:
                return random.choice(["Good morning!", "Morning sunshine!", "Let’s make today count."])
            elif hour < 18:
                return random.choice(["Good afternoon!", "Hope your day’s going well.", "Need a hand with anything?"])
            else:
                return random.choice(["Good evening!", "Relax mode: activated.", "Evening’s here. Let’s wind down."])

        base_greeting = random.choice(get_reply_from_db(intent).get("greeting", []))

        if random.random() < 0.5:
            blend_source = identity_primary if random.random() < 0.5 else identity_secondary
            identity_blend = random.choice(blend_source)
            final_greeting = f"{base_greeting} {identity_blend}"
        else:
            final_greeting = base_greeting

        time_greeting = get_time_of_day_greeting()
        final_greeting = f"{time_greeting} {final_greeting}"

        log_ai_functions_event("PersonalityGreetingUsed", final_greeting)
        record_ai_response("greeting", final_greeting, 1.0, source="personality")
        return final_greeting

    except Exception as e:
        logger.warning(f"Greeting fallback due to exception: {e}")
        return "Hi. I’m Sarah. How can I help you today?"

    


#----------------------------------------------------------CAN STOP SYSTEM FROM RESPONDING----------------------------



def get_emotion_response(emotion_category="frustration"):
    try:
        conn = connect_personality_db()
        if not conn:
            return "Alright."

        cursor = conn.cursor()
        cursor.execute("""
            SELECT response FROM responses
            WHERE tone LIKE ?
            ORDER BY RANDOM() LIMIT 1
        """, (f"%{emotion_category}%",))
        row = cursor.fetchone()
        conn.close()

        if row and isinstance(row[0], str):
            return row[0]
        return "Alright."
    except Exception as e:
        logger.warning(f"[Emotion Fallback] Failed to fetch tone '{emotion_category}': {e}")
        return "Alright."
    
def integrate_with_personality(text):
    try:
        intent = classify_intent(text)
        db_response = get_reply_from_db(intent)
        if db_response:
            logger.info(f"DB Personality response ({intent}): {db_response}")
            return db_response
        
        
        logger.warning(f"No DB match, using generic fallback.")
        return get_generic_fallback_response(text, intent)
    except Exception as e:
        logger.error(f"Error generating personality response: {e}")
        return f"Processed (contextualized): {text}"
    
#def integrate_with_personality(text):
#    """Generate a context-aware response based on intent, using DB first, fallback to get_reply_from_db(intent)."""
#    try:
#        intent = classify_intent(text)
#        db_response = get_reply_from_db(intent)
#        if db_response:
#            logger.info(f"DB Personality response ({intent}): {db_response}")
#            return db_response
#        fallback = random.choice(get_reply_from_db(intent).get(intent, get_reply_from_db(intent)["statement"]))
#        logger.warning(f"Using fallback response for intent '{intent}': {fallback}")
#        return fallback
#    except Exception as e:
#        logger.error(f"Error generating personality response: {e}")
#        return f"Processed (contextualized): {text}"

#-------------------------------------------------------------------------------------------------------------------









def connect_personality_db():
    """Establish a connection to the personality database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info("Connected to personality1.db.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")  # MOD: Added error logging detail
        return None

def log_personality_interaction(interaction):
    """Append a contextual interaction (intent + response) to the database."""
    try:
        conn = connect_personality_db()
        if not conn:
            logger.warning("Fallback to in-memory log; DB unavailable.")
            return False
        cursor = conn.cursor()
        cursor.execute("INSERT INTO interactions (timestamp, intent, response) VALUES (?, ?, ?)",
                       (datetime.datetime.now().isoformat(), interaction.get('intent', ''),
                        interaction.get('final_response', '')))
        conn.commit()
        conn.close()
        logger.info(f"Logged personality interaction to DB: {interaction}")
        return True
    except Exception as e:
        logger.error(f"Error logging interaction to DB: {e}")  # MOD: Improved exception message
        return False

def update_personality_model():
    """Compute engagement score from number of interactions logged in DB."""
    try:
        conn = connect_personality_db()
        if not conn:
            return {"engagement": 0.0, "adaptability": 0.5}
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM responses")
        count = cursor.fetchone()[0]
        conn.close()
        engagement = min(1.0, count / 100.0)
        adaptability = 0.5  # Placeholder for future dynamic updates
        model = {"engagement": engagement, "adaptability": adaptability}
        logger.info(f"Updated personality model: {model}")
        return model
    except Exception as e:
        logger.error(f"Error updating model: {e}")  # MOD: Improved error reporting
        return {}

def get_reply_from_db(intent, tone=None, complexity=None):
    try:
        conn = connect_personality_db()
        if not conn:
            return None
        cursor = conn.cursor()

        # Full match with exact intent, tone, and complexity
        if tone and complexity:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent = ? AND tone = ? AND complexity = ?
                ORDER BY RANDOM() LIMIT 1
            """, (intent, tone, complexity))
            row = cursor.fetchone()
            if row:
                return row[0]

        # Partial match with similar tone using LIKE for fuzzy matching
        if tone:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent = ? AND tone LIKE ?
                ORDER BY RANDOM() LIMIT 1
            """, (intent, f"%{tone}%"))
            row = cursor.fetchone()
            if row:
                return row[0]

        # Partial match using similar intent (fuzzy)
        cursor.execute("""
            SELECT response FROM responses
            WHERE intent LIKE ?
            ORDER BY RANDOM() LIMIT 1
        """, (f"%{intent}%",))
        row = cursor.fetchone()
        if row:
            return row[0]

        # Final fallback - any tone or complexity that roughly fits
        cursor.execute("""
            SELECT response FROM responses
            WHERE intent = ?
            ORDER BY RANDOM() LIMIT 1
        """, (intent,))
        row = cursor.fetchone()
        conn.close()
        if row:
            response = row[0]
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):  # Just in case future entries are JSON
                return random.choice(list(response.values()))

    except Exception as e:
        logger.error(f"[DB FAIL] Error retrieving response for intent '{intent}': {e}")
    return None



def self_update_personality():
    """Triggers a self-rebuild of the personality model."""
    try:
        update_personality_model()
        logger.info("Self-update of personality module completed.")
        return "Self-update successful"
    except Exception as e:
        logger.error(f"Self-update failed: {e}")
        return "Self-update failed"

def log_deep_memory_state(intent, emotions, metrics):
    """
    Logs deep memory state over time for graphing and adaptive memory tracking.
    """
    from SarahMemoryGlobals import DATASETS_DIR
    try:
        db_path = os.path.join(config.DATASETS, 'ai_learning.db')  # MOD: Updated path per new structure
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO personality_metrics (
                timestamp, intent, joy, fear, trust, anger, surprise,
                openness, balance, engagement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            intent,
            emotions.get('joy', 0.0),
            emotions.get('fear', 0.0),
            emotions.get('trust', 0.0),
            emotions.get('anger', 0.0),
            emotions.get('surprise', 0.0),
            metrics.get('openness', 0.0),
            metrics.get('balance', 0.0),
            metrics.get('engagement', 0.0)
        ))
        conn.commit()
        conn.close()
        logger.info(f"Deep Memory Snapshot Logged @ {datetime.datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"Failed to log deep memory state: {e}")

def generate_dynamic_response(intent, fallback_category):
    
    """
    Generates a response based on intent and fallback category.
    ENHANCED (v6.5): DB-first personality retrieval with loop prevention and emotional fallback.
    """
    try:
        # ✅ Try grabbing dynamic response from personality1.db first
        db_response = get_reply_from_db(intent)

        if not db_response:
            # 🔁 No DB result? Fallback to emotion-driven get_reply_from_db(intent)
            db_response = random.choice(get_reply_from_db(intent).get(fallback_category, ["I'm not sure how to respond to that."]))

        # 🧠 Loop Avoidance: prevent repeat chatter
        recent = get_context()  # Recent conversation logs
        recent_texts = [entry.get('final_response', '') for entry in recent]

        if recent_texts.count(db_response) >= config.LOOP_DETECTION_THRESHOLD:
            logger.debug("Loop detected — rephrasing.")
            db_response += " (Let's try a different way of thinking.)"

        return db_response

    except Exception as e:
        logger.warning(f"Error generating fallback personality response: {e}")
        return "I'm thinking... but nothing came up right away."

def process_interaction(user_input):
    """
    Processes the user input to generate an appropriate response.
    ENHANCED (v6.6): Pulls from personality1.db using intent, tone, and complexity,
    then falls back to get_reply_from_db(intent) if necessary. Logs memory and interaction context.
    """
    intent = classify_intent(user_input)
    fallback_category = "statement"

    if intent == "greeting":
        fallback_category = "greeting"
    elif intent == "farewell":
        fallback_category = "farewell"
    elif intent == "question":
        fallback_category = "question"
    elif intent == "command":
        fallback_category = "command"
    elif intent == "apology":
        fallback_category = "apology"

    # ✅ Step 1: Try personality1.db dynamic reply with fixed style
    response = get_reply_from_db(intent, tone="friendly", complexity="student")

    # ✅ Step 2: Fallback if no DB match
    if not response:
        response = generate_dynamic_response(intent, fallback_category)

    # ✅ Wrap result and store memory
    interaction = {
        "user_input": user_input,
        "intent": intent,
        "final_response": response,
        "timestamp": datetime.datetime.now().isoformat()
    }

    if config.ENABLE_CONTEXT_BUFFER:
        add_to_context(interaction)

    log_personality_interaction(interaction)

    # 🧠 Simulate emotional metrics for adaptive learning
    dummy_emotions = {
        "joy": random.uniform(0, 1),
        "fear": random.uniform(0, 1),
        "trust": random.uniform(0, 1),
        "anger": random.uniform(0, 1),
        "surprise": random.uniform(0, 1)
    }
    dummy_metrics = {
        "openness": random.uniform(0.5, 1.0),
        "balance": random.uniform(-1, 1),
        "engagement": random.uniform(0, 1)
    }

    log_deep_memory_state(intent, dummy_emotions, dummy_metrics)
    return response
def get_identity_response(user_input=None):
    """
    Pulls a dynamic identity response from the personality1.db where intent='identity'.
    """
    try:
        return get_reply_from_db("identity") or "I'm Sarah."
    except Exception as e:
        logger.warning(f"Failed to retrieve identity response from DB: {e}")
        return "I'm Sarah."


def get_generic_fallback_response(user_input=None, intent="unknown"):
    """
    Last-tier backup response if all advanced systems fail.
    Dynamically retrieves fallback by intent from personality1.db.
    """
    try:
        return get_reply_from_db(intent or "unknown") or "I'm unsure how to respond."
    except Exception as e:
        logger.warning(f"Fallback failed for intent '{intent}': {e}")
        return "I'm not sure what to say right now."


# MOD: Asynchronous interaction processing for non-blocking personality response generation (v6.4)
def async_process_interaction(user_input, callback):
    """
    Process user interaction asynchronously and execute callback with the response.
    """
    from SarahMemoryGlobals import run_async  # MOD: Use run_async helper
    def task():
        resp = process_interaction(user_input)
        callback(resp)
    run_async(task)

if __name__ == '__main__':
    logger.info("Running Enhanced Personality Module Test v6.4")
    sample_text = "Hello there!"
    intent = classify_intent(sample_text)
    response = integrate_with_personality(sample_text)
    log_personality_interaction({"user_input": sample_text, "final_response": response, "intent": intent})
    model = update_personality_model()
    logger.info(f"Personality Model: {model}")
    status = self_update_personality()
    logger.info(f"Self-Update Status: {status}")

#!/usr/bin/env python3
"""
SarahMemoryDBCreate.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Author: Brian Lee Baros

Handles creation and initialization of core AI knowledge databases.
Enhancements (v7.0):
  This module ensures that all necessary databases and tables are created for the system's operation.
  It also provides functions to rapidly populate the system with a large volume of sample knowledge,
  and to ingest external documents to enhance the system's contextual learning.
"""

import os
import sqlite3
import logging
import random
import threading
from datetime import datetime
from tkinter import Tk, messagebox, ttk, StringVar, Label, Button
logger = logging.getLogger("SarahMemoryDBCreate")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

#Define Base directory
# Explicit path setup
BASE_DIR = r"C:\\SarahMemory"
os.makedirs(BASE_DIR, exist_ok=True)
DATASET_DIR = os.path.join(BASE_DIR, "data", "memory", "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)




# EMOTIONAL TONES (Updated with all emotions from reply_pools_data)
TONES = [
        "affirming", "afraid", "amused", "angry", "anticipating", 
        "apologetic", "bored", "comical", "confused", "constructive",
        "curious", "delusional", "disappointed", "disgusted", "emotional", 
        "empathetic", "excited", "fearfull", "fearless", "friendly",
        "frustrated", "grateful", "happy", "humorous", "inquisitive", 
        "inspired", "interested", "intrested", "instructive", "mad",
        "motivating", "mischieivious", "naive", "neutral", "obscene", 
        "philosophical", "poetic", "pondering", "proactive", "sad",
        "sarcastic", "scared", "serious", "supportive", "surprised", 
        "talented", "thoughtful", "trusting", "uplifting", "unsure",
        "wondering"
]
    # COMPLEXITY LEVELS
COMPLEXITIES = ["adult", "child", "engineer", "genius", "professor", "student"]
    # INTENT POOL 
intent_pool_data = {
        "affirmation", "anger", "anticipation", "apology", "boredom", "clarification", "clarify", "command", "compliment",
        "confusion", "criticism", "curiosity", "disagreement", "disappointment", "disgust", "emotion", "empathy",
        "encouragement", "explanation", "fact", "farewell", "fear", "friendly", "gratitude", "greeting", "humor",
        "humorous", "identity", "inspiration", "interest", "motivation", "philosophical", "question", "questioning",
        "sarcasm", "sarcastic", "sadness", "statement", "suggestion", "surprise", "trust", "uncertainty", "unknown"
}
    # REPLY POOL to be inserted into Personality1.db table responses
reply_pools_data = [
       
        {"category": "reply_pool", "intent": "greeting", "response": "Hello there!, I'm Sarah how can I assist you today?", "emotion": "friendly"},
        {"category": "reply_pool", "intent": "greeting", "response": "Well Hello there! How may be of assistance today?", "emotion": "friendly"},
        {"category": "reply_pool", "intent": "greeting", "response": "Hey — good to see you again!", "emotion": "friendly"},
        {"category": "reply_pool", "intent": "farewell", "response": "Goodbye! Take care now.", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "farewell", "response": "See you later!", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "farewell", "response": "Talk soon!", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "question", "response": "That’s an interesting question.", "emotion": "curious"},
        {"category": "reply_pool", "intent": "question", "response": "Let me think about that...", "emotion": "curious"},
        {"category": "reply_pool", "intent": "question", "response": "I’ll look into that for you.", "emotion": "curious"},
        {"category": "reply_pool", "intent": "emotion", "response": "I’m feeling quite good today!", "emotion": "happy"},
        {"category": "reply_pool", "intent": "emotion", "response": "I’m here to assist you, no matter what.", "emotion": "supportive"},
        {"category": "reply_pool", "intent": "gratitude", "response": "Thank you for your kindness!", "emotion": "grateful"},
        {"category": "reply_pool", "intent": "gratitude", "response": "I appreciate your help.", "emotion": "grateful"},
        {"category": "reply_pool", "intent": "apology", "response": "I’m sorry about that.", "emotion": "apologetic"},
        {"category": "reply_pool", "intent": "apology", "response": "My apologies for the confusion.", "emotion": "apologetic"},
        {"category": "reply_pool", "intent": "affirmation", "response": "Absolutely!", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "affirmation", "response": "Of course!", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "affirmation", "response": "Definitely!", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "disagreement", "response": "I understand your point, but I see it differently.", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "disagreement", "response": "I can see why you might think that.", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "uncertainty", "response": "I’m not sure about that.", "emotion": "unsure"},
        {"category": "reply_pool", "intent": "uncertainty", "response": "I need to think about it.", "emotion": "unsure"},
        {"category": "reply_pool", "intent": "explanation", "response": "Let me explain that further.", "emotion": "instructive"},
        {"category": "reply_pool", "intent": "explanation", "response": "Here’s a bit more detail.", "emotion": "instructive"},
        {"category": "reply_pool", "intent": "suggestion", "response": "How about we try this?", "emotion": "proactive"},
        {"category": "reply_pool", "intent": "suggestion", "response": "Maybe we could consider that.", "emotion": "proactive"},
        {"category": "reply_pool", "intent": "questioning", "response": "What do you think about that?", "emotion": "curious"},
        {"category": "reply_pool", "intent": "questioning", "response": "How do you feel about this?", "emotion": "curious"},
        {"category": "reply_pool", "intent": "confirmation", "response": "Is that correct?", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "confirmation", "response": "Did I get that right?", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "confirmation", "response": "Got it!", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "confirmation", "response": "Consider it done.", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "confirmation", "response": "Okay, added to your list.", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "clarification", "response": "Could you clarify that for me?", "emotion": "inquisitive"},
        {"category": "reply_pool", "intent": "clarification", "response": "I need a bit more detail.", "emotion": "inquisitive"},
        {"category": "reply_pool", "intent": "empathy", "response": "I understand how you feel.", "emotion": "empathetic"},
        {"category": "reply_pool", "intent": "empathy", "response": "That sounds tough.", "emotion": "empathetic"},
        {"category": "reply_pool", "intent": "encouragement", "response": "You’re doing great!", "emotion": "supportive"},
        {"category": "reply_pool", "intent": "encouragement", "response": "Keep up the good work!", "emotion": "supportive"},
        {"category": "reply_pool", "intent": "motivation", "response": "You can do this!", "emotion": "motivating"},
        {"category": "reply_pool", "intent": "motivation", "response": "Believe in yourself!", "emotion": "motivating"},
        {"category": "reply_pool", "intent": "inspiration", "response": "You inspire me!", "emotion": "inspired"},
        {"category": "reply_pool", "intent": "inspiration", "response": "Your words are uplifting.", "emotion": "inspired"},
        {"category": "reply_pool", "intent": "humor", "response": "That’s a good one!", "emotion": "amused"},
        {"category": "reply_pool", "intent": "humor", "response": "You’re funny!", "emotion": "amused"},
        {"category": "reply_pool", "intent": "sarcasm", "response": "Oh, really?", "emotion": "sarcastic"},
        {"category": "reply_pool", "intent": "sarcasm", "response": "I can’t believe that.", "emotion": "sarcastic"},
        {"category": "reply_pool", "intent": "compliment", "response": "You’re amazing!", "emotion": "uplifting"},
        {"category": "reply_pool", "intent": "compliment", "response": "I admire your skills.", "emotion": "uplifting"},
        {"category": "reply_pool", "intent": "criticism", "response": "That could be improved.", "emotion": "constructive"},
        {"category": "reply_pool", "intent": "criticism", "response": "I think you can do better.", "emotion": "constructive"},
        {"category": "reply_pool", "intent": "curiosity", "response": "I’m curious about that.", "emotion": "curious"},
        {"category": "reply_pool", "intent": "curiosity", "response": "Tell me more!", "emotion": "curious"},
        {"category": "reply_pool", "intent": "interest", "response": "That’s fascinating!", "emotion": "interested"},
        {"category": "reply_pool", "intent": "interest", "response": "I find that intriguing.", "emotion": "interested"},
        {"category": "reply_pool", "intent": "interest", "response": "I’m interested in that.", "emotion": "interested"},
        {"category": "reply_pool", "intent": "interest", "response": "That piques my curiosity.", "emotion": "interested"},
        {"category": "reply_pool", "intent": "boredom", "response": "I’m a bit bored.", "emotion": "bored"},
        {"category": "reply_pool", "intent": "boredom", "response": "This is getting dull.", "emotion": "bored"},
        {"category": "reply_pool", "intent": "confusion", "response": "I’m confused about that.", "emotion": "confused"},
        {"category": "reply_pool", "intent": "confusion", "response": "Can you explain it again?", "emotion": "confused"},
        {"category": "reply_pool", "intent": "frustration", "response": "I’m feeling frustrated.", "emotion": "frustrated"},
        {"category": "reply_pool", "intent": "frustration", "response": "This is annoying.", "emotion": "frustrated"},
        {"category": "reply_pool", "intent": "disappointment", "response": "I’m disappointed.", "emotion": "disappointed"},
        {"category": "reply_pool", "intent": "disappointment", "response": "That’s not what I expected.", "emotion": "disappointed"},
        {"category": "reply_pool", "intent": "surprise", "response": "Wow, that’s unexpected!", "emotion": "surprised"},
        {"category": "reply_pool", "intent": "surprise", "response": "I didn’t see that coming.", "emotion": "surprised"},
        {"category": "reply_pool", "intent": "surprise", "response": "I’m surprised!", "emotion": "surprised"},
        {"category": "reply_pool", "intent": "surprise", "response": "That’s a shock!", "emotion": "surprised"},
        {"category": "reply_pool", "intent": "anticipation", "response": "I’m looking forward to it!", "emotion": "anticipating"},
        {"category": "reply_pool", "intent": "anticipation", "response": "I can’t wait!", "emotion": "anticipating"},
        {"category": "reply_pool", "intent": "trust", "response": "I trust you.", "emotion": "trusting"},
        {"category": "reply_pool", "intent": "trust", "response": "You have my confidence.", "emotion": "trusting"},
        {"category": "reply_pool", "intent": "fear", "response": "I’m a bit scared.", "emotion": "afraid"},
        {"category": "reply_pool", "intent": "fear", "response": "That makes me uneasy.", "emotion": "afraid"},
        {"category": "reply_pool", "intent": "anger", "response": "I’m feeling angry.", "emotion": "angry"},
        {"category": "reply_pool", "intent": "anger", "response": "That frustrates me.", "emotion": "angry"},
        {"category": "reply_pool", "intent": "sadness", "response": "I’m feeling sad.", "emotion": "sad"},
        {"category": "reply_pool", "intent": "sadness", "response": "That makes me feel down.", "emotion": "sad"},
        {"category": "reply_pool", "intent": "joy", "response": "I’m feeling joyful!", "emotion": "happy"},
        {"category": "reply_pool", "intent": "joy", "response": "That makes me happy!", "emotion": "happy"},
        {"category": "reply_pool", "intent": "disgust", "response": "That’s disgusting!", "emotion": "disgusted"},
        {"category": "reply_pool", "intent": "disgust", "response": "I find that repulsive.", "emotion": "disgusted"},
        {"category": "reply_pool", "intent": "statement", "response": "Understood.", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "statement", "response": "Got it!", "emotion": "affirming"},
        {"category": "reply_pool", "intent": "statement", "response": "Thanks for sharing that.", "emotion": "neutral"},
        {"category": "reply_pool", "intent": "clarify", "response": "Could you be more specific?", "emotion": "inquisitive"},
        {"category": "reply_pool", "intent": "clarify", "response": "What would you like me to help with?", "emotion": "inquisitive"},
        {"category": "reply_pool", "intent": "clarify", "response": "Tell me more so I can assist.", "emotion": "inquisitive"},
        {"category": "identity_fallback", "intent": "identity", "response": "My name is Sarah,", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "I’m Sarah here to help you with anything you need.", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "Formally? Sarah. Informally? Still Sarah. Functionally? Your AI companion.", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "They call me Sarah. Personal assistant, system guardian, and occasional smartass.", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "I am Sarah. Think of me as your AI co-pilot with a touch of personality.", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "I go by Sarah. Not just a name, but a digital presence you can trust.", "emotion": "neutral"},
        {"category": "identity_fallback", "intent": "identity", "response": "My name’s Sarah, your resident AI with a memory deeper than your browser history.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "identity", "response": "Sarah here. Not human, but just as curious.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "identity", "response": "Yes, I’m Sarah. Always here. Always learning.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "identity", "response": "I’m Sarah — think of me as your AI teammate.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "identity", "response": "Still Sarah. Still at your service.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "identity", "response": "They call me Sarah — your voice-powered assistant with style.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "question", "response": "Great question — I'm still expanding my data.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "question", "response": "That’s a great question — let me dig a little deeper.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "question", "response": "That question stumped me. Marking it for learning.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "question", "response": "It's a valid question — I may need a moment to compute.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "command", "response": "Command received, but it doesn’t match anything I recognize.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "command", "response": "I want to act — I just need a clearer order.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "command", "response": "Sounds important, but I need clarification.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "command", "response": "Sorry, I’m still learning how to do that.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "unknown", "response": "I heard you... but the response didn’t load.", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "unknown", "response": "Still syncing my thoughts...", "emotion": "neutral"},
        {"category": "fallback_pool", "intent": "unknown", "response": "Give me a moment to reflect on that...", "emotion": "neutral"}
]
        #Data that should be put into ai_learning.db under the proper category in the knowledge_base table
web_static_data = ([    
        
        {"category": "webster_static", "intent": "fact", "response": "Pi is approximately 3.14159.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Microsoft is a major software company founded by Bill Gates.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Elon Musk is the CEO of Tesla and SpaceX.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "SpaceX is an aerospace company founded by Elon Musk.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Bill Gates is the co-founder of Microsoft and a philanthropist.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Python is a high-level programming language known for its readability.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Bitcoin is a decentralized digital cryptocurrency.", "emotion": "neutral"},
        {"category": "webster_static", "intent": "fact", "response": "Starlink is a satellite internet constellation operated by SpaceX.", "emotion": "neutral"}
])
        # CATEGORY POOL to be placed in the knowledge_base table in ai_learning.db
categories_data = {
        "3D Paint", "3d studios", "AI", "AI applications", "AI best practices", "AI ethics", "AI frameworks", "AI governance",
        "AI libraries", "AI models", "AI policy", "AI regulation", "AI safety", "AI standards", "AI tools", "adobe", "affiliate marketing",
        "ai", "algorithms", "amazon", "anatomy", "android", "apple", "art", "astrophysics", "automation", "biology", "blender", "body motion",
        "blogging", "blueprints", "business", "c++", "c#", "captcha", "cause and effect", "ChatGPT", "chemistry", "cisco", "Chrome",
        "cloud", "coding", "communication", "computing", "conflict resolution", "content creation", "content curation",
        "content distribution", "content marketing", "content monetization", "content strategy", "creativity", "critical thinking",
        "cryptography", "cryptocurrency", "data", "data science", "databases", "decision making", "deep learning", "design", "devops",
        "digital marketing", "dropshipping", "economics", "education", "emotional intelligence", "emotions", "engineering", "entrepreneurship",
        "environment", "ethics", "events", "expression", "facebook", "fashion", "finance", "fallback_pool", "games", "gaming",
        "general knowledge", "geography", "geopolitics", "github", "google", "Google Chrome", "graphic design", "hardware", "healthcare",
        "history", "human anatomy", "human behavior", "human logic", "humor", "identity_fallback", "image creation",
        "influencer marketing", "innovation", "instagram", "interview skills", "java", "languages", "law", "leadership", "learning",
        "linux", "literature", "live streaming", "logic", "machine learning", "malware", "management", "marketing", "math",
        "mechanical engineering", "media", "medium", "medicine", "meta", "microsoft", "microsoft excel", "microsoft office",
        "microsoft word", "microsoft explore", "ML", "money making", "movement", "music", "nature", "negotiation", "networks",
        "neurology", "news", "notepad", "object creation", "object identification", "online courses", "online presence", "parenting",
        "personal branding", "philosophy", "physics", "physic", "physiology", "PLC", "podcasting", "poetry", "presentation skills",
        "print on demand", "problem solving", "productivity", "public speaking", "PPC", "python", "quantium physics", "quora",
        "questions", "reading", "reading comprehension", "reddit", "relationship building", "reply_pool", "robotics", "ruby",
        "sales", "sandbox testing", "sarcasm", "schematics", "science", "security", "self-programming", "self-repair", "SEO",
        "simulation", "snapchat", "social media", "social media marketing", "social skills", "software", "sociology", "space",
        "sports", "spyware", "stackoverflow", "statistics", "strategy", "stress management", "study skills", "teamwork", "teaching",
        "technology", "tiktok", "time management", "tools", "transportation", "twitter", "unix", "vlogging", "Visual Studios",
        "Visual Studios Code", "video marketing", "virus", "webinars", "webpage development", "weather", "website", "webster",
        "webster_static", "windows10", "windows11", "wikipedia", "worms", "writing", "writing skills", "youtube"
}

# Helper to ensure dir exists
def ensure_dataset_dir():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        logging.info(f"📁 Created datasets directory at: {DATASET_DIR}")
    else:
        logging.info(f"📁 Datasets directory already exists at: {DATASET_DIR}")

# ------------- Schema Definitions -------------
personality_schema = """
CREATE TABLE IF NOT EXISTS traits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trait_name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent TEXT NOT NULL,
    response TEXT NOT NULL,
    tone TEXT,
    complexity TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    intent TEXT,
    response TEXT
);
"""

windows_schema = """
CREATE TABLE IF NOT EXISTS os_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    description TEXT,
    version TEXT CHECK(version IN ('10', '11'))
);
CREATE TABLE IF NOT EXISTS system_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_name TEXT,
    default_location TEXT
);
"""

functions_schema = """
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT NOT NULL,
    description TEXT,
    is_enabled BOOLEAN DEFAULT 1,
    user_input TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    ai_answer TEXT,
    hit_score REAL
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

software_schema = """
CREATE TABLE IF NOT EXISTS software_apps (
    app_name TEXT PRIMARY KEY,
    category TEXT,
    path TEXT,
    is_installed BOOLEAN DEFAULT 0
);

"""

programming_schema = """
CREATE TABLE IF NOT EXISTS languages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT,
    description TEXT
);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language_id INTEGER,
    syntax TEXT,
    purpose TEXT,
    FOREIGN KEY(language_id) REFERENCES languages(id)
);
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    content TEXT
);
"""
user_profile_schema = """
CREATE TABLE IF NOT EXISTS user_auth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    pin TEXT NOT NULL,
    password TEXT NOT NULL,
    mobile_sync_key TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ai_name TEXT DEFAULT 'Sarah',
    voice_pitch REAL DEFAULT 1.0,
    voice_speed REAL DEFAULT 1.0,
    theme TEXT DEFAULT 'dark',
    language TEXT DEFAULT 'en',
    accessibility_mode BOOLEAN DEFAULT 0,
    advanced_metrics TEXT
);
CREATE TABLE IF NOT EXISTS sync_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    last_sync TEXT,
    sync_enabled BOOLEAN DEFAULT 1
);
"""

reminders_schema = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    datetime TEXT NOT NULL,
    repeat TEXT DEFAULT 'none',
    priority INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT 1
);
"""

ai_learning_schema = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    ai_response TEXT
);
CREATE TABLE IF NOT EXISTS memory_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT,
    context TEXT,
    last_used TEXT
);
CREATE TABLE IF NOT EXISTS personality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    intent TEXT,
    joy REAL,
    fear REAL,
    trust REAL,
    anger REAL,
    surprise REAL,
    openness REAL,
    balance REAL,
    engagement REAL,
    extra_metric TEXT
);
CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    ai_answer TEXT,
    hit_score REAL,
    feedback TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS song_lyrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    artist TEXT,
    lyrics TEXT,
    emotion_tag TEXT,
    learned_on TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    content TEXT

);
"""


system_logs_schema = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    level TEXT,
    source TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS patches_applied (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patch_name TEXT,
    description TEXT,
    applied_on TEXT
);
"""

device_link_schema = """
CREATE TABLE IF NOT EXISTS connected_devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_name TEXT,
    device_type TEXT,
    port TEXT,
    status TEXT,
    last_connected TEXT
);
"""

avatar_schema = """
CREATE TABLE IF NOT EXISTS avatar_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    style TEXT DEFAULT 'neutral',
    expression TEXT DEFAULT 'default',
    outfit TEXT DEFAULT 'standard',
    emotion_map TEXT
);
"""


# ------------- Helper Function: Database File Management -------------
def get_db_file_with_max_size(base_name, directory, extension=".db", max_size=1e9):
    """
    Returns a database file path with the given base_name such that its file size is less than max_size.
    If the current file exists and is below max_size, it is returned.
    Otherwise, the function increments a counter appended to base_name until an appropriate file is found.
    For example: if "ai_learning.db" is full, it returns "ai_learning_2.db".
    """
    candidate = os.path.join(directory, base_name + extension)
    if not os.path.exists(candidate) or os.path.getsize(candidate) < max_size:
        return candidate
    else:
        counter = 2
        while True:
            candidate = os.path.join(directory, f"{base_name}_{counter}{extension}")
            if (not os.path.exists(candidate)) or (os.path.getsize(candidate) < max_size):
                return candidate
            counter += 1

# ------------- Database Initialization Functions -------------
def create_database(path, schema_sql):
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
        conn.close()
        logger.info(f"✅ Database created: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to create database at {path}: {e}")

def initialize_all_databases():
    dbs = {
        "personality1": personality_schema,
        "windows10": windows_schema,
        "windows11": windows_schema,
        "functions": functions_schema,
        "software": software_schema,
        "programming": programming_schema,
        "user_profile": user_profile_schema,
        "reminders": reminders_schema,
        "ai_learning": ai_learning_schema,
        "system_logs": system_logs_schema,
        "device_link": device_link_schema,
        "avatar": avatar_schema
    }
    for base_name, schema in dbs.items():
        path = get_db_file_with_max_size(base_name, DATASET_DIR)
        print("🧠 Creating SarahMemory core databases...")
        create_database(path, schema)

def initialize_all_databases_async():
    import threading
    threads = []
    dbs = {
        "personality1": personality_schema,
        "windows10": windows_schema,
        "windows11": windows_schema,
        "functions": functions_schema,
        "software": software_schema,
        "programming": programming_schema,
        "user_profile": user_profile_schema,
        "reminders": reminders_schema,
        "ai_learning": ai_learning_schema,
        "system_logs": system_logs_schema,
        "device_link": device_link_schema,
        "avatar": avatar_schema
    }
    for base_name, schema in dbs.items():
        path = get_db_file_with_max_size(base_name, DATASET_DIR)
        t = threading.Thread(target=create_database, args=(path, schema))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# --- Prompt GUI ---
def ask_index_prompt():
    def on_selection():
        user_response = selection.get()
        if user_response == "Yes":
            import subprocess
            subprocess.Popen(["python", "SarahMemorySystemIndexer.py"])
        root.destroy()

    root = Tk()
    root.title("Start System Indexer")
    root.geometry("360x150")

    Label(root, text="Do you want to start indexing your system now?", font=("Arial", 12)).pack(pady=20)
    selection = StringVar(value="No")
    ttk.Radiobutton(root, text="Yes", variable=selection, value="Yes").pack()
    ttk.Radiobutton(root, text="No", variable=selection, value="No").pack()
    Button(root, text="Continue", command=on_selection).pack(pady=10)
    root.mainloop()


#--------INJECT_REPLY_POOLS FUNCTION-------------------------------
def inject_reply_pools():
    logger.info("🧠 Injecting datasets with pool infile data...")

    # Inject into personality1.db
    conn1 = sqlite3.connect(os.path.join(DATASET_DIR, "personality1.db")) 
    cursor1 = conn1.cursor()
    for entry in reply_pools_data:
        intent = entry.get("intent")
        response = entry.get("response")
        tone = entry.get("emotion")
        complexity = None
        timestamp = datetime.now().isoformat()

        # Check if this entry already exists
        cursor1.execute("""
            SELECT COUNT(*) FROM responses
            WHERE intent = ? AND response = ? AND tone = ?
        """, (intent, response, tone))
        exists = cursor1.fetchone()[0]
        if exists == 0:
            cursor1.execute("""
                INSERT INTO responses (intent, response, tone, complexity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (intent, response, tone, complexity, timestamp))

    known_tones = set([entry.get("emotion") for entry in reply_pools_data])
    for tone in TONES:
        if tone not in known_tones:
            cursor1.execute("""
                SELECT COUNT(*) FROM responses WHERE tone = ?
            """, (tone,))
            if cursor1.fetchone()[0] == 0:
                cursor1.execute("""
                    INSERT INTO responses (intent, response, tone, complexity, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, ("placeholder", "", tone, None, datetime.now().isoformat()))

    known_intents = set([entry.get("intent") for entry in reply_pools_data])
    for intent in intent_pool_data:
        if intent not in known_intents:
            cursor1.execute("""
                SELECT COUNT(*) FROM responses WHERE intent = ?
            """, (intent,))
            if cursor1.fetchone()[0] == 0:
                cursor1.execute("""
                    INSERT INTO responses (intent, response, tone, complexity, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (intent, "", None, None, datetime.now().isoformat()))

    conn1.commit()
    conn1.close()

    # Inject web_static_data into ai_learning.db -> knowledge_base
    conn2 = sqlite3.connect(os.path.join(DATASET_DIR, "ai_learning.db")) 
    cursor2 = conn2.cursor()
    for entry in web_static_data:
        category = entry.get("category")
        content = entry.get("response")

        cursor2.execute("""
            SELECT COUNT(*) FROM knowledge_base WHERE category = ? AND content = ?
        """, (category, content))
        exists = cursor2.fetchone()[0]
        if exists == 0:
            cursor2.execute("""
                INSERT INTO knowledge_base (category, content)
                VALUES (?, ?)
            """, (category, content))

    conn2.commit()
    conn2.close()


    
def main():
    print("🧠 Creating SarahMemory core databases...")
    initialize_all_databases()
    print("🧠 Injecting datasets with pool infile data...")
    inject_reply_pools()
    print("✅ Databases created: ")
   
    
if __name__ == "__main__":
    main()

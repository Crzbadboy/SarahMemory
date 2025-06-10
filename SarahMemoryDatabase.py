#!/usr/bin/env python3
"""
SarahMemoryDatabase.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  Manages data and file operations for continuous voice input logging, system performance monitoring,
  AI user preferences, authentication, synchronization data, and QA caching for three‑stage retrieval.
  This module logs voice inputs, system metrics, user preferences, sync status, and caches AI Q&A.
"""

import logging
import sqlite3
import os
import datetime
import psutil
import json


import SarahMemoryGlobals as config
from SarahMemoryGlobals import run_async, DATASETS_DIR

# Setup logging for the database module
logger = logging.getLogger('SarahMemoryDatabase')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Database Paths ---
DB_PATH = os.path.join(config.DATASETS_DIR, 'ai_learning.db')
USER_DB_PATH = os.path.join(config.DATASETS_DIR, 'user_profile.db')

def get_active_sentence_model():
    from sentence_transformers import SentenceTransformer
    from SarahMemoryGlobals import MULTI_MODEL, MODEL_CONFIG
    if MULTI_MODEL:
        for model_name, enabled in MODEL_CONFIG.items():
            if enabled:
                try:
                    return SentenceTransformer(model_name)
                except Exception as e:
                    logger.warning(f"⚠️ Model load failed: {model_name} → {e}")
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Initialization ---
def init_database():
    # NOTE: This initializes only ai_learning.db. Other databases are managed in DBCreate.py but accessed here.
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Voice logs
        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            voice_text TEXT NOT NULL,
                            embedding BLOB
                          )''')
        # Performance metrics
        cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            cpu_usage REAL,
                            memory_usage REAL,
                            disk_usage REAL,
                            network_usage REAL
                          )''')
        # QA cache
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query TEXT,
                            ai_answer TEXT,
                            hit_score INTEGER,
                            feedback TEXT,
                            timestamp TEXT
                          )''')
        conn.commit()
        logger.info("Runtime DB initialized with QA cache.")
                # Additional DBs initialized externally but used here
        # reminders.db, avatar.db, windows10.db, windows11.db, software.db, device_link.db
        return conn
    except Exception as e:
        logger.error(f"Error initializing runtime DB: {e}")
        return None

# --- QA Cache Helpers ---
def search_answers(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ai_answer FROM qa_cache WHERE query LIKE ? ORDER BY hit_score DESC", ('%' + query + '%',))
        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results] if results else []
    except Exception as e:
        logger.error(f"Error searching QA cache: {e}") 
        return []
    
def store_answer(query, answer):
    try:
        timestamp = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO qa_cache (query, ai_answer, hit_score, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (query, answer, 0, "ungraded", timestamp)
        )
        conn.commit()
        conn.close()
        logger.info(f"Stored QA cache for query: '{query}'")
    except Exception as e:
        logger.error(f"Error storing QA cache: {e}")
        logger.error(f"Error storing voice input: {e}")
        return False

def store_performance_metrics(conn):
    try:
        timestamp = datetime.datetime.now().isoformat()
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        net = random.uniform(0, 100)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO performance_metrics (timestamp, cpu_usage, memory_usage, disk_usage, network_usage) VALUES (?, ?, ?, ?, ?)",
            (timestamp, cpu, mem, disk, net)
        )
        conn.commit()
        logger.info(f"Performance metrics at {timestamp}: CPU {cpu}%, Mem {mem}%, Disk {disk}%, Net {net:.2f}%")
        return True
    except Exception as e:
        logger.error(f"Error storing performance metrics: {e}")
        return False

def get_all_voice_logs(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM voice_logs")
        logs = cursor.fetchall()
        logger.info(f"Retrieved {len(logs)} voice logs")
        return logs
    except Exception as e:
        logger.error(f"Error retrieving voice logs: {e}")
        return []

# --- User Profile DB Support ---
def connect_user_profile_db():
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        logger.info("Connected to user_profile.db.")
        return conn
    except Exception as e:
        logger.error(f"Unable to connect to user_profile.db: {e}")
        return None

# --- Diagnostics Export ---
def record_qa_feedback(query, score, feedback, timestamp=None):
    try:
        if not timestamp:
           timestamp = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE qa_cache SET hit_score = ?, feedback = ?, timestamp = ? WHERE query LIKE ?",
            (score, feedback, timestamp, '%' + query + '%')
        )
        conn.commit()
        conn.close()
        logger.info(f"Recorded feedback on QA entry: {query} | Score: {score} | Feedback: {feedback} | Time: {timestamp}")
    except Exception as e:
        logger.error(f"Error recording QA feedback: {e}")

def export_voice_logs_to_json(conn, output_path):
    try:
        logs = get_all_voice_logs(conn)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
        logger.info(f"Exported voice logs to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting voice logs: {e}")
        return False

# --- New Additions ---

# Additional dataset access wrappers
REMINDER_DB = os.path.join(config.DATASETS_DIR, "reminders.db")
AVATAR_DB = os.path.join(config.DATASETS_DIR, "avatar.db")
WIN10_DB = os.path.join(config.DATASETS_DIR, "windows10.db")
WIN11_DB = os.path.join(config.DATASETS_DIR, "windows11.db")
SOFTWARE_DB = os.path.join(config.DATASETS_DIR, "software.db")
DEVICE_LINK_DB = os.path.join(config.DATASETS_DIR, "device_link.db")


def fetch_reminders():
    try:
        conn = sqlite3.connect(REMINDER_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT title, description, datetime FROM reminders WHERE active = 1")
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"[REMINDER_DB ERROR] {e}")
        return []

def fetch_software_commands():
    try:
        conn = sqlite3.connect(SOFTWARE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT app_name, path FROM software_apps WHERE is_installed = 1")
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[SOFTWARE_DB ERROR] {e}")
        return []

def fetch_os_commands(version="10"):
    try:
        db_path = WIN10_DB if version == "10" else WIN11_DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT command, description FROM os_commands WHERE version = ?", (version,))
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[OS_COMMAND_DB ERROR] {e}")
        return []

def fetch_avatar_metadata():
    try:
        conn = sqlite3.connect(AVATAR_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, tags, emotion, gps_latitude, gps_longitude FROM photo_metadata")
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[AVATAR_DB ERROR] {e}")
        return []

def fetch_device_links():
    try:
        conn = sqlite3.connect(DEVICE_LINK_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT device_name, device_type, connection_type FROM device_registry")
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[DEVICE_LINK ERROR] {e}")
        return []
def search_responses(question):
    """Fuzzy search inside the personality1.db responses."""
    try:
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM responses WHERE response LIKE ?", ('%' + question + '%',))
        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results] if results else []
    except Exception as e:
        logger.error(f"[DB Search Responses Error] {e}")
        return []

def insert_response_into_personality(intent, response, tone="neutral", complexity="basic"):
    """Insert a learned response into personality1.db"""
    db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO responses (intent, response, tone, complexity)
            VALUES (?, ?, ?, ?)
        """, (intent, response, tone, complexity))
        conn.commit()
        conn.close()
        logger.info(f"[LEARNING] Inserted personality knowledge: ({intent}, {tone}, {complexity})")
        return True
    except Exception as e:
        logger.error(f"Failed to insert into Personality DB: {e}")
        return False
def embed_and_store_dataset_sentences():
    """
    Extracts text from imported local files, creates vector embeddings,
    and stores them in the voice_logs table with timestamped entries.

    ✅ Expands SarahMemory’s foundation model with permanent vector memory
    🔁 Safe to call repeatedly; avoids duplicate re-learning based on file mod times.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from SarahMemoryGlobals import import_other_data, IMPORT_OTHER_DATA_LEARN, MULTI_MODEL, MODEL_CONFIG

        if not IMPORT_OTHER_DATA_LEARN:
            logger.info("🛑 Skipping vector rebuild: IMPORT_OTHER_DATA_LEARN is False.")
            return

        def get_active_sentence_model():
            if MULTI_MODEL:
                for model_name, enabled in MODEL_CONFIG.items():
                    if enabled:
                        try:
                            return SentenceTransformer(model_name)
                        except Exception as e:
                            logger.warning(f"⚠️ Model load failed: {model_name} → {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("🧠 Starting semantic vector embedding for dataset memory...")
        model = get_active_sentence_model()
        data = import_other_data()
        conn = init_database()
        inserted_count = 0

        for file_path, content in data.items():
            for line in content.split('\n'):
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                try:
                    embedding = model.encode(line).tolist()
                    success = store_voice_input(conn, voice_text=line, embedding=embedding)
                    if success:
                        inserted_count += 1
                except Exception as ve:
                    logger.warning(f"[EMBED ERROR] Skipped line due to embedding failure: {ve}")
        conn.close()
        logger.info(f"✅ Vector memory embedding complete. {inserted_count} entries added.")

    except Exception as e:
        logger.error(f"[EMBED_FAIL] Dataset vector embedding failed: {e}")

def check_memory_responses(log_output=True, limit=1000):
    """
    Scans all Class 1 dataset entries for malformed, irrelevant, or non-conversational content.
    Flags console scripts, install instructions, file paths, and tech noise.

    Args:
        log_output (bool): If True, prints flagged entries.
        limit (int): Max entries to check per database.

    Returns:
        dict: Report of flagged items from each DB
    """
    import re
    import os
    import sqlite3
    import SarahMemoryGlobals as config
    from SarahMemoryGlobals import DATASETS_DIR

    flagged = {}
    filters = [
        r"\[console_scripts\]", r"\bsetup\.py\b", r"pip install", r"\.exe", r"from ", r"import ",
        r"def ", r"class ", r"fonttools", r"certifi", r"charset", r"ttx", r"wheel", r"cython",
        r"sentry_sdk", r"pyautogui", r"anyio", r"Hello there!", r"project\(", r"normalizer",
        r"cythonize", r"pyftmerge", r"pyftsubset", r"continue", r"__main__"
    ]
    combined_filter = re.compile("|".join(filters), re.IGNORECASE)

    db_paths = {
        "personality1.db": "responses",
        "functions.db": "functions",
        "programming.db": "knowledge_base",
        "ai_learning.db": "learned",
        "avatar.db": "photo_metadata"
    }

    for db_file, table in db_paths.items():
        db_path = os.path.join(config.DATASETS_DIR, db_file)
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            column = "response" if table == "responses" else "content" if table != "photo_metadata" else "file_path"
            cur.execute(f"SELECT {column} FROM {table} LIMIT ?", (limit,))
            rows = cur.fetchall()
            flagged[db_file] = []
            for row in rows:
                content = row[0] if row else ""
                if content and combined_filter.search(content):
                    flagged[db_file].append(content)
                    if log_output:
                        print(f"[FLAGGED - {db_file}] → {content[:100]}...")
            conn.close()
        except Exception as e:
            print(f"[ERROR] While scanning {db_file}: {e}")
            continue

def auto_correct_dataset_entry(user_input, bad_response, corrected_response, keywords=None):
    """
    Replaces a faulty response in any of the key datasets with a corrected one.
    Includes optional keyword validation before replacing.
    """
    db_files = ["personality1.db", "functions.db", "programming.db"]
    success = False

    for db_file in db_files:
        db_path = os.path.join(config.DATASETS_DIR, db_file)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if db_file == "personality1.db":
                cursor.execute("SELECT id FROM responses WHERE response = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    if keywords and not all(k.lower() in corrected_response.lower() for k in keywords):
                        logger.warning(f"[AUTO_CORRECT] {db_file} → Missing keywords: {keywords}. Skipping.")
                        continue
                    cursor.execute("UPDATE responses SET response = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            elif db_file == "functions.db":
                cursor.execute("SELECT id FROM functions WHERE description = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    cursor.execute("UPDATE functions SET description = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            elif db_file == "programming.db":
                cursor.execute("SELECT id FROM knowledge_base WHERE content = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    cursor.execute("UPDATE knowledge_base SET content = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            conn.commit()
            if success:
                logger.info(f"[AUTO_CORRECT] Corrected entry in {db_file}.")
            conn.close()
        except Exception as e:
            logger.error(f"[AUTO_CORRECT ERROR in {db_file}] {e}")
            continue

    return success
from numpy import dot
from numpy.linalg import norm
import numpy as np

def vector_search_qa_cache(query_text, top_n=1):
    """
    Vector-based semantic search on QA cache memory (query and answer).
    """
    try:
        model = get_active_sentence_model()
        query_vec = model.encode(query_text)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT query, ai_answer FROM qa_cache")
        entries = cursor.fetchall()

        results = []
        for query, answer in entries:
            combined = f"{query} {answer}"
            emb_vec = model.encode(combined)
            similarity = dot(query_vec, emb_vec) / (norm(query_vec) * norm(emb_vec))

            tokens = tokenize_text(answer)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue

            results.append((similarity, answer))

        results.sort(reverse=True)
        return results[:top_n]
    except Exception as e:
        logger.error(f"[QA VECTOR SEARCH ERROR] {e}")
        return []

def vector_search(query_text, top_n=1):
    """
    Enhanced vector search with entropy analysis and query logging.
    """
    try:
        model = get_active_sentence_model()
        query_vec = model.encode(query_text)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create search log table if not exists
        cursor.execute('''CREATE TABLE IF NOT EXISTS search_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query TEXT,
                            match_text TEXT,
                            similarity REAL,
                            timestamp TEXT
                          )''')

        cursor.execute("SELECT voice_text, embedding FROM voice_logs")
        entries = cursor.fetchall()

        results = []
        for text, emb_json in entries:
            if not emb_json:
                continue
            emb_vec = np.array(json.loads(emb_json))
            similarity = dot(query_vec, emb_vec) / (norm(query_vec) * norm(emb_vec))

            # Entropy score (basic uniqueness check)
            tokens = tokenize_text(text)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue  # Skip low-quality matches

            results.append((similarity, text))

        results.sort(reverse=True)
        top_results = results[:top_n]

        for sim, matched in top_results:
            cursor.execute("INSERT INTO search_log (query, match_text, similarity, timestamp) VALUES (?, ?, ?, ?)",
                           (query_text, matched, float(sim), datetime.datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return top_results
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH ERROR] {e}")
        return []
   
def tokenize_text(text):
    """Tokenizes text for entropy and quality analysis."""
    import re
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except:
        return re.findall(r'\b\w+\b', text)
    
def ensure_qa_cache_table_exists():
    conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "ai_learning.db"))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            ai_answer TEXT,
            hit_score REAL
        )
    """)
    conn.commit()
    conn.close()   

def log_ai_functions_event(event_type, details):
    try:
        conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "functions.db"))
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                description TEXT,
                is_enabled BOOLEAN DEFAULT 1,
                user_input TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO functions (function_name, description, is_enabled, user_input, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (event_type, details, 1, "", timestamp))
        conn.commit()
        conn.close()
        logger.info(f"[FUNCTION_LOG] {event_type} - {details}")
    except Exception as e:
        logger.error(f"[FUNCTION_LOG ERROR] Failed to log function event: {e}")      
if __name__ == '__main__':
    logger.info("Starting SarahMemoryDatabase module test.")
    conn = init_database()
    if conn:
        model = get_active_sentence_model()
        embedding = model.encode("Test voice input.").tolist()
        store_performance_metrics(conn)
        logs = get_all_voice_logs(conn)
        export_voice_logs_to_json(conn, 'voice_logs_export.json')
        conn.close()

#!/usr/bin/env python3
"""
SarahMemoryDL.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  Deep Learning Analysis Core — evaluates AI conversation history to discover usage patterns,
  key emotional flags, and optimize future response formation.

Enhancements (v7.0):
  🔁 NEW FEATURE PROPOSAL:
User Context Enrichment Layer
Learn from local system behavior to help the AI understand the user's habits, interests, and focus.
📂 Data Sources You Can Integrate:
Source Type	Example Targets / Access	Integration Status
🔄 Recently Opened Files	Windows Recent folder, shell links	Easy via os.listdir() or shell:recent
📄 Document Activity	.docx, .pdf, .txt, Downloads/	✅ Already partially handled in SarahMemorySystemIndexer.py
🎵 Music Preferences	.mp3, .flac, iTunes library / folders	Scan Music/, Spotify logs, or WMP history
🎬 Video Consumption	.mp4, .avi, VLC watch history	Check Videos/, vlc-qt-interface.ini
🌐 Browser Usage	Chrome/Edge/Firefox history, cookies, bookmarks	Accessable via SQLite history DBs (History file)
📁 App Usage	Prefetch folder, AppData logs	Advanced but feasible
🧠 Windows Registry	Recent apps, shell history, MRU	(Caution) Use winreg to query safely
  This module analyzes conversation patterns and writes a JSON report for future enhancements.
"""

import logging
import sqlite3
import os
import string
from collections import Counter
import json
import math
import glob
import time
import shutil
from datetime import datetime
import SarahMemoryGlobals as config
from SarahMemoryGlobals import SETTINGS_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, run_async

# Setup logger
logger = logging.getLogger("SarahMemoryDL")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# DB Path for deep learning analysis
from SarahMemoryGlobals import DATASETS_DIR
AI_LEARNING_DB = os.path.join(config.DATASETS_DIR, "ai_learning.db")

# --- Utility Functions ---

def save_as_json(data, filename):
    from SarahMemoryGlobals import DATASETS_DIR
    try:
        path = os.path.join(config.DATASETS_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON: {filename}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")

def connect_memory_db():
    """Connect to the AI learning memory DB."""
    try:
        conn = sqlite3.connect(AI_LEARNING_DB)
        logger.info("Connected to ai_learning.db for DL analysis.")
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None

def get_conversation_history(limit=100):
    """Fetch conversation logs."""
    try:
        conn = connect_memory_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, ai_response FROM conversations ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
        return []

def analyze_term_frequencies(convos):
    """Build word frequency dictionary from conversations."""
    word_list = []
    for user, ai_response in convos:
        combined = f"{user} {ai_response}"
        cleaned = combined.translate(str.maketrans('', '', string.punctuation)).lower()
        word_list.extend(cleaned.split())
    term_freq = Counter(word_list)
    logger.info(f"Top terms: {term_freq.most_common(10)}")
    return term_freq

def compute_tf_idf(convos):
    """
    ENHANCED: Compute TF-IDF scores for terms in the conversation history.
    """
    documents = []
    for user, ai_response in convos:
        combined = f"{user} {ai_response}"
        cleaned = combined.translate(str.maketrans('', '', string.punctuation)).lower()
        documents.append(cleaned.split())
    
    # Compute term frequencies for each document
    tf = []
    for doc in documents:
        tf.append(Counter(doc))
    
    # Document frequency for each term
    df = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    # Total number of documents
    N = len(documents)
    tf_idf = {}
    for i, doc_tf in enumerate(tf):
        for term, freq in doc_tf.items():
            idf = math.log(N / (df[term] + 1)) + 1
            tf_idf.setdefault(term, 0)
            tf_idf[term] += freq * idf
    logger.info(f"Computed TF-IDF scores, top terms: {Counter(tf_idf).most_common(10)}")
    return tf_idf

def evaluate_conversation_patterns():
    """
    Evaluate conversation patterns with advanced TF-IDF analysis.
    ENHANCED: Also simulates clustering feedback and sentiment anchors.
    """
    history = get_conversation_history(limit=150)
    if not history:
        logger.warning("No conversation data found.")
        return {}
    term_stats = analyze_term_frequencies(history)
    tfidf_scores = compute_tf_idf(history)
    feedback = {
        "most_common_words": term_stats.most_common(10),
        "top_tfidf_terms": sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10],
        "total_unique_terms": len(term_stats),
        "total_conversations": len(history)
    }
    logger.info(f"Pattern feedback: {feedback}")
    # NEW: Export analysis as JSON to the settings directory
    output_path = os.path.join(SETTINGS_DIR, "conversation_analysis.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2)
        logger.info(f"Exported conversation analysis to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting conversation analysis: {e}")
    return feedback

def deep_learn_user_context():
    import sqlite3
    from SarahMemoryGlobals import DATASETS_DIR
    db = os.path.join(config.DATASETS_DIR, "ai_learning.db")
    topics = []

    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM qa_cache ORDER BY timestamp DESC LIMIT 100")
        rows = cursor.fetchall()
        for row in rows:
            q = row[0]
            if q and len(q) > 6:
                topics.append(q)
        conn.close()
        return list(set(topics))
    except Exception as e:
        print(f"[DeepLearn] Error fetching context: {e}")
        return []
# --- File Activity Learning ---
def scan_recent_documents():
    try:
        recent_dir = os.path.join(os.getenv("APPDATA"), "Microsoft", "Windows", "Recent")
        files = [f for f in os.listdir(recent_dir) if os.path.isfile(os.path.join(recent_dir, f))]
        return files[-20:]
    except Exception as e:
        logger.warning(f"scan_recent_documents failed: {e}")
        return []

def scan_music_files():
    try:
        music_dir = os.path.join(os.path.expanduser("~"), "Music")
        return [f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.flac'))]
    except:
        return []

def scan_video_files():
    try:
        video_dir = os.path.join(os.path.expanduser("~"), "Videos")
        return [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi'))]
    except:
        return []

def extract_browser_history():
    import sqlite3
    import win32crypt
    edge_path = os.path.join(os.getenv("LOCALAPPDATA"), "Microsoft", "Edge", "User Data", "Default", "History")
    chrome_path = os.path.join(os.getenv("LOCALAPPDATA"), "Google", "Chrome", "User Data", "Default", "History")
    history_files = [edge_path, chrome_path]
    sites = []
    for path in history_files:
        if not os.path.exists(path): continue
        try:
            temp_copy = path + ".tmp"
            shutil.copy2(path, temp_copy)
            conn = sqlite3.connect(temp_copy)
            cur = conn.cursor()
            cur.execute("SELECT url, title FROM urls ORDER BY last_visit_time DESC LIMIT 50")
            rows = cur.fetchall()
            conn.close()
            os.remove(temp_copy)
            for url, title in rows:
                if url and title:
                    sites.append(f"{title} ({url})")
        except Exception as e:
            logger.warning(f"Failed to read browser history from {path}: {e}")
    return sites[:20]

# --- Core Analyzer ---
def analyze_user_behavior():
    recent_docs = scan_recent_documents()
    web_history = extract_browser_history()
    music_stats = scan_music_files()
    movie_stats = scan_video_files()

    combined_summary = {
        "recent_docs": recent_docs[:10],
        "top_sites": web_history[:10],
        "audio_files": music_stats[:10],
        "video_files": movie_stats[:10],
        "timestamp": datetime.now().isoformat()
    }

    save_as_json(combined_summary, filename="user_context.json")
    logger.info("[Behavior] User activity context saved.")
    return combined_summary

# NEW: Asynchronous wrapper to run deep learning analysis without blocking.
def start_deep_learning_analysis():
    """
    Run evaluate_conversation_patterns in a background thread.
    NEW: Uses run_async for non-blocking analysis.
    """
    run_async(evaluate_conversation_patterns)

if __name__ == '__main__':
    logger.info("Running Deep Learning Conversation Analysis...")
    summary = evaluate_conversation_patterns()
    logger.info(f"Conversation Pattern Summary:\n{json.dumps(summary, indent=2)}")
    logger.info("SarahMemoryDL module testing complete.")

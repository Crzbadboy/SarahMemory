#!/usr/bin/env python3
"""
SarahMemoryCleanupDaily.py <Version 7.0 CLI Enhanced>
Automated Dataset Purification and Intelligence Correction Tool
Author: Brian Lee Baros

Features:
- Cleanses context history and QA cache intelligently
- Removes invalid/foreign/duplicate or low-quality rows
- Prevents over-cleaning using last-run timestamps
- CLI Support:
    --force          : Ignore 5-day threshold
    --dry-run        : Simulate without deleting
    --verbose        : Print each record purge
    --context-only   : Only clean context_history.db
    --qa-only        : Only clean ai_learning.db
"""

import sqlite3
import re
import logging
from langdetect import detect
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import os
import argparse
import SarahMemoryGlobals as config
from SarahMemoryGlobals import CONTEXT_HISTORY_DB_PATH

logger = logging.getLogger("SarahMemoryCleanup")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

GARBAGE_PATTERNS = [
    r"^\s*$",
    r"^\W{3,}$",
    r"[\u4e00-\u9fff]+",
    r"[\u0400-\u04FF]+",
    r"[\u0600-\u06FF]+",
    r"[^\x00-\x7F]+",
]

COLUMNS_TO_CLEAN = ["input", "output", "corrected_response"]
SIMILARITY_THRESHOLD = 0.90
LAST_RUN_FILE = "data/memory/cleanup_last_run.txt"


def is_garbage(text):
    try:
        if any(re.search(p, text) for p in GARBAGE_PATTERNS):
            return True
        if detect(text) != "en":
            return True
        return False
    except Exception:
        return True

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def purge_context_db(dry_run=False, verbose=False):
    conn = sqlite3.connect(CONTEXT_HISTORY_DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, input, output, corrected_response FROM context_history")
        entries = cur.fetchall()
        seen_clean = []
        for entry_id, input_text, output_text, corrected_text in entries:
            original = [input_text or "", output_text or "", corrected_text or ""]
            cleaned = [clean_text(x) for x in original]
            if any(is_garbage(x) for x in cleaned):
                logger.warning(f"Purging garbage ID={entry_id}: {original}")
                if not dry_run:
                    cur.execute("DELETE FROM context_history WHERE id = ?", (entry_id,))
                continue
            if any(similar(c1, c2) > SIMILARITY_THRESHOLD for c1 in seen_clean for c2 in cleaned if c2):
                logger.info(f"Duplicate detected, skipping ID={entry_id}")
                if not dry_run:
                    cur.execute("DELETE FROM context_history WHERE id = ?", (entry_id,))
                continue
            seen_clean.extend(cleaned)
        if not dry_run:
            conn.commit()
        logger.info("✅ Context history cleanup complete.")
    except Exception as e:
        logger.error(f"[ERROR] Cleanup failed: {e}")
    finally:
        conn.close()

def smart_qa_cleanup(dry_run=False, verbose=False):
    """
    Cleans low-scoring QA entries in the ai_learning.db > qa_cache table,
    keeping only top 5 confidence-based entries and removing stale/zero-score ones.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT id, query, ai_answer, hit_score, feedback FROM qa_cache")
        rows = cur.fetchall()
        scored = []

        for row in rows:
            fid, q, a, score, fb = row
            conf = 0
            if fb and "confidence=" in fb:
                try:
                    conf = float(fb.split("confidence=")[-1].strip())
                except:
                    pass
            scored.append((fid, conf, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_ids = set([fid for fid, _, _ in scored[:5]])
        deleted = 0

        for fid, conf, score in scored[5:]:
            if score == 0 and fid not in keep_ids:
                if not dry_run:
                    cur.execute("DELETE FROM qa_cache WHERE id = ?", (fid,))
                deleted += 1
                if verbose or dry_run:
                    logger.info(f"Purged qa_cache ID={fid} (conf={conf:.2f})")

        if not dry_run:
            conn.commit()
        conn.close()
        logger.info(f"🧹 QA cache cleanup complete. Deleted {deleted} low-scoring entries.")

    except Exception as e:
        logger.error(f"[CLEANUP ERROR] {e}")

def has_recent_cleanup(days=5):
    if not os.path.exists(LAST_RUN_FILE):
        return False
    try:
        with open(LAST_RUN_FILE, 'r') as f:
            last = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        return datetime.now() - last < timedelta(days=days)
    except:
        return False

def update_cleanup_timestamp():
    os.makedirs(os.path.dirname(LAST_RUN_FILE), exist_ok=True)
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def parse_args():
    parser = argparse.ArgumentParser(description="Run SarahMemory daily intelligent cleanup process.")
    parser.add_argument("--force", action="store_true", help="Ignore timestamp threshold")
    parser.add_argument("--dry-run", action="store_true", help="Simulate cleanup without deleting anything")
    parser.add_argument("--verbose", action="store_true", help="Print each action live")
    parser.add_argument("--context-only", action="store_true", help="Only run context cleanup")
    parser.add_argument("--qa-only", action="store_true", help="Only run QA cleanup")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.force and has_recent_cleanup():
        logger.info("⏳ Cleanup skipped. Recently completed.")
        exit(0)
    logger.info("🚿 Running SarahMemory dataset cleanup...")
    if not args.qa_only:
        purge_context_db(dry_run=args.dry_run, verbose=args.verbose)
    if not args.context_only:
        smart_qa_cleanup(dry_run=args.dry_run, verbose=args.verbose)
    if not args.dry_run:
        update_cleanup_timestamp()

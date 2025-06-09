#!/usr/bin/env python3
"""
SarahMemoryCleanup.py <Version 7.0 CLI Enhanced>
Author: Brian Lee Baros

Purpose:
    Cleans invalid, foreign, or nonsense entries from:
        - personality1.db
        - programming.db
        - functions.db
        - software.db

New Features:
    --dry-run       Only preview what would be deleted
    --target-db     Specify one DB file to clean manually
    --verbose       Show every record purged in real-time
"""
import os
import sqlite3
import datetime
import re
import argparse
from langdetect import detect
import spacy
import SarahMemoryGlobals as config
nlp = spacy.load("en_core_web_sm")

DATASETS_DIR = os.path.join("C:\\SarahMemory", "data", "memory", "datasets")
SYSTEM_LOGS_DB = os.path.join(config.DATASETS_DIR, "system_logs.db")

def log_cleanup(msg):
    timestamp = datetime.datetime.now().isoformat()
    print(f"[CLEANUP] {msg}")
    try:
        conn = sqlite3.connect(SYSTEM_LOGS_DB)
        conn.execute("""CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            source TEXT,
            message TEXT
        )""")
        conn.execute("INSERT INTO logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)",
                     (timestamp, "CLEANUP", "SarahMemoryCleanup", msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[LogError] {e}")

def is_garbage(text):
    if not text or not text.strip():
        return True
    if len(text.strip()) < 5:
        return True
    if len(set(text.lower())) <= 2:
        return True
    if re.fullmatch(r'[^\w\s]+', text):
        return True
    try:
        if detect(text) != 'en':
            return True
    except:
        return True
    doc = nlp(text)
    if not any(t.pos_ in {"NOUN", "VERB", "PROPN"} for t in doc):
        return True
    return False

def clean_table(db_path, table, column, dry_run=False, verbose=False):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT id, {column} FROM {table}")
    rows = cur.fetchall()
    purged = 0

    for row_id, content in rows:
        if is_garbage(content):
            if not dry_run:
                cur.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            purged += 1
            msg = f"Purged from {os.path.basename(db_path)}::{table} → ID {row_id} | Content: {content[:60]}"
            if verbose or dry_run:
                print("[PREVIEW] " + msg if dry_run else msg)
            log_cleanup(msg)

    if not dry_run:
        conn.commit()
    conn.close()
    return purged

def run_cleanup(dry_run=False, verbose=False, target_db=None):
    total_purged = 0
    db_targets = [
        ("personality1.db", "responses", "response"),
        ("programming.db", "knowledge_base", "content"),
        ("functions.db", "functions", "description"),
        ("software.db", "software_apps", "app_name"),
    ]

    if target_db:
        db_targets = [item for item in db_targets if item[0] == target_db]
        if not db_targets:
            print(f"[ERROR] Specified target-db '{target_db}' not valid.")
            return

    for db_file, table, col in db_targets:
        path = os.path.join(config.DATASETS_DIR, db_file)
        if os.path.exists(path):
            purged = clean_table(path, table, col, dry_run, verbose)
            log_cleanup(f"{purged} entries purged from {db_file}")
            total_purged += purged
        else:
            log_cleanup(f"{db_file} not found.")

    print(f"[FINISHED] Cleanup complete. Total entries purged: {total_purged}")

def parse_args():
    parser = argparse.ArgumentParser(description="Clean SarahMemory database datasets intelligently.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not delete entries")
    parser.add_argument("--verbose", action="store_true", help="Print each purge record live")
    parser.add_argument("--target-db", type=str, help="Specify one DB to target only (e.g., functions.db)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_cleanup(
        dry_run=args.dry_run,
        verbose=args.verbose,
        target_db=args.target_db
    )

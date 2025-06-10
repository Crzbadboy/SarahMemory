#!/usr/bin/env python3
"""
SarahMemoryReminder.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: Consolidates reminder functionality using APScheduler for scheduling tasks.
Reminders are stored and reloaded from and logged to the reminders.db.
"""

import logging
import datetime
import time
import os
import sqlite3
import SarahMemoryGlobals as config
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from SarahMemoryEncryption import encrypt_data as encrypt_password, decrypt_data as decrypt_password

logger = logging.getLogger('SarahMemoryReminder')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

REMINDER_DB = os.path.join(config.DATASETS_DIR, 'reminders.db')
scheduler = BackgroundScheduler()


def init_db():
    conn = sqlite3.connect(REMINDER_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            message TEXT,
            remind_time TEXT,
            created_at TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            name TEXT PRIMARY KEY,
            email TEXT,
            phone TEXT,
            address TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS passwords (
            label TEXT PRIMARY KEY,
            encrypted_value TEXT,
            created_at TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS webpages (
            label TEXT PRIMARY KEY,
            url TEXT,
            saved_on TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminder_events (
            event TEXT,
            timestamp TEXT,
            description TEXT
        )""")
    conn.commit()
    conn.close()


def log_reminder_event(event, description):
    conn = sqlite3.connect(REMINDER_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO reminder_events (event, timestamp, description) VALUES (?, ?, ?)",
                   (event, datetime.datetime.now().isoformat(), description))
    conn.commit()
    conn.close()


def add_reminder(reminder_id, message, remind_time):
    try:
        conn = sqlite3.connect(REMINDER_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO reminders (id, message, remind_time, created_at) VALUES (?, ?, ?, ?)",
                       (reminder_id, message, remind_time.isoformat(), datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()

        trigger = DateTrigger(run_date=remind_time)
        scheduler.add_job(reminder_action, trigger=trigger, args=[reminder_id, message])

        logger.info(f"[REMINDER] Scheduled reminder '{reminder_id}' for {remind_time.isoformat()}.")
        log_reminder_event("Add Reminder", f"Scheduled reminder '{reminder_id}' for {remind_time.isoformat()}.")
        return True
    except Exception as e:
        logger.error(f"[REMINDER ERROR] Could not add reminder '{reminder_id}': {e}")
        log_reminder_event("Add Reminder Error", str(e))
        return False


def reminder_action(reminder_id, message):
    try:
        logger.info(f"Reminder Triggered [{reminder_id}]: {message}")
        print(f"REMINDER: {message}")
        log_reminder_event("Reminder Triggered", f"Triggered reminder '{reminder_id}': {message}.")
    except Exception as e:
        logger.error(f"Error in reminder action for '{reminder_id}': {e}")
        log_reminder_event("Reminder Action Error", f"Error: {e}")


def store_contact(name, email, phone, address):
    conn = sqlite3.connect(REMINDER_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO contacts (name, email, phone, address) VALUES (?, ?, ?, ?)",
                   (name, email, phone, address))
    conn.commit()
    conn.close()
    log_reminder_event("Store Contact", f"Stored contact: {name}")


def store_webpage(label, url):
    conn = sqlite3.connect(REMINDER_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO webpages (label, url, saved_on) VALUES (?, ?, ?)",
                   (label, url, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    log_reminder_event("Store Webpage", f"Saved website: {label}")


def store_password(label, plaintext_password):
    encrypted = encrypt_password(plaintext_password)
    conn = sqlite3.connect(REMINDER_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO passwords (label, encrypted_value, created_at) VALUES (?, ?, ?)",
                   (label, encrypted, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    log_reminder_event("Store Password", f"Stored password for: {label}")


def start_scheduler():
    try:
        scheduler.start()
        logger.info("Reminder scheduler started successfully.")
        log_reminder_event("Start Scheduler", "Reminder scheduler started successfully.")
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        log_reminder_event("Start Scheduler Error", str(e))


def load_reminders():
    try:
        conn = sqlite3.connect(REMINDER_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT id, message, remind_time FROM reminders")
        rows = cursor.fetchall()
        for rid, msg, rtime in rows:
            dt = datetime.datetime.fromisoformat(rtime)
            if dt > datetime.datetime.now():
                trigger = DateTrigger(run_date=dt)
                scheduler.add_job(reminder_action, trigger=trigger, args=[rid, msg])
        conn.close()
        logger.info(f"Reloaded {len(rows)} reminders from database.")
        log_reminder_event("Load Reminders", f"Reloaded {len(rows)} reminders from reminders.db.")
    except Exception as e:
        logger.error(f"[DB REMINDER LOAD ERROR] {e}")
        log_reminder_event("Load Reminder Error", str(e))


def start_reminder_monitor():
    init_db()
    start_scheduler()
    load_reminders()


if __name__ == '__main__':
    start_reminder_monitor()
    logger.info("Starting SarahMemoryReminder module test.")
    test_reminder_id = "test1"
    test_message = "This is a test reminder."
    remind_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
    add_reminder(test_reminder_id, test_message, remind_time)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Reminder module interrupted by user. Shutting down scheduler.")
        scheduler.shutdown()

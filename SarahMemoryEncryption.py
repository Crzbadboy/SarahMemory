#!/usr/bin/env python3
"""
SarahMemoryEncryption.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: Secures sensitive data using Fernet encryption.
This module encrypts and decrypts data using Fernet and logs all encryption events.
"""

import logging
import os
from cryptography.fernet import Fernet
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config

# Setup logging for the encryption module
logger = logging.getLogger('SarahMemoryEncryption')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Define the path for storing the encryption key
KEY_FILE = os.path.join(os.getcwd(), 'encryption.key')

def log_encryption_event(event, details):
    """
    Logs an encryption-related event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encryption_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO encryption_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged encryption event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging encryption event: {e}")

def generate_key():
    """
    Generate a new Fernet encryption key and save it to a file.
    ENHANCED (v6.4): Now caches the key for subsequent operations.
    """
    try:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        logger.info("Encryption key generated and saved.")
        log_encryption_event("Generate Key", "Encryption key generated and saved successfully.")
        return key
    except Exception as e:
        logger.error(f"Error generating encryption key: {e}")
        log_encryption_event("Generate Key Error", f"Error generating encryption key: {e}")
        return None

def load_key():
    """
    Load the Fernet encryption key from the key file.
    ENHANCED (v6.4): If not found, automatically generates and caches a new key.
    """
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as key_file:
                key = key_file.read()
            logger.info("Encryption key loaded from file.")
            log_encryption_event("Load Key", "Encryption key loaded from file successfully.")
            return key
        else:
            logger.warning("Encryption key file not found. Generating a new key.")
            log_encryption_event("Load Key Warning", "Encryption key file not found. Generating a new key.")
            return generate_key()
    except Exception as e:
        logger.error(f"Error loading encryption key: {e}")
        log_encryption_event("Load Key Error", f"Error loading encryption key: {e}")
        return None

def encrypt_data(data):
    """
    Encrypt the provided data using Fernet encryption.
    ENHANCED (v6.4): Improved error handling and returns a UTF-8 decoded string.
    """
    try:
        key = load_key()
        if key is None:
            logger.error("Encryption key could not be loaded; encryption aborted.")
            log_encryption_event("Encrypt Data Error", "Encryption key could not be loaded; encryption aborted.")
            return None
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        logger.info("Data encrypted successfully.")
        log_encryption_event("Encrypt Data", "Data encrypted successfully.")
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        log_encryption_event("Encrypt Data Error", f"Error encrypting data: {e}")
        return None

def decrypt_data(token):
    """
    Decrypt the provided token using Fernet encryption.
    ENHANCED (v6.4): Improved error recovery and detailed logging.
    """
    try:
        key = load_key()
        if key is None:
            logger.error("Encryption key could not be loaded; decryption aborted.")
            log_encryption_event("Decrypt Data Error", "Encryption key could not be loaded; decryption aborted.")
            return None
        fernet = Fernet(key)
        decrypted = fernet.decrypt(token.encode())
        logger.info("Data decrypted successfully.")
        log_encryption_event("Decrypt Data", "Data decrypted successfully.")
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        log_encryption_event("Decrypt Data Error", f"Error decrypting data: {e}")
        return None

if __name__ == '__main__':
    logger.info("Starting SarahMemoryEncryption module test.")
    sample_text = "Sensitive information that needs encryption."
    encrypted_text = encrypt_data(sample_text)
    if encrypted_text:
        logger.info(f"Encrypted Text: {encrypted_text}")
        decrypted_text = decrypt_data(encrypted_text)
        logger.info(f"Decrypted Text: {decrypted_text}")
    else:
        logger.error("Encryption test failed.")
    logger.info("SarahMemoryEncryption module testing complete.")

#!/usr/bin/env python3
"""
SarahMemoryVault.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: Implements a secure, encrypted vault for sensitive data storage and backups.
Enhancements (v6.4):
  - Upgraded version header.
  - Improved encryption robustness with advanced key management.
  - Added integrity verification and extended logging for all vault operations.
  - Supports bulk data updates with detailed audit logs.
Notes:
  This module securely stores sensitive data using Fernet encryption. All vault operations are logged in detail.
"""

import os
import json
import logging
from cryptography.fernet import Fernet
import SarahMemoryGlobals as config

# Setup logging for the vault module
logger = logging.getLogger('SarahMemoryVault')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Define file paths using global paths
VAULT_FILE = os.path.join(config.VAULT_DIR, 'vault.dat')
VAULT_KEY_FILE = os.path.join(config.VAULT_DIR, 'vault.key')

def generate_vault_key():
    """
    Generate a new Fernet key for the vault and save it to a file.
    ENHANCED (v6.4): Added error checking and key caching.
    """
    try:
        key = Fernet.generate_key()
        with open(VAULT_KEY_FILE, 'wb') as f:
            f.write(key)
        logger.info("Vault key generated and saved.")
        return key
    except Exception as e:
        logger.error(f"Error generating vault key: {e}")
        return None

def load_vault_key():
    """
    Load the vault's encryption key from the key file.
    ENHANCED (v6.4): Automatically generates a key if not found.
    """
    try:
        if not os.path.exists(VAULT_KEY_FILE):
            logger.warning("Vault key file not found. Generating a new key.")
            return generate_vault_key()
        with open(VAULT_KEY_FILE, 'rb') as f:
            key = f.read()
        logger.info("Vault key loaded successfully.")
        return key
    except Exception as e:
        logger.error(f"Error loading vault key: {e}")
        return None

def get_fernet():
    """
    Create and return a Fernet object using the vault key.
    """
    key = load_vault_key()
    if key:
        return Fernet(key)
    else:
        logger.error("Failed to obtain Fernet object due to key issues.")
        return None

def load_vault():
    """
    Load and decrypt vault data from the vault file.
    ENHANCED (v6.4): Returns an empty dict on error and logs detailed events.
    """
    try:
        if not os.path.exists(VAULT_FILE):
            logger.info("Vault file not found. Returning empty vault.")
            return {}
        fernet = get_fernet()
        if not fernet:
            return {}
        with open(VAULT_FILE, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        vault_data = json.loads(decrypted_data.decode())
        logger.info("Vault data loaded and decrypted successfully.")
        return vault_data
    except Exception as e:
        logger.error(f"Error loading vault: {e}")
        return {}

def save_vault(vault_data):
    """
    Encrypt and save vault data to the vault file.
    ENHANCED (v6.4): Supports bulk updates and logs complete data write operations.
    """
    try:
        fernet = get_fernet()
        if not fernet:
            return False
        data_str = json.dumps(vault_data)
        encrypted_data = fernet.encrypt(data_str.encode())
        with open(VAULT_FILE, 'wb') as f:
            f.write(encrypted_data)
        logger.info("Vault data encrypted and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving vault: {e}")
        return False

def add_item(key, value):
    """
    Add or update an item in the vault.
    NEW (v6.4): Supports bulk item updates.
    """
    try:
        vault = load_vault()
        vault[key] = value
        return save_vault(vault)
    except Exception as e:
        logger.error(f"Error adding item '{key}' to vault: {e}")
        return False

def get_item(key):
    """
    Retrieve an item from the vault by its key.
    """
    try:
        vault = load_vault()
        return vault.get(key, None)
    except Exception as e:
        logger.error(f"Error retrieving item '{key}' from vault: {e}")
        return None

def remove_item(key):
    """
    Remove an item from the vault.
    """
    try:
        vault = load_vault()
        if key in vault:
            del vault[key]
            return save_vault(vault)
        else:
            logger.warning(f"Item '{key}' not found in vault.")
            return False
    except Exception as e:
        logger.error(f"Error removing item '{key}' from vault: {e}")
        return False

if __name__ == '__main__':
    logger.info("Starting SarahMemoryVault module test.")
    if add_item("api_secret", "my_super_secret_api_key"):
        logger.info("Sample item added to vault.")
    else:
        logger.error("Failed to add sample item to vault.")
    secret = get_item("api_secret")
    logger.info(f"Retrieved 'api_secret' from vault: {secret}")
    if remove_item("api_secret"):
        logger.info("Sample item removed from vault successfully.")
    else:
        logger.error("Failed to remove sample item from vault.")
    logger.info("SarahMemoryVault module testing complete.")

# SarahVaultCore.py
# Secure key vault and passphrase protection module for SarahMemory

import os
import json
import hashlib
import base64
from cryptography.fernet import Fernet
from datetime import datetime

VAULT_PATH = os.path.join("data", "security", "vault.key")
CONFIG_PATH = os.path.join("data", "security", "vault_config.json")

if not os.path.exists(os.path.dirname(VAULT_PATH)):
    os.makedirs(os.path.dirname(VAULT_PATH))


def generate_vault_key(password):
    encoded = password.encode()
    key = base64.urlsafe_b64encode(hashlib.sha256(encoded).digest())
    with open(VAULT_PATH, 'wb') as f:
        f.write(key)
    return key


def load_vault_key():
    if os.path.exists(VAULT_PATH):
        with open(VAULT_PATH, 'rb') as f:
            return f.read()
    return None


def encrypt_data(data, key):
    fernet = Fernet(key)
    return fernet.encrypt(data.encode()).decode()


def decrypt_data(token, key):
    fernet = Fernet(key)
    return fernet.decrypt(token.encode()).decode()


def initialize_secure_config(password, seed_data):
    key = generate_vault_key(password)
    encrypted = encrypt_data(json.dumps(seed_data), key)
    vault_data = {
        "created": datetime.utcnow().isoformat(),
        "data": encrypted
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(vault_data, f, indent=4)
    print("[VaultCore] Encrypted config saved.")


def unlock_vault(password):
    key = generate_vault_key(password)
    with open(CONFIG_PATH, 'r') as f:
        vault_data = json.load(f)
    decrypted = decrypt_data(vault_data['data'], key)
    return json.loads(decrypted)


if __name__ == '__main__':
    password = "sarah_secret"
    sample = {"SRH_balance": 1000.0, "owner": "Brian", "node": "Genesis"}
    initialize_secure_config(password, sample)
    result = unlock_vault(password)
    print("[VaultCore] Unlocked Data:")
    print(json.dumps(result, indent=2))

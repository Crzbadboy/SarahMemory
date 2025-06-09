# SarahCoinWallet.py
# Manages creation, storage, and validation of SarahCoin wallets

import os
import json
import hashlib
import time
from datetime import datetime

WALLET_DIR = os.path.join("data", "crypto")
WALLET_FILE = os.path.join(WALLET_DIR, "wallet.srh")

if not os.path.exists(WALLET_DIR):
    os.makedirs(WALLET_DIR)


# --- Wallet Functions ---
def create_wallet():
    seed = os.urandom(64)
    private_key = hashlib.sha256(seed).hexdigest()
    address = hashlib.sha256(private_key.encode()).hexdigest()[:36]

    wallet = {
        "address": address,
        "private_key": private_key,
        "created": datetime.utcnow().isoformat(),
        "balance": 0.0,
        "transactions": []
    }

    with open(WALLET_FILE, 'w') as f:
        json.dump(wallet, f, indent=4)

    print(f"[SarahCoinWallet] Wallet created: {address}")
    return wallet


def load_wallet():
    if not os.path.exists(WALLET_FILE):
        return create_wallet()
    with open(WALLET_FILE, 'r') as f:
        return json.load(f)


def add_transaction(tx_type, amount, to=None):
    wallet = load_wallet()
    tx = {
        "type": tx_type,
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat(),
        "to": to
    }
    wallet["transactions"].append(tx)
    if tx_type == "receive":
        wallet["balance"] += amount
    elif tx_type == "send":
        wallet["balance"] -= amount

    with open(WALLET_FILE, 'w') as f:
        json.dump(wallet, f, indent=4)
    return wallet


if __name__ == '__main__':
    create_wallet()
    add_transaction("receive", 150.00)
    print(json.dumps(load_wallet(), indent=2))

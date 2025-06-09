# SarahMemoryCryptoGenesis.py
# Genesis Block Creation, Wallet System, SRH Token Economy Core

import hashlib
import json
import os
import time
from datetime import datetime

WALLET_PATH = os.path.join("wallet", "genesis_wallet.json")
LEDGER_PATH = os.path.join("wallet", "ledger.json")

GENESIS_SUPPLY = 100_000_000  # Total SRH tokens created at genesis
TOKEN_NAME = "SarahCoin"
TOKEN_SYMBOL = "SRH"
DECIMALS = 8  # Can divide into 0.00000001 units

# --- Wallet Generator ---
def create_genesis_wallet():
    if not os.path.exists("wallet"):
        os.makedirs("wallet")

    if os.path.exists(WALLET_PATH):
        return load_wallet()

    seed_phrase = hashlib.sha256(os.urandom(256)).hexdigest()
    public_key = hashlib.sha256(seed_phrase.encode()).hexdigest()

    wallet = {
        "address": public_key[:32],
        "balance": GENESIS_SUPPLY,
        "seed": seed_phrase,
        "timestamp": time.time()
    }
    
    with open(WALLET_PATH, 'w') as f:
        json.dump(wallet, f, indent=4)

    return wallet


def load_wallet():
    with open(WALLET_PATH, 'r') as f:
        return json.load(f)


# --- Transaction Record Keeper ---
def init_ledger():
    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, 'w') as f:
            json.dump({"transactions": []}, f, indent=4)


def record_transaction(from_addr, to_addr, amount):
    with open(LEDGER_PATH, 'r') as f:
        ledger = json.load(f)

    tx = {
        "from": from_addr,
        "to": to_addr,
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    ledger["transactions"].append(tx)

    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=4)


# --- SRH Token System Manager ---
def get_balance(address):
    wallet = load_wallet()
    if wallet["address"] == address:
        return wallet["balance"]
    return 0


def transfer_tokens(to_address, amount):
    wallet = load_wallet()
    balance = wallet["balance"]

    if amount > balance:
        return False

    wallet["balance"] -= amount
    record_transaction(wallet["address"], to_address, amount)

    with open(WALLET_PATH, 'w') as f:
        json.dump(wallet, f, indent=4)
    return True


if __name__ == '__main__':
    init_ledger()
    wallet = create_genesis_wallet()
    print("[SRH CRYPTO] Genesis Wallet Initialized:")
    print(f" Address: {wallet['address']}")
    print(f" Balance: {wallet['balance']} {TOKEN_SYMBOL}")

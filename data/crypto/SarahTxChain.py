# SarahTxChain.py
# Maintains transaction chain linked to SarahCoin Genesis block (simplified blockchain logic)

import os
import json
import hashlib
from datetime import datetime

CHAIN_DIR = os.path.join("data", "chain")
CHAIN_FILE = os.path.join(CHAIN_DIR, "tx_chain.json")

if not os.path.exists(CHAIN_DIR):
    os.makedirs(CHAIN_DIR)


def create_block(data, previous_hash):
    block = {
        "index": get_chain_length(),
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
        "previous_hash": previous_hash
    }
    block["hash"] = hash_block(block)
    return block


def hash_block(block):
    block_string = json.dumps(block, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()


def load_chain():
    if os.path.exists(CHAIN_FILE):
        with open(CHAIN_FILE, 'r') as f:
            return json.load(f)
    return []


def save_chain(chain):
    with open(CHAIN_FILE, 'w') as f:
        json.dump(chain, f, indent=4)


def get_chain_length():
    return len(load_chain())


def add_transaction(data):
    chain = load_chain()
    prev_hash = chain[-1]["hash"] if chain else "0" * 64
    new_block = create_block(data, prev_hash)
    chain.append(new_block)
    save_chain(chain)
    print(f"[TxChain] Added new block #{new_block['index']}")
    return new_block


def print_chain():
    chain = load_chain()
    for block in chain:
        print(json.dumps(block, indent=2))


if __name__ == '__main__':
    add_transaction({"sender": "Brian", "receiver": "SarahNode42", "amount": 42.0})
    print_chain()

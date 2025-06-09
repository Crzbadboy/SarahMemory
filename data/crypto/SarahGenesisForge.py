# SarahGenesisForge.py
# Initializes genesis conditions, economic state, and primary node info for SarahCoin

import os
import json
import hashlib
from datetime import datetime

FORGE_DIR = os.path.join("data", "genesis")
GENESIS_FILE = os.path.join(FORGE_DIR, "genesis_block.json")

if not os.path.exists(FORGE_DIR):
    os.makedirs(FORGE_DIR)


# --- Genesis Block Creation ---
def create_genesis_block():
    block = {
        "index": 0,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "creator": "SarahMemory",
            "version": "v6.6",
            "network": "SRH-AlphaNet",
            "total_supply": 100_000_000,
            "initial_allocation": {
                "genesis_address": 100_000_000
            }
        },
        "previous_hash": "0" * 64
    }
    block["hash"] = hash_block(block)
    with open(GENESIS_FILE, 'w') as f:
        json.dump(block, f, indent=4)
    print("[GenesisForge] Genesis block created.")
    return block


def hash_block(block):
    block_string = json.dumps(block, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()


def load_genesis_block():
    if os.path.exists(GENESIS_FILE):
        with open(GENESIS_FILE, 'r') as f:
            return json.load(f)
    return create_genesis_block()


if __name__ == '__main__':
    block = load_genesis_block()
    print(json.dumps(block, indent=2))

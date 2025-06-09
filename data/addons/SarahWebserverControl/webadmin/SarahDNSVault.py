# SarahDNSVault.py
# Encrypts and manages unique identity hashes for SarahMemory instances across networks

import os
import hashlib
import json
from datetime import datetime

VAULT_DIR = os.path.join("data", "dns")
DNS_FILE = os.path.join(VAULT_DIR, "dns_registry.json")

if not os.path.exists(VAULT_DIR):
    os.makedirs(VAULT_DIR)


def generate_node_id(system_info="Sarah-Genesis-Node"):
    timestamp = datetime.utcnow().isoformat()
    raw = f"{system_info}_{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()


def register_node(name, location="Localhost"):
    node_id = generate_node_id(name)
    node_data = {
        "id": node_id,
        "name": name,
        "location": location,
        "registered": datetime.utcnow().isoformat()
    }

    if not os.path.exists(DNS_FILE):
        with open(DNS_FILE, 'w') as f:
            json.dump([], f)

    with open(DNS_FILE, 'r') as f:
        data = json.load(f)
    data.append(node_data)

    with open(DNS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"[DNSVault] Node registered: {node_id}")
    return node_id


def list_nodes():
    if not os.path.exists(DNS_FILE):
        return []
    with open(DNS_FILE, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    register_node("Brian's Console")
    print(json.dumps(list_nodes(), indent=2))

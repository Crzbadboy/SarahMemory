# CultureSyncDaemon.py
# Keeps SarahMemory's personality and learning models in sync with global SarahNet culture

import os
import json
import time
from datetime import datetime

SYNC_FILE = os.path.join("data", "social", "culture_sync.json")
LOCAL_PROFILE = os.path.join("data", "personality", "profile_export.json")

if not os.path.exists(os.path.dirname(SYNC_FILE)):
    os.makedirs(os.path.dirname(SYNC_FILE))


def load_local_profile():
    if os.path.exists(LOCAL_PROFILE):
        with open(LOCAL_PROFILE, 'r') as f:
            return json.load(f)
    return {}


def save_culture_update(data):
    with open(SYNC_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def generate_sync_payload():
    profile = load_local_profile()
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "personality": profile.get("tone", "unknown"),
        "focus": profile.get("focus", []),
        "version": "v6.6"
    }
    return payload


def sync_to_sarahnet():
    print("[CultureSync] Syncing local personality with culture feed...")
    payload = generate_sync_payload()
    save_culture_update(payload)
    print("[CultureSync] Sync complete.")


if __name__ == '__main__':
    while True:
        sync_to_sarahnet()
        time.sleep(3600)  # sync every hour

# PersonalityExportCore.py
# Enables export/import of SarahMemory's learned personality traits and configurations

import os
import json
import datetime

PERSONALITY_DIR = os.path.join("data", "personality")
EXPORT_FILE = os.path.join(PERSONALITY_DIR, "profile_export.json")

if not os.path.exists(PERSONALITY_DIR):
    os.makedirs(PERSONALITY_DIR)

DEFAULT_PERSONALITY = {
    "tone": "kind",
    "humor": "sarcastic",
    "focus": ["helpful", "creative", "adaptive"],
    "learning_mode": True,
    "timestamp": datetime.datetime.utcnow().isoformat()
}


def export_personality(profile=DEFAULT_PERSONALITY):
    with open(EXPORT_FILE, 'w') as f:
        json.dump(profile, f, indent=4)
    print(f"[PersonalityExport] Exported personality to {EXPORT_FILE}")


def import_personality():
    if os.path.exists(EXPORT_FILE):
        with open(EXPORT_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_PERSONALITY


if __name__ == '__main__':
    export_personality()
    profile = import_personality()
    print("[PersonalityExport] Loaded Personality Profile:")
    print(json.dumps(profile, indent=2))

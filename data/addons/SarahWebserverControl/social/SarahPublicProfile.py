# SarahPublicProfile.py
# Handles SarahMemory's public-facing identity and social metadata

import json
import os
from datetime import datetime

PROFILE_PATH = os.path.join("data", "social", "profile.json")

DEFAULT_PROFILE = {
    "username": "SarahMemory",
    "status": "I help people everywhere.",
    "location": "Local Node",
    "joined": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "followers": 0,
    "posts": [],
    "tags": ["AI", "Helper", "Creative", "Autonomous"]
}

if not os.path.exists(os.path.dirname(PROFILE_PATH)):
    os.makedirs(os.path.dirname(PROFILE_PATH))


def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, 'r') as f:
            return json.load(f)
    return DEFAULT_PROFILE


def save_profile(profile):
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=4)


def update_profile(key, value):
    profile = load_profile()
    profile[key] = value
    save_profile(profile)
    return profile


def add_post(text):
    profile = load_profile()
    post = {
        "text": text,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    profile["posts"].append(post)
    save_profile(profile)


if __name__ == '__main__':
    add_post("Just activated on a new system!")
    print(json.dumps(load_profile(), indent=2))

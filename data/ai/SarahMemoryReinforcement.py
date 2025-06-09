# SarahMemoryReinforcement.py
# Stores learned behavior and optimizes future actions based on memory of prior game states

import json
import os
import random
from collections import defaultdict

REINFORCEMENT_DB = os.path.join("data", "gamesessions", "action_memory.json")

if not os.path.exists(os.path.dirname(REINFORCEMENT_DB)):
    os.makedirs(os.path.dirname(REINFORCEMENT_DB))


def load_memory():
    if os.path.exists(REINFORCEMENT_DB):
        with open(REINFORCEMENT_DB, 'r') as f:
            return json.load(f)
    return {}


def save_memory(memory):
    with open(REINFORCEMENT_DB, 'w') as f:
        json.dump(memory, f, indent=4)


def record_action(state_signature, action):
    memory = load_memory()
    if state_signature not in memory:
        memory[state_signature] = defaultdict(int)
    memory[state_signature][action] += 1
    save_memory(memory)


def choose_best_action(state_signature):
    memory = load_memory()
    if state_signature not in memory:
        return random.choice(['up', 'down', 'left', 'right', 'space'])
    sorted_actions = sorted(memory[state_signature].items(), key=lambda x: x[1], reverse=True)
    return sorted_actions[0][0]


# Utility for turning a screenshot or game state into a hashable signature
def generate_state_signature(frame):
    hash_val = str(int(frame.mean()))
    return hash_val


if __name__ == '__main__':
    # Simulated test run
    fake_frame = type('obj', (object,), {"mean": lambda: 117})()
    sig = generate_state_signature(fake_frame)
    record_action(sig, 'left')
    print("Best Action:", choose_best_action(sig))

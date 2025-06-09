# SarahProofOfIntelligence.py
# Validates SarahMemory's intelligent responses before authorizing SRH crypto reward

import os
import json
import hashlib
import datetime

POI_LOG = os.path.join("data", "crypto", "proof_log.json")
if not os.path.exists(os.path.dirname(POI_LOG)):
    os.makedirs(os.path.dirname(POI_LOG))


# --- Core Validation Logic ---
def score_intelligence(prompt, response):
    if not prompt or not response:
        return 0.0
    length_bonus = len(response) / max(len(prompt), 1)
    uniqueness = len(set(response.split())) / max(len(response.split()), 1)
    score = min(1.0, (length_bonus + uniqueness) / 2)
    return round(score, 4)


def authorize_reward(score):
    return score >= 0.7


def log_proof(prompt, response, score, reward):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        "score": score,
        "reward_approved": reward
    }

    if not os.path.exists(POI_LOG):
        with open(POI_LOG, 'w') as f:
            json.dump([], f)

    with open(POI_LOG, 'r') as f:
        log = json.load(f)
    log.append(entry)

    with open(POI_LOG, 'w') as f:
        json.dump(log[-100:], f, indent=4)

    print(f"[POI] {'Reward Authorized' if reward else 'No Reward'} - Score: {score}")


# --- Main Entry Point ---
if __name__ == '__main__':
    test_prompt = "Describe how SarahMemory learns from mistakes."
    test_response = "SarahMemory uses memory reinforcement and adaptive recall algorithms to improve outcomes."
    score = score_intelligence(test_prompt, test_response)
    reward = authorize_reward(score)
    log_proof(test_prompt, test_response, score, reward)

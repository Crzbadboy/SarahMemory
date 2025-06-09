# SarahPulseDaemon.py
# Continuously monitors and logs system health, AI loop frequency, and service heartbeats

import time
import json
import os
import psutil
from datetime import datetime

PULSE_LOG = os.path.join("data", "logs", "pulse_log.json")

if not os.path.exists(os.path.dirname(PULSE_LOG)):
    os.makedirs(os.path.dirname(PULSE_LOG))


def collect_pulse_data():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "running_processes": len(psutil.pids())
    }


def write_pulse_log(data):
    if not os.path.exists(PULSE_LOG):
        with open(PULSE_LOG, 'w') as f:
            json.dump([], f)

    with open(PULSE_LOG, 'r') as f:
        log = json.load(f)

    log.append(data)

    with open(PULSE_LOG, 'w') as f:
        json.dump(log[-100:], f, indent=4)  # Keep last 100 entries


def run_pulse_daemon():
    print("[PulseDaemon] Starting health monitor loop...")
    while True:
        data = collect_pulse_data()
        write_pulse_log(data)
        print(f"[PulseDaemon] Logged pulse: {data['timestamp']}")
        time.sleep(60)


if __name__ == '__main__':
    run_pulse_daemon()

# SarahCMSUploader.py
# Uploads SarahMemory-generated content to an external CMS (e.g., Wordpress, Ghost, Hugo)

import os
imp# SarahPulseDaemon.py
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
import json
import datetime

UPLOAD_LOG = os.path.join("data", "cms", "upload_log.json")
CONTENT_DIR = os.path.join("data", "canvas")

if not os.path.exists(os.path.dirname(UPLOAD_LOG)):
    os.makedirs(os.path.dirname(UPLOAD_LOG))

if not os.path.exists(CONTENT_DIR):
    os.makedirs(CONTENT_DIR)


def mock_cms_upload(filename, title="SarahMemory Upload"):
    path = os.path.join(CONTENT_DIR, filename)
    if not os.path.exists(path):
        print(f"[CMSUploader] File not found: {filename}")
        return False

    print(f"[CMSUploader] Uploading {filename} to CMS as post '{title}'...")
    log_entry = {
        "filename": filename,
        "title": title,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    if not os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, 'w') as f:
            json.dump([], f)

    with open(UPLOAD_LOG, 'r') as f:
        log = json.load(f)
    log.append(log_entry)

    with open(UPLOAD_LOG, 'w') as f:
        json.dump(log[-100:], f, indent=4)

    print(f"[CMSUploader] Upload logged.")
    return True


if __name__ == '__main__':
    mock_cms_upload("sample_post.md", title="My First AI-Published Article")

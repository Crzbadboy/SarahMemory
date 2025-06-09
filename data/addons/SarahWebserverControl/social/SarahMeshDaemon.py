# SarahMeshDaemon.py
# Synchronizes node status and shares local metrics with the SarahNet mesh

import os
import json
import time
import socket
import uuid
import psutil
from datetime import datetime

MESH_DAEMON_LOG = os.path.join("data", "network", "mesh_heartbeat.json")

if not os.path.exists(os.path.dirname(MESH_DAEMON_LOG)):
    os.makedirs(os.path.dirname(MESH_DAEMON_LOG))


def get_node_identity():
    return {
        "node_id": str(uuid.uuid4())[:12],
        "hostname": socket.gethostname(),
        "ip": get_ip_address(),
        "version": "v6.6",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat()
    }


def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def get_local_stats():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_used": psutil.virtual_memory().percent,
        "disk_free": psutil.disk_usage('/').free,
        "processes": len(psutil.pids())
    }


def log_mesh_heartbeat():
    entry = get_node_identity()
    entry.update(get_local_stats())

    if not os.path.exists(MESH_DAEMON_LOG):
        with open(MESH_DAEMON_LOG, 'w') as f:
            json.dump([], f)

    with open(MESH_DAEMON_LOG, 'r') as f:
        log = json.load(f)
    log.append(entry)

    with open(MESH_DAEMON_LOG, 'w') as f:
        json.dump(log[-100:], f, indent=4)

    print(f"[MeshDaemon] Heartbeat logged at {entry['timestamp']}")


if __name__ == '__main__':
    while True:
        log_mesh_heartbeat()
        time.sleep(300)  # every 5 minutes

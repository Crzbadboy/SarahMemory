# SarahMeshMapper.py
# Maps and logs discovered SarahMemory instances on the SarahNet mesh and their capabilities

import os
import json
import uuid
import socket
from datetime import datetime

MESH_MAP_FILE = os.path.join("data", "network", "mesh_map.json")

if not os.path.exists(os.path.dirname(MESH_MAP_FILE)):
    os.makedirs(os.path.dirname(MESH_MAP_FILE))


def generate_mesh_identity():
    return str(uuid.uuid4())[:12]


def detect_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def register_instance(name, version="v6.6"):
    node = {
        "id": generate_mesh_identity(),
        "hostname": socket.gethostname(),
        "ip": detect_local_ip(),
        "version": version,
        "status": "active",
        "label": name,
        "timestamp": datetime.utcnow().isoformat()
    }

    if not os.path.exists(MESH_MAP_FILE):
        with open(MESH_MAP_FILE, 'w') as f:
            json.dump([], f)

    with open(MESH_MAP_FILE, 'r') as f:
        mesh = json.load(f)
    mesh.append(node)

    with open(MESH_MAP_FILE, 'w') as f:
        json.dump(mesh[-50:], f, indent=4)

    print(f"[MeshMappyer] Node registered: {node['id']} @ {node['ip']}")
    return node


def list_mesh():
    if os.path.exists(MESH_MAP_FILE):
        with open(MESH_MAP_FILE, 'r') as f:
            return json.load(f)
    return []


if __name__ == '__main__':
    register_instance("SarahGenesisAI")
    print(json.dumps(list_mesh(), indent=2))

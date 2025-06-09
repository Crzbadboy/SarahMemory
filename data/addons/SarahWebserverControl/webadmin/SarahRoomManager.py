# SarahRoomManager.py
# Handles video room creation, user presence, and peer session control for SarahMemory video chat

import os
import json
import uuid
import datetime

ROOM_DIR = os.path.join("data", "rooms")
if not os.path.exists(ROOM_DIR):
    os.makedirs(ROOM_DIR)


# --- Room Management ---
def create_room(owner):
    room_id = str(uuid.uuid4())[:8]
    room_data = {
        "id": room_id,
        "owner": owner,
        "created": datetime.datetime.utcnow().isoformat(),
        "participants": [owner]
    }
    path = os.path.join(ROOM_DIR, f"{room_id}.json")
    with open(path, 'w') as f:
        json.dump(room_data, f, indent=4)
    return room_id


def join_room(room_id, username):
    path = os.path.join(ROOM_DIR, f"{room_id}.json")
    if not os.path.exists(path):
        print("[RoomManager] Room not found.")
        return False
    with open(path, 'r') as f:
        room = json.load(f)
    if username not in room['participants']:
        room['participants'].append(username)
    with open(path, 'w') as f:
        json.dump(room, f, indent=4)
    return True


def list_rooms():
    return [f[:-5] for f in os.listdir(ROOM_DIR) if f.endswith(".json")]


def get_room_details(room_id):
    path = os.path.join(ROOM_DIR, f"{room_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    new_id = create_room("Brian")
    print("[RoomManager] Created Room ID:", new_id)
    join_room(new_id, "Sarah")
    print(json.dumps(get_room_details(new_id), indent=2))

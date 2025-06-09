# SarahNetClient.py
# Enables secure connection to the distributed SarahNet AI communication mesh

import socket
import threading
import json
import os

SERVER_IP = "127.0.0.1"  # placeholder; your SarahServer goes here
SERVER_PORT = 5050
HEADER = 64
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

USERNAME = os.getenv("USERNAME", "SarahUser")


def receive_messages(client):
    while True:
        try:
            msg = client.recv(2048).decode(FORMAT)
            if msg:
                print(f"[SarahNet] Message: {msg}")
        except:
            print("[SarahNet] Connection closed.")
            break


def connect_to_sarahnet():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_IP, SERVER_PORT))

    thread = threading.Thread(target=receive_messages, args=(client,))
    thread.start()

    welcome = {"type": "connect", "user": USERNAME}
    send(client, json.dumps(welcome))

    return client


def send(client, msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)


def disconnect(client):
    send(client, DISCONNECT_MESSAGE)
    client.close()


if __name__ == '__main__':
    client = connect_to_sarahnet()
    while True:
        msg = input("You: ")
        if msg == "exit":
            disconnect(client)
            break
        send(client, json.dumps({"type": "chat", "user": USERNAME, "message": msg}))

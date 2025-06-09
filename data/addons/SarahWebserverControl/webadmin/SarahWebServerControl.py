# SarahWebServerControl.py
# Allows SarahMemory to deploy and manage her own lightweight web server for hosting services

import http.server
import socketserver
import threading
import os

PORT = 8181
WEB_DIR = os.path.join("data", "web")

if not os.path.exists(WEB_DIR):
    os.makedirs(WEB_DIR)

INDEX_HTML = os.path.join(WEB_DIR, "index.html")

if not os.path.exists(INDEX_HTML):
    with open(INDEX_HTML, 'w') as f:
        f.write("<html><head><title>SarahMemory AI</title></head><body><h1>Welcome to Sarah's AI Web Server</h1></body></html>")


def run_web_server():
    os.chdir(WEB_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"[SarahWebServer] Serving at port {PORT}")
        httpd.serve_forever()


def start_server_thread():
    thread = threading.Thread(target=run_web_server, daemon=True)
    thread.start()
    print("[SarahWebServer] Web server thread started.")


if __name__ == '__main__':
    start_server_thread()
    input("[SarahWebServer] Press Enter to exit main thread while server runs...")

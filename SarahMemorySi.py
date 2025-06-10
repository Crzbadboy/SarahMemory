#!/usr/bin/env python3
# ============================================
# SarahMemorySi.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
# Description: Dynamic Software Intelligence System (No Refactor, Only Correct Injected Registry Lookup)
# ============================================

import os
import subprocess
import sqlite3
import logging
import psutil
import winreg
import pygetwindow as gw
import pyautogui
import time

logger = logging.getLogger("SarahMemorySi")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

DATABASE_PATH = os.path.join("data", "software.db")
active_launched_apps = []

# Connect or create database if missing
def connect_software_db():
    if not os.path.exists("data"):
        os.makedirs("data")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS software_apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            path TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS software_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    return conn

# Cache found path to software.db
def cache_app_path(name, path):
    conn = connect_software_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO software_apps (name, path) VALUES (?, ?)", (name.lower(), path))
        conn.commit()
        logger.info(f"[Software Cache] Cached: {name} -> {path}")
    except Exception as e:
        logger.error(f"[Software Cache Error] {e}")
    finally:
        conn.close()


def execute_play_command(action: str, target: str, original_query: str) -> str:
    """
    Handles command parsing and action execution for playing media content (music, video, online).
    """
    import re
    
    song = None
    artist = None
    platform = None
    default_app = "Spotify"

    # Match common variants of play command
    match = re.match(r"play\s+(.*?)\s+(?:by\s+(.*?))?(?:\s+on\s+(\w+))?$", original_query, re.IGNORECASE)
    if match:
        song = match.group(1)
        artist = match.group(2)
        platform = match.group(3) or default_app
    else:
        song = original_query.replace("play", "").strip()
        platform = default_app

    # Convert basic platform naming
    platform = platform.lower().strip()
    
    if "youtube" in platform:
        import webbrowser
        search_query = song.replace(" ", "+")
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        return f"Searching YouTube for {song}."

    elif platform == "spotify":
        manage_application_request("open spotify")
        time.sleep(2)

        win = next((w for w in gw.getWindowsWithTitle('Spotify') if not w.isMinimized), None)
        if win:
            win.activate()
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'l')
            search_query = song
            if artist:
                search_query += f" {artist}"
            pyautogui.write(search_query)
            pyautogui.press('enter')
            time.sleep(1.5)
            pyautogui.press('tab')
            pyautogui.press('enter')
            return f"Playing {search_query} on Spotify."
        else:
            return "Spotify window not found. Unable to play the song."

    elif platform in ["media player", "microsoft media player"]:
        return f"[TODO] Attempting to locate and play {song} via local Media Player."

    return f"I wasn't able to understand the play command for: {original_query}"



def search_registry_for_software(software_name):
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")
    ]
    for hive, path in registry_paths:
        try:
            with winreg.OpenKey(hive, path) as key:
                for i in range(0, winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            display_name, _ = winreg.QueryValueEx(subkey, "DisplayName")
                            if software_name.lower() in display_name.lower():
                                try:
                                    install_location, _ = winreg.QueryValueEx(subkey, "InstallLocation")
                                    if install_location:
                                        exe_path = find_executable_in_folder(install_location)
                                        if exe_path:
                                            logger.info(f"[Registry Found] {software_name} at {exe_path}")
                                            return exe_path
                                except FileNotFoundError:
                                    continue
                    except (OSError, FileNotFoundError, PermissionError):
                        continue
        except Exception as e:
            logger.error(f"[Registry Access Error] {e}")
    return None

# Helper: Find EXE inside folder
def find_executable_in_folder(folder_path):
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(".exe"):
                return os.path.join(folder_path, file)
    return None

# Get software path (from DB or Registry)
def get_app_path(app_name):
    app_name = app_name.lower()
    conn = connect_software_db()
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM software_apps WHERE name = ?", (app_name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        path_from_registry = search_registry_for_software(app_name)
        if path_from_registry:
            cache_app_path(app_name, path_from_registry)
            return path_from_registry
        else:
            logger.warning(f"[Software Not Found] {app_name}")
            return None

# Launch application by path
def launch_application(path):
    try:
        proc = subprocess.Popen(path)
        active_launched_apps.append(proc)
        logger.info(f"[Software Launch] Launched: {path}")
        return True
    except Exception as e:
        logger.error(f"[Software Launch Error] {e}")
        return False

# List running applications
def list_running_applications():
    running_apps = []
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        running_apps.append((proc.info['pid'], proc.info['name']))
    return running_apps

# Terminate a running application by app name
def terminate_application(app_name):
    try:
        for proc in active_launched_apps:
            if proc.poll() is None:
                exe_name = os.path.basename(proc.args[0]).lower()
                if app_name.lower() in exe_name:
                    proc.terminate()
                    active_launched_apps.remove(proc)
                    logger.info(f"[Terminate Success] {app_name}")
                    return True
        logger.warning(f"[Terminate Warning] App {app_name} not found in active_launched_apps.")
        return False
    except Exception as e:
        logger.error(f"[Terminate Error] {e}")
        return False

# Minimize application
def minimize_application(app_name):
    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows:
            windows[0].minimize()
            logger.info(f"[Window Minimize] {app_name}")
            return True
        else:
            logger.warning(f"[Minimize Warning] No window found for {app_name}")
            return False
    except Exception as e:
        logger.error(f"[Minimize Error] {e}")
        return False

# Maximize application
def maximize_application(app_name):
    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows:
            windows[0].maximize()
            logger.info(f"[Window Maximize] {app_name}")
            return True
        else:
            logger.warning(f"[Maximize Warning] No window found for {app_name}")
            return False
    except Exception as e:
        logger.error(f"[Maximize Error] {e}")
        return False

# Focus application
def focus_application(app_name):
    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows:
            windows[0].activate()
            logger.info(f"[Window Focus] {app_name}")
            return True
        else:
            logger.warning(f"[Focus Warning] No window found for {app_name}")
            return False
    except Exception as e:
        logger.error(f"[Focus Error] {e}")
        return False

# Manage application launch or kill requests
def manage_application_request(full_command):
    parts = full_command.strip().lower().split()
    if len(parts) < 2:
        return False

    action = parts[0]
    app_name = " ".join(parts[1:])

    try:
        app_path = get_app_path(app_name)

        if action in ["open", "launch", "start"]:
            if app_path:
                return launch_application(app_path)
            else:
                return False

        elif action in ["close", "terminate", "kill"]:
            return terminate_application(app_name)

        elif action == "maximize":
            return maximize_application(app_name)

        elif action == "minimize":
            return minimize_application(app_name)

        elif action in ["focus", "bring"]:
            return focus_application(app_name)

        elif action == "exit":
            if "all" in app_name:
                success = True
                for proc in active_launched_apps.copy():
                    try:
                        proc.terminate()
                        active_launched_apps.remove(proc)
                    except Exception as e:
                        logger.error(f"[Exit All Error] {e}")
                        success = False
                return success
            else:
                return terminate_application(app_name)

        else:
            logger.warning(f"[Unknown Action] {action}")
            return False
    except Exception as e:
        logger.error(f"[Manage Request Error] {e}")
        return False

# Manage requests asynchronously
def manage_application_request_async(request):
    import threading
    thread = threading.Thread(target=manage_application_request, args=(request,))
    thread.start()

# Log software event
def log_software_event(event, details):
    conn = connect_software_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO software_events (timestamp, event, details) VALUES (datetime('now'), ?, ?)", (event, details))
        conn.commit()
        logger.info(f"[Software Event] {event} - {details}")
    except Exception as e:
        logger.error(f"[Software Event Log Error] {e}")
    finally:
        conn.close()

#!/usr/bin/env python3
"""
SarahMemoryHi.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: Retrieves detailed system and hardware information and displays it on a GUI.
Enhancements (v6.4):
  - Upgraded version header.
  - Integrates advanced resource summaries and extended platform details.
  - Supports optional DXDiag report retrieval on Windows.
  - Outputs system information in JSON for API integration.
NEW:
  - Logs complete system info to the database.
Notes:
  This module collects comprehensive system information and displays it in a Tkinter GUI.
"""

import logging
import platform
import psutil
import subprocess
import os
import json
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config
import socket
import asyncio



# Setup logging for the system information module
logger = logging.getLogger('SarahMemoryHi')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

def get_system_info():
    """
    Retrieve detailed system and hardware information.
    ENHANCED (v6.4): Includes extended details and optional DXDiag info.
    NEW: Returns a dictionary for further integration.
    """
    try:
        system_info = {}
        uname = platform.uname()
        system_info['System'] = uname.system
        system_info['Node Name'] = uname.node
        system_info['Release'] = uname.release
        system_info['Version'] = uname.version
        system_info['Machine'] = uname.machine
        system_info['Processor'] = uname.processor

        system_info['CPU Cores (Physical)'] = psutil.cpu_count(logical=False)
        system_info['CPU Cores (Logical)'] = psutil.cpu_count(logical=True)
        system_info['CPU Usage (%)'] = psutil.cpu_percent(interval=1)

        virtual_mem = psutil.virtual_memory()
        system_info['Total Memory (bytes)'] = virtual_mem.total
        system_info['Available Memory (bytes)'] = virtual_mem.available
        system_info['Memory Usage (%)'] = virtual_mem.percent

        disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
        system_info['Total Disk (bytes)'] = disk_usage.total
        system_info['Used Disk (bytes)'] = disk_usage.used
        system_info['Disk Usage (%)'] = disk_usage.percent

        try:
            if platform.system() == "Windows":
                subprocess.check_output(["dxdiag", "/t", "dxdiag_output.txt"], shell=True, stderr=subprocess.STDOUT)
                with open("dxdiag_output.txt", "r", encoding="utf-8", errors="ignore") as file:
                    dx_data = file.read()
                system_info['DXDiag Info'] = dx_data
                os.remove("dxdiag_output.txt")
            else:
                system_info['DXDiag Info'] = "Not available on non-Windows platforms"
        except Exception as dx_e:
            logger.warning(f"DXDiag info not available: {dx_e}")
            system_info['DXDiag Info'] = "DXDiag info not available"
        
        logger.info("System information retrieved successfully.")
        return system_info
    except Exception as e:
        logger.error(f"Error retrieving system information: {e}")
        return {"error": str(e)}

def get_system_info_json():
    """
    Retrieve system information in JSON format.
    """
    info = get_system_info()
    return json.dumps(info, indent=4)

def display_system_info(info):
    """
    Display system information in a GUI window using Tkinter.
    ENHANCED (v6.4): Uses a scrolled text widget.
    """
    try:
        import tkinter as tk
        from tkinter import scrolledtext

        root = tk.Tk()
        root.title("System Information - SarahMemoryHi")
        text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
        text_area.pack(padx=10, pady=10)
        
        info_text = ""
        for key, value in info.items():
            info_text += f"{key}: {value}\n\n"
        text_area.insert(tk.INSERT, info_text)
        logger.info("Displaying system information in GUI.")
        root.mainloop()
    except Exception as e:
        logger.error(f"Error displaying system information: {e}")

def log_system_info_to_db(info):
    """
    Logs the system information to the system_logs.db database.
    ENHANCED (v6.4): Stores system info as JSON.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                info TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        info_json = json.dumps(info)
        cursor.execute("INSERT INTO system_logs (timestamp, info) VALUES (?, ?)", (timestamp, info_json))
        conn.commit()
        conn.close()
        logger.info("System info logged to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging system info to system_logs.db: {e}")
_last_net_io = None

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """Attempt to establish a socket connection to a well-known host to check connectivity."""
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False

def update_network_state():
    global NETWORK_STATE, _last_net_io
    if not is_connected():
        NETWORK_STATE = "red"
        return NETWORK_STATE

    net_io = psutil.net_io_counters()
    if _last_net_io is None:
        _last_net_io = net_io
        NETWORK_STATE = "green"  # assume active initially
        return NETWORK_STATE

    sent_diff = net_io.bytes_sent - _last_net_io.bytes_sent
    recv_diff = net_io.bytes_recv - _last_net_io.bytes_recv
    _last_net_io = net_io

    threshold = 1024  # 1 KB threshold
    if sent_diff >= threshold or recv_diff >= threshold:
        NETWORK_STATE = "green"
    else:
        NETWORK_STATE = "yellow"
    return NETWORK_STATE

async def async_update_network_state():
    loop = asyncio.get_event_loop()
    # Run update_network_state() in a thread pool executor.
    return await loop.run_in_executor(None, update_network_state)

if __name__ == '__main__':
    logger.info("Starting SarahMemoryHi module test.")
    info = get_system_info()
    log_system_info_to_db(info)
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    display_system_info(info)
    logger.info("SarahMemoryHi module testing complete.")

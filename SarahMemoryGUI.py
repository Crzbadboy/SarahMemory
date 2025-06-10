#!/usr/bin/env python3
"""
SarahMemoryGUI.py <Unified GUI Interface Enhanced> <Version #7.0 Enhanced> <Author: Brian Lee Baros>
Description:
  This unified GUI combines video chat, text chat, file transfer, live voice input, and an animated avatar.
  The avatar is animated using PyTorch with GPU acceleration (if available) and displayed in the Avatar Panel.
  
MODIFICATIONS: Split in the def generate_response() into 2 files SarahMemoryReply.py and SarahMemoryCompare.py for better Answers.
  - Loads previous settings (voice, API mode, theme, etc.) from a JSON file.
  - The bottom control panel (with theme drop‑down and apply button) is placed above the status bar.
  - VideoPanel displays the local camera feed (320×240) in its own cell.
  - A global (singleton) instance of ExtendedAvatarController is created within AvatarPanel to avoid repeated instantiation.
  - The open_settings_window function no longer requires a parent argument.
  - Chat input is now a multi‑line Text widget (3 rows).
  - Window geometry is fixed at 1368×768.
"""
# Import the necessary modules for GUI and image processing
import re
import tkinter as tk
from tkinter import BOTH, LEFT, RIGHT, BOTTOM, X, Y, TOP, NSEW
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading
import logging
import os
import time
import random
import pyautogui
import threading
import io
import json
import shutil
import numpy as np
import asyncio
import datetime
# import openai
import socket
import glob
from PyQt5 import QtWidgets, QtGui, QtCore, QtOpenGL
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtOpenGL import QGLWidget
# import bpy # if using Blender scripting context
# import unreal # if calling Unreal from Python - only works in Unreal's embedded Python
# Try to import torch for DLSS simulation; if unavailable, set to None.
try:
    import torch
except ImportError:
    torch = None

import SarahMemoryGlobals as config # Global configuration module
import SarahMemoryVoice as voice # Voice synthesis module
import SarahMemoryAvatar as avatar # Avatar module
# Testing import SarahMemoryAPI as oai # OpenAI API module
from SarahMemoryGlobals import run_async # Async function to run tasks
from SarahMemoryHi import async_update_network_state  # Async network function
import SarahMemorySOBJE as sobje # Object detection module
from SarahMemoryADDONLCHR import AddonLauncher # Addon launcher module
from SarahMemoryResearch import get_research_data #get data from intent statements given by the user in the text box or by voice
from PIL import Image, ImageTk
import trimesh

UNREAL_AVAILABLE = True
UNREAL_HOST = '127.0.0.1'
UNREAL_PORT = 7777
RENDER_LOOP_PATH = r"C:\SarahMemory\resources\Unreal Projects\SarahMemory\Saved\MovieRenders"
RENDER_PATTERN = "3D_MotionDesign.*.jpeg"
FRAME_RATE = 24

try:
    import psmove
    PSMOVE_ENABLED = True
except ImportError:
    PSMOVE_ENABLED = False


# Import UnifiedAvatarController; if unavailable, define a dummy.
try:
    from UnifiedAvatarController import UnifiedAvatarController


except ImportError:
    class UnifiedAvatarController:
        def show_avatar(self):
            return None

# Import intent classification and personality integration functions.
try:
    from SarahMemoryAdvCU import classify_intent
except ImportError:
    def classify_intent(text):
        return "statement"

try:
    from SarahMemoryPersonality import process_interaction, integrate_with_personality
except ImportError:




    def integrate_with_personality(text):
        return "I am here to help you."

# Set up a dedicated asyncio event loop.



# ==================================================
# 🚨 QUERY RESEARCH PATH LOGGER SETUP
# Logs the research/debug path of every query issued by the GUI
# ==================================================
research_log_path = os.path.join(config.BASE_DIR, "data", "logs", "research.log")
research_path_logger = logging.getLogger("ResearchPathLogger")
research_path_logger.setLevel(logging.DEBUG)

research_file_handler = logging.FileHandler(research_log_path, mode='a', encoding='utf-8')
research_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

if not research_path_logger.hasHandlers():
    research_file_handler = logging.FileHandler(research_log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    research_file_handler.setFormatter(formatter)
    research_path_logger.addHandler(research_file_handler)
    research_path_logger.setLevel(logging.INFO)

def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async_loop = asyncio.new_event_loop()
async_thread = threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True)
async_thread.start()

# ------------------------- Global Variables -------------------------
SETTINGS_FILE = os.path.join(config.SETTINGS_DIR, "settings.json")
NETWORK_STATE = "red"  # red, yellow, or green
MIC_STATUS = "On"
CAMERA_STATUS = "On"
# ------------------------- Shared Frame Buffer -------------------------
shared_frame = None
shared_lock = threading.Lock()

# ------------------------- Theme Loader -------------------------
theme_logger = logging.getLogger("ThemeLoader")
theme_logger.setLevel(logging.DEBUG)
if not theme_logger.hasHandlers():
    th = logging.StreamHandler()
    th.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    theme_logger.addHandler(th)

def get_active_sentence_model():
    from sentence_transformers import SentenceTransformer
    if MULTI_MODEL:
        for model_name, enabled in MODEL_CONFIG.items():
            if enabled:
                try:
                    return SentenceTransformer(model_name)
                except Exception as e:
                    print(f"[MODEL LOAD ERROR] {model_name} failed: {e}")
    return SentenceTransformer("all-MiniLM-L6-v2")  # Fallback default

def apply_theme_from_choice(css_filename):
    """
    Applies the chosen theme by parsing the CSS file and applying styles to ttk widgets.
    """
    mods_dir = config.THEMES_DIR
    css_path = os.path.join(mods_dir, css_filename)
    if not os.path.exists(css_path):
        theme_logger.error("CSS file not found: " + css_path)
        return
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        theme_logger.error("Failed to open CSS file: " + str(e))
        return

    style = ttk.Style()
    pattern = r'([.\w-]+)\s*\{([^}]+)\}'
    matches = re.findall(pattern, css_content)
    css_to_ttk = {
        "background-color": "background",
        "background": "background",
        "color": "foreground",
        "foreground": "foreground",
        "font": "font",
        "borderwidth": "borderwidth",
        "relief": "relief"
    }
    for selector, properties in matches:
        style_name = selector.lstrip('.')
        props = {}
        for declaration in properties.split(';'):
            declaration = declaration.strip()
            if not declaration:
                continue
            if ':' in declaration:
                key, value = declaration.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                ttk_key = css_to_ttk.get(key, key)
                props[ttk_key] = value
        if props:
            try:
                style.configure(style_name, **props)
                theme_logger.info(f"Applied theme '{css_filename}' to style '{style_name}' with properties: {props}")
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                theme_logger.error(f"Error applying theme for style '{style_name}': {e}")

# ------------------------- Logger Setup -------------------------
logger = logging.getLogger("SarahMemoryGUI")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

# ------------------------- Utility Functions -------------------------
def log_gui_event(event: str, details: str) -> None:
    try:
        db_path = os.path.join(config.BASE_DIR, "data", "memory", "datasets", "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        import sqlite3
        from datetime import datetime
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gui_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event TEXT,
                    details TEXT
                )
            """)
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO gui_events (timestamp, event, details) VALUES (?, ?, ?)",
                           (timestamp, event, details))
            conn.commit()
        logger.info(f"Logged GUI event: {event} - {details}")
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        logger.error(f"Error logging GUI event: {e}")

# ------------------------- Extended Avatar Controller -------------------------
class ExtendedAvatarController(UnifiedAvatarController):
    def show_avatar(self):
        # ✅ Try primary Blender-rendered avatar
        rendered_avatar_path = os.path.join(config.AVATAR_DIR, "avatar_rendered.jpg")
        try:
            if os.path.exists(rendered_avatar_path):
                img = Image.open(rendered_avatar_path).convert("RGB")
                img = img.resize((640, 800))
                logger.info("Loaded rendered 3D avatar image.")
            else:
                raise FileNotFoundError("avatar_rendered.jpg not found.")
        except Exception as e:
            logger.warning(f"⚠️ Rendered avatar load failed: {e}")
            # 🔄 Fallback to a random static image from avatar folder
            avatar_files = [f for f in os.listdir(config.AVATAR_DIR) if f.lower().endswith(".jpg")]
            if not avatar_files:
                logger.error("No fallback avatars found in avatar folder.")
                return None
            fallback_path = os.path.join(config.AVATAR_DIR, random.choice(avatar_files))
            img = Image.open(fallback_path).convert("RGB")
            img = img.resize((640, 800))
            logger.info("Loaded fallback static avatar.")

        try:
            if torch is not None:
                arr = np.array(img)
                tensor = torch.tensor(arr).float()
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                    noise = torch.randn_like(tensor) * config.NOISE_SCALE
                    tensor = torch.clamp(tensor + noise, 0, 255).cpu()
                else:
                    noise = torch.randn_like(tensor) * config.NOISE_SCALE
                    tensor = torch.clamp(tensor + noise, 0, 255)
                animated_img = Image.fromarray(tensor.byte().numpy())
                photo = ImageTk.PhotoImage(animated_img, master=tk._default_root)
                logger.info("PyTorch-augmented avatar displayed.")
            else:
                photo = ImageTk.PhotoImage(img, master=tk._default_root)
                logger.info("Static avatar displayed (no PyTorch).")
            return photo
        except Exception as e:
            logger.error(f"Avatar display/render error: {e}")
            return None

# ------------------------- Connection Panel (Top Bar) -------------------------
class ConnectionPanel:
    def __init__(self, parent, settings_callback):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        HOST_ROOM_TEXT = "Host Room"
        JOIN_ROOM_TEXT = "Join Room"
        SEND_FILE_TEXT = "Send File"
        MIC_STATUS_TEXT = "Mic Mute"
        CAM_STATUS_TEXT = "Camera On/Off"
        SCAN_STATUS_TEXT = "Scan Item"
        ADDON_SELECT_TEXT = "Open Add-ons"
        MEMORY_REFRESH_TEXT = "Memory AutoCorrect"
        EXIT_TEXT = "Exit"

        self.host_button = ttk.Button(self.frame, text=HOST_ROOM_TEXT, command=self.host_room)
        self.join_button = ttk.Button(self.frame, text=JOIN_ROOM_TEXT, command=self.join_room)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(self.frame, text=SEND_FILE_TEXT, command=self.send_file)
        self.file_button.pack(side=tk.LEFT, padx=5)        
        # Call open_settings_window without expecting an extra argument.
        self.host_button = ttk.Button(self.frame, text=MIC_STATUS_TEXT, command=self.toggle_mic)
        self.host_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=CAM_STATUS_TEXT, command=self.toggle_camera)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=SCAN_STATUS_TEXT, command=self.scan_item)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=ADDON_SELECT_TEXT, command=self.open_addons)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=MEMORY_REFRESH_TEXT, command=self.memory_autocorrect)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(self.frame, text=EXIT_TEXT, command=self.exit_app)
        self.file_button.pack(side=tk.LEFT, padx=5)
    def host_room(self):
        pwd = simpledialog.askstring("Host Room", "Enter a password for your room:")
        if pwd:
            logger.info("Hosting room with provided password.")
            messagebox.showinfo("Host Room", "Room hosted successfully.")
        else:
            logger.warning("Host room cancelled.")
    
    def join_room(self):
        pwd = simpledialog.askstring("Join Room", "Enter the room password:")
        if pwd:
            logger.info("Joining room with provided password.")
            messagebox.showinfo("Join Room", "Joined room successfully.")
        else:
            logger.warning("Join room cancelled.")
    
    def send_file(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to send")
        if file_paths:
            for f in file_paths:
                _, ext = os.path.splitext(f)
                ext = ext.lower()
                mapping = {
                    ".csv": config.DATASETS_DIR,
                    ".json": config.DATASETS_DIR,
                    ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
                    ".png": os.path.join(config.PROJECTS_DIR, "images"),
                    ".py": config.IMPORTS_DIR
                }
                target = mapping.get(ext, config.ADDONS_DIR)
                os.makedirs(target, exist_ok=True)
                try:
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    messagebox.showerror("Error", f"Failed to send file {f}: {e}")
            messagebox.showinfo("Send File", "Files sent successfully!")
    def scan_item(self):
    # Placeholder: call scanning function from avatar module (if implemented)
        try:
            from SarahMemoryAvatar import perform_scan_capture
            perform_scan_capture()
            log_gui_event("Scan Item", "Scan triggered.")
        except Exception as e:
            logger.error(f"Scan error: {e}")
            messagebox.showerror("Error", f"Scan failed: {e}")

    def toggle_mic(self):
        global MIC_STATUS
        MIC_STATUS = "Off" if MIC_STATUS == "On" else "On"
        log_gui_event("Mic Toggle", f"Mic status set to {MIC_STATUS}")

    def toggle_camera(self):
        global CAMERA_STATUS
        CAMERA_STATUS = "Off" if CAMERA_STATUS == "On" else "On"
        log_gui_event("Camera Toggle", f"Camera status set to {CAMERA_STATUS}")
    #--------------------------------OPEN ADDONS MODULE-----------------------------
    def open_addons(self):
        self.addon_launcher = AddonLauncher(parent=self.frame)
        self.addon_launcher.open_addons()
    #-------------------------------MEMORY AUTOCORRECT FUNCTION---------------------
    def memory_autocorrect(self):
        from SarahMemorySystemLearn import memory_autocorrect
        memory_autocorrect()
        messagebox.showinfo("Memory AutoCorrect", "Intent overrides updated from system logs.")

    #-------------------------------END ADDONS MODULE-------------------------------
    def exit_app(self):
        import sys
        log_gui_event("Shutdown", "User clicked Exit button.")
        sys.exit(0)
        

# ------------------------- Video Panel (Left Column) -------------------------
class VideoPanel:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent, relief=tk.GROOVE, borderwidth=1)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        self.local_label = tk.Label(self.frame, text="Local Camera", bg="black")
        self.local_label.grid(row=0, column=0, sticky="nsew")
        self.remote_label = tk.Label(self.frame, text="Remote Camera", bg="black")
        self.remote_label.grid(row=1, column=0, sticky="nsew")
        self.screen_preview = tk.Label(self.frame, text="Desktop Mirror", bg="black")
        self.screen_preview.grid(row=2, column=0, sticky="nsew")

        self.local_camera = cv2.VideoCapture(0)
        if not self.local_camera.isOpened():
            logger.error("Failed to open local camera.")
            self.local_camera = None

        self.remote_camera = None
        self.local_photo = None
        self.remote_photo = None
        self.update_video()

        # Start background object and facial detection loop
        if config.ASYNC_PROCESSING_ENABLED:
            from SarahMemorySOBJE import ultra_detect_objects
            from SarahMemoryFacialRecognition import detect_faces_dnn

            def vision_processing_loop():
                global shared_frame
                while True:
                    with shared_lock:
                        frame = shared_frame.copy() if shared_frame is not None else None
                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        tags = ultra_detect_objects(frame)
                        faces = detect_faces_dnn(frame)
                        if hasattr(config, 'status_bar'):
                            try:
                                text_summary = f"Tags: {', '.join(tags[:3])} | Faces: {len(faces)}"
                                config.status_bar.set_status(text_summary)
                            except Exception as e:
                                logger.warning(f"Status bar update failed: {e}")
                    time.sleep(3)
            if config.OBJECT_DETECTION_ENABLED:
                threading.Thread(target=vision_processing_loop, daemon=True).start()


    def update_video(self):
        global shared_frame
        if self.local_camera and self.local_camera.isOpened():
            ret, frame = self.local_camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                resized = cv2.resize(frame, (320, 240))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self.local_photo = ImageTk.PhotoImage(img)
                self.local_label.configure(image=self.local_photo)
                self.local_label.image = self.local_photo
                with shared_lock:
                    shared_frame = frame.copy()
        
        # Display placeholder for remote camera
        blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        blank_img = Image.fromarray(blank_frame)
        self.remote_photo = ImageTk.PhotoImage(blank_img)
        self.remote_label.configure(image=self.remote_photo, text="")
        self.remote_label.image = self.remote_photo

        self.frame.after(30, self.update_video)


    def update_desktop_mirror(self):
        try:
        # For demonstration, use pyautogui.screenshot() to get desktop image.
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot = screenshot.resize((320, 240))
            photo = ImageTk.PhotoImage(screenshot, master=tk._default_root)
            self.desktop_label.configure(image=photo)
            self.desktop_label.image = photo
        except Exception as e:
            logger.error(f"Desktop mirror update failed: {e}")
        self.frame.after(5000, self.update_desktop_mirror)

    def host_room(self):
        pwd = simpledialog.askstring("Host Room", "Enter a password for your room:")
        if pwd:
            logger.info("Hosting room with provided password.")
            messagebox.showinfo("Host Room", "Room hosted successfully.")
        else:
            logger.warning("Host room cancelled.")
    
    def join_room(self):
        pwd = simpledialog.askstring("Join Room", "Enter the room password:")
        if pwd:
            logger.info("Joining room with provided password.")
            messagebox.showinfo("Join Room", "Joined room successfully.")
        else:
            logger.warning("Join room cancelled.")
    
    def send_file(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to send")
        if file_paths:
            for f in file_paths:
                _, ext = os.path.splitext(f)
                ext = ext.lower()
                mapping = {
                    ".csv": config.DATASETS_DIR,
                    ".json": config.DATASETS_DIR,
                    ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
                    ".png": os.path.join(config.PROJECTS_DIR, "images"),
                    ".py": config.IMPORTS_DIR
                }
                target = mapping.get(ext, config.ADDONS_DIR)
                os.makedirs(target, exist_ok=True)
                try:
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    messagebox.showerror("Error", f"Failed to send file {f}: {e}")
            messagebox.showinfo("Send File", "Files sent successfully!")
    
    def release_resources(self):
        if self.local_camera and hasattr(self.local_camera, 'isOpened') and self.local_camera.isOpened():
            self.local_camera.release()
        if self.remote_camera and hasattr(self.remote_camera, 'isOpened') and self.remote_camera.isOpened():
            self.remote_camera.release()
        # screen_preview is a Label, so we skip checking isOpened
# ------------------------- Chat Panel (Middle Column) -------------------------
# ------------------------- Chat Panel (Middle Column) -------------------------
class ChatPanel:
    def __init__(self, parent, gui_instance, avatar_controller):
        # Store GUI and avatar references
        self.gui = gui_instance
        self.avatar_controller = avatar_controller

        # Main container frame setup
        self.frame = ttk.Frame(parent, relief=tk.GROOVE, borderwidth=2)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        # Output display widget for chat history
        self.chat_output = tk.Text(self.frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_output.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Scrollbar for the output box
        self.scrollbar = ttk.Scrollbar(self.frame, command=self.chat_output.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # User input box
        self.chat_input = tk.Text(self.frame, height=3)
        self.chat_input.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.chat_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ttk.Button(self.frame, text="Send", command=self.send_message)
        self.send_button.grid(row=2, column=1, padx=5, pady=5)

        # Store images embedded in chat
        self.chat_images = []

    def insert_avatar_image(self, photo):
        self.chat_output.configure(state=tk.NORMAL)
        self.chat_images.append(photo)
        self.chat_output.image_create(tk.END, image=photo)
        self.chat_output.insert(tk.END, "\n")
        self.chat_output.see(tk.END)
        self.chat_output.configure(state=tk.DISABLED)

    def scan_item(self):
        from SarahMemoryAvatar import perform_scan_capture
        perform_scan_capture()
        log_gui_event("Manual Scan", "User clicked scan button.")

    def capture_desktop_loop(self):
        import time
        while True:
            try:
                screenshot = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                small = cv2.resize(img, (100, 60))
                img_pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(img_pil)
                self.screen_preview.configure(image=photo)
                self.screen_preview.image = photo
                from SarahMemoryAvatar import ultra_detect_objects
                tags = ultra_detect_objects(img)
                if hasattr(config, 'status_bar'):
                    config.status_bar.set_status(f"Desktop View: {', '.join(tags[:3])}")
            except Exception as e:
                logger.warning(f"Screen capture error: {e}")
            time.sleep(15)

    def send_message(self, event=None):
        from SarahMemoryGlobals import INTERRUPT_KEYWORDS, INTERRUPT_FLAG
        from SarahMemoryPersonality import get_emotion_response

        # ✅ Correct access to data mode settings
        override_mode = getattr(self.gui, 'override_data_mode', 'api').lower()
        config.LOCAL_DATA_ENABLED = override_mode == "local"
        config.API_RESEARCH_ENABLED = override_mode in ["api", "web"]
        config.WEB_RESEARCH_ENABLED = override_mode == "web"

        if event:
            event.widget.mark_set("insert", "end")
            event.widget.tag_remove("sel", "1.0", "end")

        text = self.chat_input.get("1.0", tk.END).strip()
        research_path_logger.debug(f"[GUI] Received user input: '{text}' at send_message()")
        if not text:
            return "break"

        self.chat_input.delete("1.0", tk.END)
        self.append_message("You: " + text)
        log_gui_event("Chat Sent", text)
        self.gui.status_bar.set_intent_light("yellow")

        if any(word in text.lower() for word in INTERRUPT_KEYWORDS):
            import SarahMemoryGlobals
            SarahMemoryGlobals.INTERRUPT_FLAG = True
            emotion = "frustration"
            response = get_emotion_response(emotion)
            self.append_message("Sarah: " + response + " I’ve stopped the request as you asked.")
            run_async(voice.synthesize_voice, response + " I’ve stopped the request as you asked.")
            return "break"

        if "exit" in text.lower():
            self.exit_chat()
            return "break"

        if "show me your avatar" in text.lower():
            self.gui.update_avatar_window()
            return "break"

        if "create your avatar" in text.lower() or "change your" in text.lower():
            run_async(self.avatar_controller.create_avatar, text)
            return "break"

        run_async(self.generate_response, text)
        return "break"

    def generate_response(self, user_text):
        from SarahMemoryReply import generate_reply
        from SarahMemoryCompare import compare_reply
        from SarahMemoryGlobals import COMPARE_VOTE
        from SarahMemoryDatabase import record_qa_feedback
        research_path_logger.debug(f"[GUI] Forwarding to generate_reply() from GUI input: '{user_text}'")
        result_bundle = generate_reply(self, user_text) # LOGGING STARTS HERE
            


        if isinstance(result_bundle, dict):
            response = result_bundle.get("response", "")
            source = result_bundle.get("source", "unknown")
            intent = result_bundle.get("intent", "undetermined")
            timestamp = result_bundle.get("timestamp", "")

            #self.append_message(f"[Source: {source}] (Intent: {intent})")
            self.append_message
            self.gui.last_user_interaction = time.time()

            if config.API_RESPONSE_CHECK_TRAINER:
                compare_result = compare_reply(user_text, response)
                if compare_result and isinstance(compare_result, dict):
                    conf = compare_result.get("similarity_score", 'N/A')
                    status = compare_result.get("status", 'N/A')
                    self.append_message(f"[Comparison] Status: {status} | Confidence: {conf} | [Source: {source}] (Intent: {intent} ")

                    if COMPARE_VOTE:
                        from tkinter import messagebox
                        vote = messagebox.askyesno("Feedback", "Was this a helpful response?")
                        vote_label = "Yes" if vote else "No"
                        record_qa_feedback(user_text, score=1 if vote else 0, feedback=f"UserVote: {vote_label}")
                        self.append_message(f"[User Vote] You said: {vote_label}")
        else:
            self.append_message("[ERROR] Failed to get a valid response from AI pipeline.")

    def compare_responses(self, user_text, generated_response):
        from SarahMemoryCompare import compare_reply
        result = compare_reply(user_text, generated_response)
        if result and isinstance(result, dict):
            self.append_message(f"[Comparison Result] Status: {result['status']} | Confidence: {result.get('similarity_score', 'N/A')}")
        else:
            self.append_message("[Comparison Result] No feedback returned.")

    def exit_chat(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            log_gui_event("Exit Command", "User typed 'exit' in chat.")
            self.gui.shutdown()

    def append_message(self, message):
        self.chat_output.configure(state='normal')
        self.chat_output.insert(tk.END, message + "\n")
        self.chat_output.configure(state='disabled')
        self.chat_output.see(tk.END)


# ------------------------- Avatar Panel (Right Column) -------------------------

class AvatarPanel(QGLWidget):
    
    def __init__(self, parent=None):
        if not config.ENABLE_AVATAR_PANEL:
                return  # Exit if avatar panel is disabled
        super(AvatarPanel, self).__init__(parent)
        self.setMinimumSize(640, 800)
        self.frames = []
        self.current_frame = 0
        self.video_cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.mode = "static"
        self.static_frame = None
        self.controller_position = (320, 400)
        self.last_path = r"C:\SarahMemory\resources\Unreal Projects\SarahMemory\Saved\MovieRenders"

        self.selector_widget = QWidget(self)
        self.selector_widget.setGeometry(0, 0, 640, 40)
        self.selector_layout = QHBoxLayout()
        self.selector_button = QPushButton("Select Avatar Media")
        self.selector_layout.addWidget(self.selector_button)
        self.selector_widget.setLayout(self.selector_layout)
        self.selector_button.clicked.connect(self.select_folder)

        if PSMOVE_ENABLED:
            self.ps_move = psmove.PSMove()
            self.controller_thread = threading.Thread(target=self.poll_controller)
            self.controller_thread.daemon = True
            self.controller_thread.start()

        self.load_media(self.last_path)
        self.timer.start(int(1000 / 24))

    def poll_controller(self):
        while True:
            if self.ps_move.poll():
                x, y, _ = self.ps_move.get_accelerometer_frame(psmove.Frame.Last)
                self.controller_position = (
                    int(320 + x * 100),
                    int(400 + y * 100)
                )
            time.sleep(0.01)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder or Media")
        if folder:
            self.last_path = folder
            self.cleanup_previous()
            self.load_media(folder)

    def cleanup_previous(self):
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()
        self.frames.clear()
        self.static_frame = None
        self.mode = "static"

    def load_media(self, path):
        try:
            if os.path.isdir(path):
                image_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
                if len(image_files) > 1:
                    self.mode = "frame_sequence"
                    self.frames = [cv2.imread(img) for img in image_files]
                    logger.info(f"Loaded {len(self.frames)} frames for looping sequence.")
                elif len(image_files) == 1:
                    self.mode = "static"
                    self.static_frame = cv2.imread(image_files[0])
                    logger.info(f"Loaded single frame: {image_files[0]}")
                else:
                    video_files = glob.glob(os.path.join(path, "*.mp4"))
                    if video_files:
                        self.mode = "video"
                        self.video_cap = cv2.VideoCapture(video_files[0])
                        logger.info(f"Loaded MP4 video: {video_files[0]}")
                    else:
                        model_files = glob.glob(os.path.join(path, "*.glb")) + glob.glob(os.path.join(path, "*.stl"))
                        if model_files:
                            self.mode = "model_3d"
                            self.static_frame = self.render_model_preview(model_files[0])
                            logger.info(f"Loaded 3D model: {model_files[0]}")
            else:
                logger.warning("Provided path is not valid.")
        except Exception as e:
            logger.error(f"Media load error: {e}")

    def update_frame(self):
        if self.mode == "frame_sequence":
            if self.frames:
                self.current_frame = (self.current_frame + 1) % len(self.frames)
                frame = self.frames[self.current_frame]
                self.render_image(frame)
        elif self.mode == "static":
            if self.static_frame is not None:
                self.render_image(self.static_frame)
        elif self.mode == "video":
            if self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    self.render_image(frame)
                else:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.mode == "model_3d":
            if self.static_frame is not None:
                self.render_image(self.static_frame)

    def render_model_preview(self, model_path):
        try:
            mesh = trimesh.load(model_path)
            preview = mesh.scene().save_image(resolution=(640, 800))
            nparr = np.frombuffer(preview, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"3D model preview failed: {e}")
            return None

    def render_image(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 40, qimg)
            if PSMOVE_ENABLED:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                painter.drawEllipse(self.controller_position[0], self.controller_position[1], 30, 30)
            painter.end()
        except Exception as e:
            logger.error(f"Image rendering failed: {e}")

    def update_virtual_object(self, object_description):
        try:
            logger.info(f"3D Engine: Creating object: {object_description}")
            # Unreal integration: Blueprint call
            if UNREAL_AVAILABLE: 
                try:
                    with socket.create_connection((UNREAL_HOST, UNREAL_PORT), timeout=5) as sock:
                        command = json.dumps({"command": "spawn_object", "name": object_description})
                        sock.sendall(command.encode('utf-8'))
                        response = sock.recv(1024).decode('utf-8')
                        logger.info(f"Unreal Engine responded: {response}")
                        return
                except Exception as ue_err:
                    logger.error(f"Socket to Unreal failed: {ue_err}")
                try:
                    unreal.log(f"Request to generate: {object_description}")
                    world = ue.get_editor_world()
                    actor = world.actor_spawn(ue.find_class('StaticMeshActor'), ue.FVector(0, 0, 200))
                    system_lib = unreal.SystemLibrary()
                    actor_util = unreal.EditorLevelLibrary()
                    new_actor = actor_util.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), location=(0, 0, 100))
                    logger.info("Unreal object spawned.")
                    return
                except Exception as e:
                    pass
        except Exception as ue:
            logger.warning(f"Unreal fallback to Blender: {ue}")
            try:
                bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1, depth=2, location=(0, 0, 0))
                bpy.context.active_object.name = object_description.replace(" ", "_")
                bpy.ops.render.opengl(animation=False)
                logger.info("Blender object generated and rendered.")
            except Exception as be:
                logger.error(f"Blender object error: {be}")# MAIN LOGIC: Create item from voice/text command and animate it in world
        try:
            logger.info(f"3D Engine: Creating object: {object_description}")
            from UnifiedAvatarController import UnifiedAvatarController
            controller = UnifiedAvatarController()
            controller.create_avatar(object_description)  # Handles Blender/Unreal command pipe
            logger.info("Object creation dispatched to engine controller.")
        except Exception as e:
            logger.error(f"Error generating 3D object: {e}")

    def control_camera(self, direction):
        x, y, z = self.camera_position
        if direction == "left": x -= 1
        elif direction == "right": x += 1
        elif direction == "forward": z -= 1
        elif direction == "back": z += 1
        self.camera_position = (x, y, z)
        logger.info(f"Camera moved to: {self.camera_position}")
        try:
            bpy.context.scene.camera.location = (x, y, z)
            logger.info("Blender camera moved.")
        except Exception as e:
            logger.warning(f"Camera control failed: {e}")

    def enable_vr_mirror(self):
        try:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.context.window_manager.xr_session_settings.base_pose_type = 'CUSTOM'
            bpy.ops.wm.xr_session_start()
            logger.info("VR session started with viewport mirror.")
        except Exception as vr:
            logger.error(f"Failed to start VR session: {vr}")

        
        
        
        self.update_avatar()  # Auto-update on startup
    
    def load_avatar_image(self):
        try:
            avatar_path = os.path.join(globals_module.AVATAR_DIR, "avatar_rendered.jpg")
            if not os.path.exists(avatar_path):
                avatar_path = os.path.join(globals_module.AVATAR_DIR, "default_avatar.jpg")

            photo = self.avatar_controller.show_avatar()
            if photo:
                self.photo = photo
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
                logger.info("Avatar loaded using avatar_controller.")
            else:
                image = Image.open(avatar_path)
                image = image.resize((300, 300), Image.ANTIALIAS)
                self.avatar_image = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.avatar_image)
                self.image_label.image = self.avatar_image
                logger.warning("Fallback: Avatar image loaded directly.")
        except Exception as e:
            logger.error(f"Failed to load avatar image: {e}")
    
    def update_avatar(self):
        def _update():
            try:
                # Attempt to load avatar image (if this method does not raise, we exit early)
                self.load_avatar_image()
            
            except Exception as primary_exception:
                # If load_avatar_image fails, fallback to showing a random avatar
                try:
                    photo = self.avatar_controller.show_avatar()
                    if photo:
                        self.photo = photo
                        self.avatar_label.configure(image=self.photo, text="")
                        self.avatar_label.image = self.photo
                        log_gui_event("Avatar Update", "Avatar Panel updated with avatar image.")
                    else:
                        self.avatar_label.configure(text="No Avatar Available", image="")
                        log_gui_event("Avatar Update", "No avatar available for display.")
                
                except Exception as e:
                    # Attempt to update visual warning light (optional)
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")

                    logger.error(f"Error updating avatar: {e}")
                    log_gui_event("Avatar Update Error", str(e))

            # If load_avatar_image succeeds, log that event
            else:
                log_gui_event("Avatar Update", "Avatar loaded successfully via load_avatar_image().")

        _update()

        # Schedule the next update after 10 seconds (10000 milliseconds)
        self.frame.after(10000, self.update_avatar)
        # Alternative scheduling method from globals (commented out as requested)
        # self.after(globals_module.AVATAR_REFRESH_RATE * 1000, self.update_avatar)

# ------------------------- Files Panel (Files Tab) -------------------------
class FilesPanel:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.files_display = tk.Listbox(self.frame, height=20)
        self.files_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.options_frame = ttk.Frame(self.frame)
        self.options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.upload_button = ttk.Button(self.options_frame, text="Upload Files", command=self.add_files)
        self.upload_button.pack(side=tk.LEFT, padx=5)
        self.refresh_button = ttk.Button(self.options_frame, text="Refresh List", command=self.refresh_list)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        self.uploaded_files = []
    
    def add_files(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to upload")
        if file_paths:
            uploaded_count = 0
            failed_count = 0
            for f in file_paths:
                try:
                    target = self.categorize_file(f)
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    self.uploaded_files.append(f"{os.path.basename(f)} -> {dest_path}")
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                    uploaded_count += 1
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    logger.error(f"Failed to upload {f}: {e}")
                    messagebox.showerror("Error", f"Failed to upload {f}: {e}")
                    failed_count += 1
            messagebox.showinfo("Files Uploaded", f"{uploaded_count} files uploaded successfully!\n{failed_count} files failed.")
            self.refresh_list()
        else:
            logger.warning("No files selected for upload.")
            messagebox.showwarning("Upload Files", "No files selected.")
    
    def refresh_list(self):
        if not self.uploaded_files:
            messagebox.showinfo("Refresh List", "No files available to display.")
            logger.info("File list refresh attempted with no files.")
        self.files_display.delete(0, tk.END)
        for entry in self.uploaded_files:
            self.files_display.insert(tk.END, entry)
        log_gui_event("Files Refreshed", f"{len(self.uploaded_files)} files listed.")
    
    def categorize_file(self, file_path: str) -> str:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        mapping = {
            ".csv": config.DATASETS_DIR,
            ".json": config.DATASETS_DIR,
            ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
            ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
            ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
            ".png": os.path.join(config.PROJECTS_DIR, "images"),
            ".py": config.IMPORTS_DIR
        }
        target = mapping.get(ext, config.ADDONS_DIR)
        os.makedirs(target, exist_ok=True)
        return target

# ------------------------- Settings Panel (Settings Tab) -------------------------
class SettingsPanel:
    def __init__(self, parent, gui_instance):
        # Re-enforce Voice Profile after GUI load
        try:
            from SarahMemoryVoice import set_voice_profile, set_pitch, set_bass, set_treble
            
            settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    data = json.load(f)
                if "voice_profile" in data:
                    set_voice_profile(data["voice_profile"])
                    if "pitch" in data:
                        set_pitch(data["pitch"])
                    if "bass" in data:
                        set_bass(data["bass"])
                    if "treble" in data:
                        set_treble(data["treble"])
                    logger.info(f"🔊 Re-applied voice profile after GUI load: {data['voice_profile']}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to re-apply voice profile in GUI: {e}")

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.gui = gui_instance
        self.settings = self.load_settings()

        tk.Label(self.frame, text="AI Mode:").pack(pady=5)
        self.api_mode = tk.StringVar(value=self.settings.get("api_mode", "Local"))
        self.data_mode_var = self.api_mode  # ✅ Ensure variable exists
        ttk.Combobox(self.frame, textvariable=self.api_mode, values=["Local", "API"]).pack()

        tk.Label(self.frame, text="Voice Profile:").pack(pady=5)
        self.voice_profile = tk.StringVar(value=self.settings.get("voice_profile", "Default"))
        ttk.Combobox(self.frame, textvariable=self.voice_profile, values=voice.get_voice_profiles()).pack(pady=5)

        for label, var_name in [("Pitch", "pitch"), ("Bass", "bass"), ("Treble", "treble")]:
            tk.Label(self.frame, text=label).pack(pady=5)
            var = tk.DoubleVar(value=self.settings.get(var_name, 1.0))
            setattr(self, var_name, var)
            tk.Scale(self.frame, variable=var, from_=0.5, to=2.0, resolution=0.1, orient="horizontal").pack(fill=tk.X)

        tk.Button(self.frame, text="Import Custom Voice", command=self.import_voice).pack(pady=5)

        tk.Label(self.frame, text="Avatar Settings", font=("Arial", 12, "bold")).pack(pady=10)
        self.avatar_selection = tk.StringVar(value=self.settings.get("avatar_file", ""))
        self.avatar_dropdown = ttk.Combobox(self.frame, textvariable=self.avatar_selection, values=self.get_uploaded_avatars())
        self.avatar_dropdown.pack(pady=5)
        tk.Button(self.frame, text="Upload Avatar", command=self.upload_avatar).pack(pady=5)

        tk.Label(self.frame, text="3D Engine:").pack(pady=5)
        self.engine_selection = tk.StringVar(value=self.settings.get("engine_selection", "Auto"))
        ttk.Combobox(self.frame, textvariable=self.engine_selection, values=["Microsoft3DViewer", "Blender", "Unreal", "Auto"]).pack(pady=5)

        tk.Label(self.frame, text="Theme:").pack(pady=5)
        self.theme_selection = tk.StringVar(value=self.settings.get("theme", ""))
        theme_files = []
        if os.path.exists(config.THEMES_DIR):
            theme_files = [f for f in os.listdir(config.THEMES_DIR) if f.lower().endswith(".css")]
        if not self.theme_selection.get() and theme_files:
            self.theme_selection.set(theme_files[0])
        ttk.Combobox(self.frame, textvariable=self.theme_selection, values=theme_files).pack(pady=5)

        tk.Button(self.frame, text="Save Settings", command=self.save_settings).pack(pady=10)

    def load_settings(self):
        if os.path.exists(config.SETTINGS_FILE):
            try:
                with open(config.SETTINGS_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error("Failed to load settings: " + str(e))

        return {
            "api_mode": "Local",
            "voice_profile": "female",
            "pitch": 1.0,
            "bass": 1.0,
            "treble": 1.0,
            "avatar_file": "",
            "engine_selection": "Auto",
            "theme": ""
        }

    def get_uploaded_avatars(self):
        avatars = []
        if os.path.exists(config.AVATAR_DIR):
            avatars = [file for file in os.listdir(config.AVATAR_DIR) if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
        return avatars

    def upload_avatar(self):
        path = filedialog.askopenfilename(title="Select Avatar Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
        if path:
            try:
                dest = os.path.join(config.AVATAR_DIR, os.path.basename(path))
                shutil.copy(path, dest)
                messagebox.showinfo("Avatar Upload", "Avatar uploaded successfully!")
                log_gui_event("Avatar Upload", f"Copied avatar {path} to {dest}")
                self.avatar_dropdown['values'] = self.get_uploaded_avatars()
                self.avatar_selection.set(os.path.basename(path))
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error(f"Failed to upload avatar: {e}")
                messagebox.showerror("Error", f"Failed to upload avatar: {e}")

    def import_voice(self):
        path = filedialog.askopenfilename(title="Select Voice Profile")
        if path:
            try:
                voice.import_custom_voice_profile(path)
                log_gui_event("Import Voice", f"Imported voice profile from: {path}")
                messagebox.showinfo("Voice Import", "Voice profile imported successfully!")
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error(f"Failed to import voice profile: {e}")
                messagebox.showerror("Error", f"Failed to import voice profile: {e}")

    def save_settings(self):
        from SarahMemoryVoice import set_voice_profile, set_pitch, set_bass, set_treble, save_voice_settings

        data = {
            "api_mode": self.api_mode.get(),
            "voice_profile": self.voice_profile.get(),
            "pitch": self.pitch.get(),
            "bass": self.bass.get(),
            "treble": self.treble.get(),
            "avatar_file": self.avatar_selection.get(),
            "engine_selection": self.engine_selection.get(),
            "theme": self.theme_selection.get()
        }

        selected_mode = self.api_mode.get().lower()
        self.gui.override_data_mode = selected_mode
        log_gui_event("Mode Override", f"Mode changed to: {selected_mode.upper()}")

        try:
            with open(config.SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=4)

            voice.set_voice_profile(data["voice_profile"])
            voice.set_pitch(data["pitch"])
            voice.set_bass(data["bass"])
            voice.set_treble(data["treble"])

            if data["theme"]:
                apply_theme_from_choice(data["theme"])

            save_voice_settings()

            self.gui.update_status()

            messagebox.showinfo("Settings", "Settings saved successfully!")
            log_gui_event("Settings Saved", str(data))

        except Exception as e:
            try:
                if hasattr(config, 'vision_canvas'):
                    config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
            except Exception as ce:
                logger.warning(f"Vision light update failed: {ce}")

            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

# ------------------------- Status Bar -------------------------

class StatusBar:
    def __init__(self, parent):
        # Initialize status bar with three equal-sized status lights: intent, object, network.
        self.frame = ttk.Frame(parent, style="StatusBar.TFrame")
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        # status text
        self.status_label = ttk.Label(self.frame, text="", anchor=tk.W, style="StatusBar.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=5)
        # Intent status light
        self.intent_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.intent_canvas.pack(side=tk.RIGHT, padx=5)
        self.intent_light = self.intent_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Object detection status light
        self.object_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.object_canvas.pack(side=tk.RIGHT, padx=5)
        self.object_light = self.object_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Network status light
        self.network_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.network_canvas.pack(side=tk.RIGHT, padx=5)
        self.network_indicator = self.network_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Start periodic status updates
        self.update_status()

    def set_intent_light(self, color):
        # color: 'green', 'yellow', or 'red'
        self.intent_canvas.itemconfig(self.intent_light, fill=config.STATUS_LIGHTS[color])

    def set_object_light(self, color):
        self.object_canvas.itemconfig(self.object_light, fill=config.STATUS_LIGHTS[color])

    def update_status(self):
        try:
            settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                mode = settings.get("api_mode", "Local")
            else:
                mode = "Local"
        except Exception as e:
            logger.warning(f"⚠️ Failed to read settings for status bar: {e}")
            mode = "Local"

        voice_profile = voice.get_voice_profiles()[0] if voice.get_voice_profiles() else "Default"
        avatar_file = "None"
        global MIC_STATUS, CAMERA_STATUS
        status_text = f"Mode: {mode} | Voice: {voice_profile} | Avatar: {avatar_file} | Mic: {MIC_STATUS} | Camera: {CAMERA_STATUS}"
        self.status_label.config(text=status_text)
        self.gui.status_bar.display_message("© 2025 Brian Lee Baros.")

        try:
            future = asyncio.run_coroutine_threadsafe(async_update_network_state(), async_loop)
            network_state = future.result(timeout=2)
        except Exception as e:
            try:
                if hasattr(config, 'vision_canvas'):
                    config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
            except Exception as ce:
                logger.warning(f"Vision light update failed: {ce}")
            network_state = "red"

        color = {"red": "red", "yellow": "yellow", "green": "green"}.get(network_state, "green")
        self.network_canvas.itemconfig(self.network_indicator, fill=color)

        self.frame.after(1000, self.update_status)


# ------------------------- Main Unified GUI -------------------------
class SarahMemoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=0)  # Tabs
        self.root.rowconfigure(1, weight=0)  # Top button bar
        self.root.rowconfigure(2, weight=1)  # Main content
        self.root.rowconfigure(3, weight=0)  # Status bar
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.title("SarahMemory AI-Bot Companion Platform")
        self.root.geometry("1980x1240")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.override_data_mode = "api"  # Default value



        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.video_chat_tab = ttk.Frame(self.notebook)
        self.files_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.video_chat_tab, text="Video Chat")
        self.notebook.add(self.files_tab, text="Files")
        self.notebook.add(self.settings_tab, text="Settings")
        
        self.connection_panel = ConnectionPanel(self.video_chat_tab, open_settings_window)
        
        self.main_frame = ttk.Frame(self.video_chat_tab)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=2)
        self.main_frame.rowconfigure(0, weight=1)
        
        self.video_panel = VideoPanel(self.main_frame)



        self.avatar_controller = ExtendedAvatarController()  # Create a single instance here
        self.chat_panel = ChatPanel(self.main_frame, self, self.avatar_controller)
        #self.avatar_panel = AvatarPanel(self.main_frame)
        #globals_module.avatar_panel_instance = self.avatar_panel  # ✅ Allow external update after avatar render
 
        self.video_panel.frame.grid(row=0, column=0, sticky="nsew")
        self.chat_panel.frame.grid(row=0, column=1, sticky="nsew")
        #self.avatar_panel.frame.grid(row=0, column=2, sticky="nsew")
        
        self.files_panel = FilesPanel(self.files_tab)
        self.settings_panel = SettingsPanel(self.settings_tab, self)
        
        # Bottom frame for Control Panel and Status Bar.
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control Panel: Theme selection drop-down and Apply button.
        self.control_panel = ttk.Frame(self.bottom_frame)
        self.control_panel.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.theme_var = tk.StringVar()
        if os.path.exists(config.THEMES_DIR):
            theme_files = [f for f in os.listdir(config.THEMES_DIR) if f.lower().endswith(".css")]
        else:
            theme_files = []
        self.theme_var.set(theme_files[0] if theme_files else "")
        theme_dropdown = ttk.Combobox(self.control_panel, textvariable=self.theme_var, values=theme_files)
        theme_dropdown.pack(side=tk.LEFT, padx=5)
        apply_theme_button = ttk.Button(self.control_panel, text="Apply Theme", command=lambda: apply_theme_from_choice(self.theme_var.get()))
        apply_theme_button.pack(side=tk.LEFT, padx=5)
        
        # Status Bar: Using the new StatusBar class with ttk styling.
        self.status_bar = StatusBar(self.bottom_frame)
        
        self.start_voice_recognition_loop()
        self.start_SuperObjectEngine()


        self.last_user_interaction = time.time()
        self.check_idle_thread = threading.Thread(target=self.monitor_idle_time, daemon=True)
        self.check_idle_thread.start()


        self.root.after(3500, self.intro_greeting)
        log_gui_event("GUI Init", "SarahMemoryGUI initialized with previous settings.")
        
        # 🔁 Avatar Panel GPU Launcher
        import subprocess
        try:
            self.avatar_proc = subprocess.Popen(["python", "SarahMemoryAvatarPanel.py"])
            #subprocess.Popen(["python", "SarahMemoryAvatarPanel.py"])
            logger.info("✅ AvatarPanel GPU window launched in parallel.")
        except Exception as launch_error:
            logger.error(f"❌ Failed to launch AvatarPanel: {launch_error}")


    def intro_greeting(self):
        try:
            from SarahMemoryPersonality import get_greeting_response
            from SarahMemoryDatabase import log_ai_functions_event
            greeting = get_greeting_response()
            self.chat_panel.append_message("Sarah: " + greeting)
            run_async(voice.synthesize_voice, greeting)
            log_ai_functions_event("Greeting", greeting)
        except Exception as e:
            logger.error(f"[Greeting Error] {e}")
            fallback = "Hello, I'm Sarah. How can I help you today?"
            self.chat_panel.append_message("Sarah: " + fallback)
            run_async(voice.synthesize_voice, fallback)
        config.AVATAR_IS_SPEAKING = False
        
    def update_avatar_window(self):
        self.avatar_panel.update_avatar()

    def start_voice_recognition_loop(self):
        def voice_loop():
            while True:
                try:
                    # Listen for voice input using the AI module's combined listen_and_process function.
                    recognized = voice.listen_and_process()
                    if recognized:
                        logger.info(f"Voice input received: {recognized}")
                        # Check for avatar commands first.
                        lower_text = recognized.lower()
                        if "create your avatar" in lower_text or "change your" in lower_text:
                            logger.info(f"Avatar command detected: {recognized}")
                            if "create your avatar" in lower_text:
                                self.create_avatar(recognized)
                            elif "change your" in lower_text:
                                self.modify_avatar(recognized)
                        else:
                            # Otherwise, treat as chat input.
                            self.chat_panel.append_message("You (voice): " + recognized)
                            run_async(self.chat_panel.generate_response, recognized)
                except Exception as e:
                    logger.error(f"Voice recognition error: {e}")
                    time.sleep(5)
        run_async(voice_loop)   
        
    
    def start_SuperObjectEngine(self):
        def detection_loop():
            while True:
                try:
                    with shared_lock:
                        frame = shared_frame.copy() if shared_frame is not None else None
                        if frame is not None:
                            objects = sobje.ultra_detect_objects(frame)
                            if objects:
                                log_gui_event("Object Detection", f"Detected objects: {', '.join(objects)}")
                    time.sleep(3)
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    logger.error(f"Object detection error: {e}")
                    time.sleep(20)
        run_async(detection_loop)

    def monitor_idle_time(self):        
        while True:
            idle_time = time.time() - self.last_user_interaction
            if idle_time > config.DL_IDLE_TIMER:  # Idle Timer is set in SarahMemoryGlobals.py
                try:
                    from SarahMemoryOptimization import run_idle_optimization_tasks
                    run_idle_optimization_tasks()
                except Exception as e:
                    logger.warning(f"[Idle Optimization Error] {e}")
            time.sleep(300)  # Check every 5 minutes
    
    def shutdown(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            log_gui_event("Shutdown", "User exited the GUI.")
            self.video_panel.release_resources()
            voice.shutdown_tts()
            if hasattr(self, 'avatar_proc') and self.avatar_proc.poll() is None:
                self.avatar_proc.terminate()
            self.root.destroy()
            try:
                root.after_cancel(update_video_job_id)
                root.after_cancel(update_status_job_id)
                root.after_cancel(update_avatar_job_id)
                root.after_cancel(self.video_update_job)
            except Exception:
                pass
            try:
                update_video_job_id = root.after(1000, update_video)
                update_status_job_id = root.after(1000, update_status)
                update_avatar_job_id = root.after(1000, update_avatar)
                root.after_cancel(self.status_update_job)
            except Exception:
                pass
            logger.info("SarahMemory GUI shutdown successfully.")
        else:
            logger.info("Shutdown cancelled by user.")



# ------------------------- Settings Window -------------------------
def open_settings_window():
    try:
        win = tk.Toplevel()
        win.title("Settings")
        win.geometry("320x240")
        SettingsPanel(win, win)
        logger.info("Settings window opened successfully.")
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        logger.error(f"Error opening settings window: {e}")
        messagebox.showerror("Error", "Failed to open the settings window.")

#================----------Place SUPER OBJECT DETECTION ENGINE HERE----------================
class SuperObjectEngine:
    def __init__(self):
        from SarahMemorySOBJE import ultra_detect_objects
        self.detect = ultra_detect_objects

    def detect_objects(self, frame):
        try:
            return self.detect(frame)
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
#==========================END OF SUPER OBJECT DETECTION ENGINE HERE===========

# ------------------------- Main Execution -------------------------
def run_gui():
    root = tk.Tk()
    app = SarahMemoryGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    logger.info("Starting unified GUI mainloop.")
    root.mainloop()

if __name__ == '__main__':
    run_gui()

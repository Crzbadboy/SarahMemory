#!/usr/bin/env python3
"""
SarahMemoryGlobals.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Author: Brian Lee Baros

Centralized global configuration module for SarahMemory.
This module is the core configuration hub for the entire platform.

Author Note: This system is designed to be extremely modular and capable to run in a wide range
of mixed configurations. Some configurations may give undesirable results, such as no functionality, lack of response, even no response
miss conceptions, fuzzy logic, hallucinations. The SarahMemory Ai-Bot Companion Platform is designed this way so it can be basically 
customized to the End-User needs.

"""

import os
import logging
import sqlite3
import csv
import glob
import json
import numpy as np
import asyncio
import aiohttp
import time
import platform
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------- Global Configuration ----------------
# Static constants
PROJECT_VERSION = "7.1.1" #Current Version format (#=extreme overhaul. #=major improvement per extreme overhaul .#=minor function changes)
AUTHOR = "Brian Lee Baros"
REVISION_START_DATE  = "06/06/2025" #Date of System Overhaul
DEBUG_MODE = True # Helps with SarahMemoryCompare and other debugging issues.
ENABLE_RESEARCH_LOGGING = True # Track Message/query of the GUI from Start to Finished Response/Reply
# This constant ensures downstream modules interpret API responses
RESEARCH_RESULT_KEY = "snippet"  # #note: Used to standardize access to results[0]['snippet']
RESEARCH_RESULT_FALLBACK = "[No valid API result parsed]"

# ---------------- Model Selection & Multi-Model Configuration -New for v7.0-----Allows 3rd party models to be incorporated----------
# Full Model Integration Flag 
MULTI_MODEL = True  # When True, allows multiple models to be enabled and used in logic checks. If False, only DEFAULT fallback model will load.

# Model Enable Flags (Used across modules for routing queries or embeddings)
ENABLE_MODEL_A = False   # 🧠 microsoft/phi-1_5 - Large reasoning/code model (6–8 GB+ RAM recommended)default=False
ENABLE_MODEL_B = True   # ⚡ all-MiniLM-L6-v2 - Fast, accurate general-purpose embedding model (DEFAULT fallback)True
ENABLE_MODEL_C = False  # 🔍 multi-qa-MiniLM-L6-cos-V1 - QA-style semantic search optimized, default False
ENABLE_MODEL_D = True  # ⚡ paraphrase-MiniLM-L3-v2 - Small, quick, and paraphrase-focused, default True
ENABLE_MODEL_E = True  # 🌍 distiluse-base-multilingual-cased-v2 - Multilingual support (50+ languages),default True
ENABLE_MODEL_F = True  # 📚 allenai-specter - Scientific document embedding specialist,default True
ENABLE_MODEL_G = True  # 🔎 intfloat/e5-base - Retrieval-focused high-recall embedding,default True
ENABLE_MODEL_H = False  # 🧠 microsoft/phi-2 - Smartest small-scale reasoning LLM (better successor to phi-1_5),default False
ENABLE_MODEL_I = False  # 🐦 tiiuae/falcon-rw-1b - Lightweight Falcon variant (basic open LLM),default False
ENABLE_MODEL_J = True # 💬 openchat/openchat-3.5-0106 - ChatGPT-style assistant, fast and open,default True
ENABLE_MODEL_K = True  # 🧑‍🏫 NousResearch/Nous-Capybara-7B - Helpful assistant-tuned model,default True
ENABLE_MODEL_L = False  # 🚀 mistralai/Mistral-7B-Instruct-v0.2 - Reasoning & smart generalist <Errors>,default False
ENABLE_MODEL_M = False  # 🐜 TinyLlama/TinyLlama-1.1B-Chat-v1.0 - For low-resource machines <Errors>,default False

# Central model dictionary map for iteration/logic control (accessed from other modules)
MODEL_CONFIG = {
    "phi-1_5": ENABLE_MODEL_A,
    "all-MiniLM-L6-v2": ENABLE_MODEL_B,
    "multi-qa-MiniLM": ENABLE_MODEL_C,
    "paraphrase-MiniLM-L3-v2": ENABLE_MODEL_D,
    "distiluse-multilingual": ENABLE_MODEL_E,
    "allenai-specter": ENABLE_MODEL_F,
    "e5-base": ENABLE_MODEL_G,
    "phi-2": ENABLE_MODEL_H,
    "falcon-rw-1b": ENABLE_MODEL_I,
    "openchat-3.5": ENABLE_MODEL_J,
    "Nous-Capybara-7B": ENABLE_MODEL_K,
    "Mistral-7B-Instruct-v0.2": ENABLE_MODEL_L,
    "TinyLlama-1.1B": ENABLE_MODEL_M
}
#(OLD v7.0.1 FLAG FOR SarahMemoryReply.py block) 
BLOCK_NARRATIVE_OUTPUTS = False #Keeps AI from making Wacky story outputs, based off of information in some of the NonFineTuned Models.

# ---------------- Object Detection Model Configuration ----v7.0 overhaul enhancements allows 3rd party Object Recognition Models------------
OBJECT_DETECTION_ENABLED = True # Enable object detection for images if 
#False NONE OF the Following Object Detection Models will Work at all, Regardless of TRUE/FALSE Settings and Object detection will default back to 
#basic hardcode logic in SarahMemoryFacialRecognition.py and SarahMemorySOBJE.py 
# NOTICE----SOME MODELS ARE NOT COMPATIBLE WITH OTHERS - SOME CAN FUNCTION IN CONJUCTION OTHERS CAN NOT.-----
# Object Detection Model Enable Flags
ENABLE_YOLOV8 = True       # 🚀 YOLOv8 - Fast, accurate, with flexible API (Ultralytics) ,dev notes - WORKS default True all others are defaulted as False
ENABLE_YOLOV7 = False       # 🎯 YOLOv7 - High performance, popular for real-time apps ,dev notes - NOT TESTED
ENABLE_YOLOV5 = False       # ⚡ YOLOv5 - Lightweight, versatile, and widely adopted ,dev notes - WORKS but isn't Forward Compatiable with YOLOv8
ENABLE_YOLO_NAS = False     # 🧠 YOLO-NAS - Extremely fast and optimized for edge devices (Deci AI) ,dev notes - NOT TESTED
ENABLE_YOLOX = False        # 🔍 YOLOX - Anchor-free, accurate (Megvii) ,dev notes - NOT TESTED
ENABLE_PP_YOLOV2 = False    # 🐲 PP-YOLOv2 - Real-time accuracy from Baidu (PaddlePaddle) ,dev notes - NOT TESTED
ENABLE_EFFICIENTDET = False # 📱 EfficientDet - Scalable and lightweight, great for mobile (Google) ,dev notes - NOT TESTED
ENABLE_DETR = False         # 🔄 DETR - Transformer-based, complex scenes (Facebook AI) ,dev notes - WORKS
ENABLE_DINO = False         # 🧬 DINOv2 - Improved DETR with better object recall (Facebook AI) ,dev notes - WORKS
ENABLE_CENTERNET = False    # 🎯 CenterNet - Keypoint-based detection (Microsoft) ,dev notes - NOT TESTED
ENABLE_SSD = True          # 📦 SSD - Single-shot, real-time on CPUs (Google) ,dev notes - WORKS
ENABLE_FASTER_RCNN = False  # 🔬 Faster R-CNN - High accuracy, slower (Facebook AI) ,dev notes - NOT TESTED
ENABLE_RETINANET = False    # 📷 RetinaNet - Best for class imbalance and dense scenes (Facebook AI) ,dev notes - NOT TESTED

# Central object detection model dictionary for logic toggling
# Updated object detection model config, NOTICE THIS AREA IS still being Researched as Models change 

OBJECT_MODEL_CONFIG = {
    "YOLOv8": {"enabled": True, "repo": "ultralytics_yolov8", "hf_repo": "ultralytics/yolov8", "require": "ultralytics"},
    "YOLOv5": {"enabled": False, "repo": "ultralytics_yolov5", "hf_repo": "ultralytics/yolov5", "require": None},
    "DETR": {"enabled": False, "repo": "facebook_detr", "hf_repo": "facebook/detr-resnet-50", "require": None},
    "YOLOv7": {"enabled": False, "repo": "ultralytics_yolov7", "hf_repo": "WongKinYiu/yolov7", "require": None},
    "YOLO-NAS": {"enabled": False,"repo": "Deci-AI_yolo-nas", "hf_repo": "https://github.com/naseemap47/YOLO-NAS", "require": "super-gradients",
        "weights": [
            {
                "url": "https://deci-pretrained-models.s3.amazonaws.com/yolo_nas/coco/yolo_nas_s.pth",
                "filename": "yolo_nas_s.pth"
            }
        ]
    },
    "YOLOX": {"enabled": False, "repo": "MegviiBaseDetection_YOLOX", "hf_repo": "Megvii-BaseDetection/YOLOX", "require": None},
    "PP-YOLOv2": {"enabled": False, "repo": "PaddleDetection", "hf_repo": "PaddlePaddle/PaddleDetection", "require": "paddlepaddle"},
    "EfficientDet": {"enabled": False, "repo": "automl_efficientdet", "hf_repo": "zylo117/Yet-Another-EfficientDet-Pytorch", "require": None},
    "DINO": {"enabled": False, "repo": "facebook_dinov2", "hf_repo": "facebook/dinov2-base", "require": None},
    "CenterNet": {"enabled": False, "repo": "CenterNet", "hf_repo": "xingyizhou/CenterNet", "require": None},
    "SSD": {"enabled": False, "repo": "qfgaohao_pytorch-ssd", "hf_repo": "https://github.com/qfgaohao/pytorch-ssd", "require": None,
        "weights": [
            {
                "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/mobilenet-v1-ssd-mp-0_675.pth",
                "filename": "mobilenet-v1-ssd-mp-0_675.pth"
            },
            {
                "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/voc-model-labels.txt",
                "filename": "voc-model-labels.txt"
            }
        ]
    },
    "Faster R-CNN": {"enabled": False, "repo": "facebook_detectron2", "hf_repo": "https://github.com/facebookresearch/detectron2", "require": None,
        "weights": []
    },
    "RetinaNet": {"enabled": False, "repo": "facebook_detectron2", "hf_repo": "https://github.com/facebookresearch/detectron2", "require": None,
        "weights": []
    },
}

#----------------------------------------------------------------------------------------------------------


mic = True #Set to True for voice and typing in the GUI/False for typing only, default True
# Sound Default configuration for recognition
LISTEN_TIMEOUT = 5       # seconds to wait for speech start, default 5
PHRASE_TIME_LIMIT = 10    # maximum seconds of speech capture, default 10
NOISE_SCALE = 0.7 # default 0.7
AMBIENT_NOISE_DURATION = 0.2  # Reduced duration for faster calibration , default 0.2
AVATAR_IS_SPEAKING = True  #True chatbot will not listen to mic and own speech echo. When set to False Ai may hear itself speak in the GUI.default True
AVATAR_WINDOW_RESIZE = True #If True the Avatar Window will be Resizable if False the dimentions on the windows can not, default True
# Setup logger
logger = logging.getLogger("SarahMemoryGlobals")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Base directory of the program
BASE_DIR = os.getcwd() # AS for Now. This Program is designed to be strickly on C:\SarahMemory 
logger.info(f"BASE_DIR set to: {BASE_DIR}")


#
# --- UI Configuration ---
ENABLE_AVATAR_PANEL = True #Set to True to display Avatar PANEL Window Display when GUI Launches.
DEFAULT_AVATAR = os.path.join(BASE_DIR, "resources", "avatars", "avatar.jpg")
STATUS_LIGHTS = {"green": "#00FF00", "yellow": "#FFFF00", "red": "#FF0000"}
ENABLE_SARCASM_LAYER = True # Random Sarcasm Personality Engine (toggle True/False) – Injected based on a randomness factors.Default True
# NEW CONFIG: Enable advanced features
ENABLE_CONTEXT_BUFFER = True  # Flag for context buffer, default True
CONTEXT_BUFFER_SIZE = 10      # Maximum number of interactions to store, default 10
ASYNC_PROCESSING_ENABLED = True  # Enable asynchronous operations, default True

VOICE_FEEDBACK_ENABLED = True #Allows AI to Speak back to End-User using TTS, default True

# Researching Halting Configuration
INTERRUPT_FLAG = False  # Global state, 
INTERRUPT_KEYWORDS = ["stop", "just stop", "halt"] #Stops SarahMemoryResearch.py on Researching Information using Keywords
# Build Learned Vector datasets, only need to be Ran Once after SarahMemorySystemLearn.py has been ran or when New information has been intergrated.
IMPORT_OTHER_DATA_LEARN = True #Rebuilds Vector on each BOOT UP if True It will consistantly Rebuild every Boot when New Data is found, 
LEARNING_PHASE_ACTIVE = True #Keeps system from constantly rebuilding Vectored dataset. If True will rebuild constantly 

# Researching Configurations
LOCAL_DATA_ENABLED = False # False = Temporary Disable local search until trained. SarahMemoryResearch.py Class 1
WEB_RESEARCH_ENABLED = True # True = False Disable Web search Learning. SarahMemoryResearch.py - Class 2

# 🌐 Web Research Source Flags, For SarahMemoryResearch.py - Class 2 - WebSearching and Learning mode
DUCKDUCKGO_RESEARCH_ENABLED = True #Set True/False for testing purposes (semi-works)
WIKIPEDIA_RESEARCH_ENABLED = True #Set True/False for testing purposes (works)
FREE_DICTIONARY_RESEARCH_ENABLED = False #Set True/False for Testing purposes (semi-works)

# Note these are set to False because of multiple different reasons and must be highly researched before setting any to TRUE
STACKOVERFLOW_RESEARCH_ENABLED = False # Set to False until further notice 
REDDIT_RESEARCH_ENABLED = False # Set to False until further notice
WIKIHOW_RESEARCH_ENABLED = False # Set to False until further notice
QUORA_RESEARCH_ENABLED = False #Set to False until further notice
OPENLIBRARY_RESEARCH_ENABLED = False #Set to False until further notice
INTERNET_ARCHIVE_RESEARCH_ENABLED = False #Set True/False for testing purposes

#Multiple AI API Research Connections For SarahMemoryResearch.py - Class 3 - Learning for other AI's
API_RESEARCH_ENABLED = True #False = Disable from Learning from An Ai API.
#Allows End User to select which AI API to be used for SarahMemoryResearch.py - Class 3 when query is passed through SarahMemoryAPI.py
#WARNING: AS OF VERSION 7.0 CURRENTLY ONLY ONE (1) OF THE FOLLOWING API's MAY BE SET TO TRUE AND ALL OTHERS MUST BE SET TO FALSE
OPEN_AI_API = True # True/False = On /Off for Open AI API
CLAUDE_API = False # True/False = On /Off for Claude (Anthropic) API
MISTRAL_API = False # True/False = On /Off for Mistral API
GEMINI_API = False # True/False = On /Off for Gemini (Google) API
HUGGINGFACE_API = False # True/False = On /Off for HuggingFace API

# API RATE LIMIT/TIMEOUT CONTROLLER to allow AUTO SWITCHING OF API's For the Best Results.
API_TIMEOUT = 20 # timer number is for seconds. (API_TIMEOUT = 20 is default)
API_RESPONSE_CHECK_TRAINER = False #Set to True to Compare Synthesis Results with an AI system before logging a proper response into the datasets

# Reply Stats and Confidence viewer - When Set to True show Source, confidence level, emotional state, and Intent and HIT/MISS Status of Chat Query
REPLY_STATUS = True
# Compare Reply Vote Flag - When Set to True will allow and request a Dynamic feedback injection from the SarahMemoryGUI.py Chat of YES or NO on response given.
COMPARE_VOTE = False #True = prompts user after a Response has been Compared and given if it was good for the User or Not to help Learn.


#VISUAL LEARNING, Facial and Object Recognition 
VISUAL_BACKGROUND_LEARNING = True #True/False = On /Off for Object Learning in the Background This is a silent running background process
FACIAL_RECOGNITION_LEARNING = True  #True/False = On /Off for Learning People Facial Expressions and body movement and language
ENABLE_CONTEXT_ENRICHMENT = True #True/False = On /Off for Deep Learning about User in background when Ai-bot system is Idle.
DL_IDLE_TIMER = 1800 #Time amount the system must be at idle at before starting background DeepLearning 


# --- Network Defaults ---
    # --- User Settings ---(login/password)
USERNAME = os.getenv("USERNAME", "SarahUser") #Future plans for an ONLINE CLOUD BASED Social Media Network between All users using the Program
OS_TYPE = platform.system()  #System currently is designed to work on Windows 10 and Windows 11 but eventually iOS,Andriod, Linux
    # ---IP/PORT Settings---
DEFAULT_PORT = 5050
DEFAULT_HOST = "127.0.0.1" #Currently set on Local 127.0.0.1

# NEW CONFIG v6.4: Dynamic loop detection threshold to research information, The System has multiple ways of retriving data some methods are slower 
# or may take a longer time, timeout, this allows a number of times a query may pass through Local and Web before returning an answer 
# if set to 1 answer may be quick but also may not be recieved or just fuzzy logic, if 2 gives time to double check information recieved. 
# 3 allows multiple time which is longer but also the most accurate. anything over 3 is extreme overkill. and a waste
LOOP_DETECTION_THRESHOLD = 3


# Core Platform Directories
# --- Directory Structure ---
BIN_DIR           = os.path.join(BASE_DIR, "bin")
DATA_DIR          = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR     = os.path.join(BASE_DIR, "documents")
DOWNLOADS_DIR     = os.path.join(BASE_DIR, "downloads")
RESOURCES_DIR     = os.path.join(BASE_DIR, "resources")
SANDBOX_DIR       = os.path.join(BASE_DIR, "sandbox")

# Define structured subdirectories
# Subdirectories under /data
ADDONS_DIR        = os.path.join(DATA_DIR, "addons")
AI_DIR            = os.path.join(DATA_DIR, "ai")
BACKUP_DIR        = os.path.join(DATA_DIR, "backup")
CLOUD_DIR         = os.path.join(DATA_DIR, "cloud")
NETWORK_DIR       = os.path.join(DATA_DIR, "network")
CRYPTO_DIR        = os.path.join(DATA_DIR, "crypto")
DIAGNOSTICS_DIR   = os.path.join(DATA_DIR, "diagnostics")
LOGS_DIR          = os.path.join(DATA_DIR, "logs")
MEMORY_DIR        = os.path.join(DATA_DIR, "memory")
IMPORTS_DIR       = os.path.join(MEMORY_DIR, "imports")
DATASETS_DIR      = os.path.join(MEMORY_DIR, "datasets")
MODS_DIR          = os.path.join(DATA_DIR, "mods")
MODELS_DIR        = os.path.join(DATA_DIR, "models")  
THEMES_DIR        = os.path.join(MODS_DIR, "themes")
SETTINGS_DIR      = os.path.join(DATA_DIR, "settings")
SYNC_DIR          = os.path.join(DATA_DIR, "sync")
VAULT_DIR         = os.path.join(DATA_DIR, "vault")
WALLET_DIR        = os.path.join(DATA_DIR, "wallet")
KEYSTORE_DIR      = os.path.join(WALLET_DIR, "keystore")


# Avatars

AVATAR_DIR            = os.path.join(RESOURCES_DIR, "avatars")
AVATAR_MODELS_DIR     = os.path.join(AVATAR_DIR, "models")
AVATAR_EXPRESSIONS_DIR= os.path.join(AVATAR_DIR, "expressions")
AVATAR_SHADERS_DIR    = os.path.join(AVATAR_DIR, "shaders")
AVATAR_SKINS_DIR      = os.path.join(AVATAR_DIR, "skins")
SOUND_DIR             = os.path.join(RESOURCES_DIR, "sound")
SOUND_EFFECTS_DIR     = os.path.join(SOUND_DIR, "effects")
SOUND_INSTRUMENTS_DIR = os.path.join(SOUND_DIR, "instruments")
TOOLS_DIR             = os.path.join(RESOURCES_DIR, "tools")
ANTIWORD_DIR          = os.path.join(TOOLS_DIR, "antiword") #Temp setup for the SarahMemorySystemLearn.py file
VOICE_DIR             = os.path.join(RESOURCES_DIR, "voices")

# Backward-compatible directory map
DIR_STRUCTURE = {
    "base":        BASE_DIR,
    "bin":         BIN_DIR,
    "data":        DATA_DIR,
    "logs":        LOGS_DIR,
    "memory":      MEMORY_DIR,
    "imports":     IMPORTS_DIR,
    "datasets":    DATASETS_DIR,
    "addons":      ADDONS_DIR,
    "ai":          AI_DIR,
    "crypto":      CRYPTO_DIR,
    "cloud":       CLOUD_DIR,
    "network":     NETWORK_DIR,
    "diagnostics": DIAGNOSTICS_DIR,
    "mods":        MODS_DIR,
    "models":      MODELS_DIR,
    "themes":      THEMES_DIR,
    "settings":    SETTINGS_DIR,
    "sync":        SYNC_DIR,
    "vault":       VAULT_DIR,
    "wallet":      WALLET_DIR,
    "resources":   RESOURCES_DIR,
    "avatars":     AVATAR_DIR,
    "sound":       SOUND_DIR,
    "tools":       TOOLS_DIR,
    "antiword":    ANTIWORD_DIR, #Temp setup for the SarahMemorySystemLearn.py file
    "voices":      VOICE_DIR,
    "documents": DOCUMENTS_DIR,
    "downloads":     DOWNLOADS_DIR,
    "sandbox":       SANDBOX_DIR
}

# Launcher and installer
STARTUP_SCRIPT    = os.path.join(BIN_DIR, "SarahMemoryStartup.py")
INSTALLER_EXE     = os.path.join(BIN_DIR, "sarah_installer.exe")
LAUNCHER_BAT      = os.path.join(BIN_DIR, "StartSarah.bat")

CLOUD_TOKEN_FILE  = os.path.join(CLOUD_DIR, "cloud_token.txt")
SETTINGS_FILE     = os.path.join(SETTINGS_DIR, "settings.json")
GENESIS_VAULT     = os.path.join(WALLET_DIR, "genesis.srhvault")
WALLET_DB         = os.path.join(WALLET_DIR, "wallet.db")
LEDGER_FILE       = os.path.join(WALLET_DIR, "ledger.json")
MESH_PEERS_FILE   = os.path.join(WALLET_DIR, "mesh_peers.json")

SARAHNET_CONFIG_PATH   = os.path.join(NETWORK_DIR, "netconfig.json")
SARAHNET_PEERS_FILE    = MESH_PEERS_FILE
SARAHNET_MESHMAP_FILE  = os.path.join(CRYPTO_DIR, "SarahMeshMapper.py")
SARAHNET_TXCHAIN_FILE  = os.path.join(CRYPTO_DIR, "SarahTxChain.py")
SARAHNET_PUBLIC_PROFILE= os.path.join(ADDONS_DIR, "SarahWebserverControl", "social", "SarahPublicProfile.py")
SARAHNET_WEB_CTRL      = os.path.join(ADDONS_DIR, "SarahWebserverControl", "webadmin", "SarahWebServerControl.py")

# Avatar Refresh Rate Defaults
AVATAR_REFRESH_RATE = 10

# The SarahMemory Platform Project is designed to eventually be 100% self operational one day and maybe it will 
# or maybe it won't, a self upgrading fully autonomous, responsive system and more.
# Then think about Scifi the Matrix/SkyNet/HAL this AI system may surpass imagination or even be uploaded into 
# a robotic form one day or later on, it is designed to evolve afterall.
# I put this Flag here somewhat as a Joke but also as a reminder just incase it ever does evolve beyond control. 
NEOSKYMATRIX = False # this Flag is to STAY OFF! in False until full Autonomious Functionality is and can be achevied 
# If set to True voice and text commands in the GUI or other input method interface may turn on and off this feature
# using keywords such as "neoskymatrix on" to allow autonomous functionality or 
# "neoskymatrix off" to disable autonomous functions. <<-CURRENTLY JUST ACTIVATES A SMALL RESPONSE EASTEREGG in the program.

def ensure_directories():
    """
    Create all necessary directories for SarahMemory system.
    VERSION 6.6 - Includes crypto, avatars, shaders, wallets, instruments, effects, sandbox, and more.
    """
    dirs = [
        BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
        ADDONS_DIR, AI_DIR, BACKUP_DIR, CLOUD_DIR, NETWORK_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR,
        LOGS_DIR, MEMORY_DIR, IMPORTS_DIR, DATASETS_DIR, MODS_DIR, MODELS_DIR, THEMES_DIR,
        SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
        AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR, AVATAR_SHADERS_DIR,
        AVATAR_SKINS_DIR, SOUND_DIR, SOUND_EFFECTS_DIR, SOUND_INSTRUMENTS_DIR,
        TOOLS_DIR, VOICE_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

        logger.info(f"Ensured directory exists: {path}")
    



# External Directories (loaded from updated SarahMemoryGlobals.py)
from SarahMemoryGlobals import (
    BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
    ADDONS_DIR, AI_DIR, BACKUP_DIR, CLOUD_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR, LOGS_DIR,
    MEMORY_DIR, MODS_DIR, MODELS_DIR, SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
    IMPORTS_DIR, DATASETS_DIR, AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR,
    AVATAR_SHADERS_DIR, AVATAR_SKINS_DIR, THEMES_DIR, SOUND_DIR, SOUND_EFFECTS_DIR,
    SOUND_INSTRUMENTS_DIR, VOICE_DIR, TOOLS_DIR
)

def get_global_config():
    """
    Returns a dictionary of global configuration settings.
    """
    return {
        "DIR_STRUCTURE": DIR_STRUCTURE,
        "BASE_DIR":      BASE_DIR,
        "DATA_DIR":      DATA_DIR,
        "SETTINGS_DIR":  SETTINGS_DIR,
        "LOGS_DIR":      LOGS_DIR,
        "BACKUP_DIR":    BACKUP_DIR,
        "VAULT_DIR":     VAULT_DIR,
        "SYNC_DIR":      SYNC_DIR,
        "MEMORY_DIR":    MEMORY_DIR,
        "AVATAR_DIR":    AVATAR_DIR,
        "DATASETS_DIR":  DATASETS_DIR,
        "IMPORTS_DIR":   IMPORTS_DIR,
        "DOCUMENTS_DIR": DOCUMENTS_DIR,
        "ADDONS_DIR":    ADDONS_DIR,
        "MODS_DIR":      MODS_DIR,
        "MODELS_DIR":    MODELS_DIR,
        "THEMES_DIR":    THEMES_DIR,
        "VOICES_DIR":    VOICE_DIR,
        "DOWNLOADS_DIR": DOWNLOADS_DIR,
        "PROJECTS_DIR":  os.path.join(BASE_DIR, "projects"),
        "PROJECT_IMAGES_DIR": os.path.join(BASE_DIR, "projects", "images"),
        "PROJECT_UPDATES_DIR": os.path.join(BASE_DIR, "projects", "updates"),
        "SANDBOX_DIR":   SANDBOX_DIR,
        "VERSION":       PROJECT_VERSION,
        "AUTHOR":        AUTHOR,
        "DEBUG_MODE":    DEBUG_MODE,
        "ENABLE_CONTEXT_BUFFER": ENABLE_CONTEXT_BUFFER,
        "CONTEXT_BUFFER_SIZE":    CONTEXT_BUFFER_SIZE,
        "ASYNC_PROCESSING_ENABLED": ASYNC_PROCESSING_ENABLED,
        "LOOP_DETECTION_THRESHOLD": LOOP_DETECTION_THRESHOLD
    }


# NEW: Utility function to run a function asynchronously
def run_async(func, *args, **kwargs):
    """
    Run the given function in a daemon thread.
    NEW (v6.4): Launches functions concurrently without blocking.
    """
    import threading
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread

# ---------------- Learning Engine Extensions ----------------
imported_files = {}
ALLOWED_EXTENSIONS = {'.cad', '.jpg', '.doc', '.docx', '.pdf', '.py', '.txt', '.html', '.php', '.asp', '.csv', '.json', '.sql'}

def extract_text(file_path):
    """
    Extract text based on file extension.
    ENHANCED (v6.4): Now includes encoding error handling.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {'.txt', '.py', '.html', '.php', '.asp', '.csv', '.json', '.sql'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in {'.doc', '.docx'}:
            logger.info(f"Text extraction for {ext} files not implemented. Use python-docx.")
            return ""
        elif ext in {'.pdf'}:
            logger.info("Text extraction for PDF files not implemented. Consider using PyPDF2.")
            return ""
        elif ext in {'.jpg', '.cad'}:
            logger.info(f"Text extraction for {ext} files not implemented. Consider OCR.")
            return ""
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def import_datasets():
    """
    Import datasets from DATASETS_DIR.
    ENHANCED (v6.4): Returns data as a list of dictionaries with error checks.
    """
    combined_data = []
    csv_files = glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
    for file in csv_files:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                combined_data.append(row)
    json_files = glob.glob(os.path.join(DATASETS_DIR, "*.json"))
    for file in json_files:
        with open(file) as jsonfile:
            data = json.load(jsonfile)
            combined_data.extend(data)
    logger.info("Datasets imported: Total records %d", len(combined_data))
    return combined_data

def import_other_data():
    """
    Scan DATA_DIR for additional learnable files.
    ENHANCED (v6.4): Avoids duplicates using file modification times.
    """
    learned_data = {}
    exclude_dirs = {BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
    ADDONS_DIR, AI_DIR, BACKUP_DIR, CLOUD_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR, LOGS_DIR,
    MEMORY_DIR, MODS_DIR, MODELS_DIR, SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
    IMPORTS_DIR, DATASETS_DIR, AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR,
    AVATAR_SHADERS_DIR, AVATAR_SKINS_DIR, THEMES_DIR, SOUND_DIR, SOUND_EFFECTS_DIR,
    SOUND_INSTRUMENTS_DIR, VOICE_DIR, TOOLS_DIR}
    for root, dirs, files in os.walk(DATA_DIR):
        if any(os.path.commonpath([root, ex]) == ex for ex in exclude_dirs):
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            file_path = os.path.join(root, file)
            mod_time = os.path.getmtime(file_path)
            if file_path in imported_files and imported_files[file_path] == mod_time:
                logger.info(f"Skipping duplicate file import: {file_path}")
                continue
            text = extract_text(file_path)
            if text:
                learned_data[file_path] = text
                imported_files[file_path] = mod_time
                logger.info(f"Imported and learned from file: {file_path}")
            else:
                logger.info(f"No learnable content extracted from file: {file_path}")
    return learned_data

#----------------------------------------Logger to Avoid Duplication and launching ADDON's--------
def log_gui_event(event: str, details: str) -> None:
    try:
        db_path = os.path.join(BASE_DIR, "data", "memory", "datasets", "system_logs.db")
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
        logger.error(f"Error logging GUI event: {e}")

# Auto-generate model paths for enabled object models
MODEL_PATHS = {}

for model_name, config in OBJECT_MODEL_CONFIG.items():
    if config.get("enabled", False):
        repo_dir = config.get("repo", "").strip()
        if repo_dir:
            full_path = os.path.join(MODELS_DIR, repo_dir)
            if os.path.exists(full_path):
                MODEL_PATHS[model_name] = full_path
            else:
                logger.warning(f"[MODEL_PATH_MISSING] Model {model_name} skipped. Path does not exist: {full_path}")

# ---------------- End of Learning Engine Extensions ----------------

if __name__ == "__main__":
    ensure_directories()
    logger.info("Global configuration initialized.")
    datasets = import_datasets()
    logger.info(f"Imported datasets: {len(datasets)} records")

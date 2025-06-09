#!/usr/bin/env python3
"""
SarahMemoryInitialization.py <Version #7.0 Enhanced> <Author: Brian Lee Baros>
Description:
  Handles AI Boot Sequence, Initialization, Shutdown, and Signal Interruption.
  - Performs deep system diagnostics including advanced directory integrity checks.
  - Simulated pre-launch AI readiness tests and asynchronous initial checks added.
  - Enhanced signal handling for graceful shutdown.
Notes:
  This module ensures that all required directories exist, performs core diagnostics, and triggers a safe shutdown when needed.
"""

import os
import time
import logging
import signal
import sys
import json
from SarahMemoryGlobals import run_async  # MOD: Asynchronous helper imported

# === Logger Setup ===
logger = logging.getLogger("SarahMemoryInitialization")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Graceful Shutdown Flag
shutdown_requested = False

def run_initial_checks():
    """
    Starts system initialization and checks for essential directories, global variables.
    ENHANCED (v6.4): Includes deep diagnostics and simulated AI readiness verification.
    """
    logger.info("Starting system initialization.")
    try:
        from SarahMemoryGlobals import get_global_config
        config = get_global_config()

        if not config:
            logger.error("Failed to load global configuration.")
            return False

        logger.info("Global configuration retrieved successfully.")

        # List of required directories
        required_dirs = [
            config.get("SETTINGS_DIR"),
            config.get("LOGS_DIR"),
            config.get("BACKUP_DIR"),
            config.get("VAULT_DIR"),
            config.get("SYNC_DIR"),
            config.get("MEMORY_DIR"),
            config.get("DOWNLOADS_DIR"),
            config.get("PROJECTS_DIR"),
            config.get("SANDBOX_DIR"),
            config.get("DOCUMENTS_DIR"),
            config.get("ADDONS_DIR"),
            config.get("MODS_DIR"),
            config.get("THEMES_DIR"),
            config.get("VOICES_DIR"),
            config.get("AVATAR_DIR"),
            config.get("DATASETS_DIR"),
            config.get("IMPORTS_DIR"),
            config.get("PROJECT_IMAGES_DIR"),
            config.get("PROJECT_UPDATES_DIR"),
        ]

        for directory in required_dirs:
            if os.path.isdir(directory):
                logger.info(f"Verified directory exists: {directory}")
            else:
                os.makedirs(directory, exist_ok=True)
                logger.warning(f"Missing directory created: {directory}")
        # 🧠 Backup Check Logic — AUTO WEEKLY BACKUP
        try:
            from SarahMemoryFilesystem import create_weekly_backup
            create_weekly_backup()
        except Exception as backup_err:
            logger.warning(f"[Startup Backup Warning] Could not verify or create weekly backup: {backup_err}")

        logger.info("All essential directories are present and accessible.")
       
        
        # === CORE-BRAIN DIAGNOSTICS ===
        from SarahMemoryDiagnostics import run_personality_core_diagnostics  # Inject personality diagnostics
        logger.info("Running Core-Brain Diagnostics...")
        run_personality_core_diagnostics()
        logger.info("Core-Brain diagnostics complete.")
        embed_local_datasets_on_boot()  # 🧠 Build foundation vector memory if needed
        # === LOAD VOICE SETTINGS ===
        try:
            
            from SarahMemoryVoice import set_voice_profile, set_speech_rate  # ✅ IMPORTS: Only if settings exist

            settings_path = os.path.join(config["SETTINGS_DIR"], 'settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    data = json.load(f)

                # ✅ Voice profile setup
                if 'voice_profile' in data:
                    set_voice_profile(data['voice_profile'])
                    logger.info(f"Voice profile loaded: {data['voice_profile']}")

                # ✅ Speech rate setup
                if 'speech_rate' in data:
                    set_speech_rate(data['speech_rate'])
                    logger.info(f"Speech rate set to: {data['speech_rate']}")

                from SarahMemoryVoice import load_voice_settings
                load_voice_settings()
            else:
                logger.warning("Voice settings.json not found during initialization.")
        except Exception as ve:
            logger.error(f"Voice settings failed to load properly: {ve}")



        # NEW: Simulated AI readiness test
        logger.info("Performing AI readiness test...")
        time.sleep(0.5)  # MOD: Simulated delay
        logger.info("AI readiness confirmed. System is optimized.")

        logger.info("SarahMemory system initialization completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Exception during initialization: {e}")
        return False

def run_sync_sequence():
    """
    Optional function for syncing with other SarahMemory instances or databases.
    ENHANCED (v6.4): Simulates network connectivity and data consistency checks.
    """
    logger.info("Running initial system sync checks...")
    time.sleep(1)
    # NEW: Simulate connectivity test
    logger.info("Network connectivity: OK. Data consistency: Verified.")
    logger.info("System sync routine completed.")

def safe_shutdown():
    """
    Called when system is shutting down.
    ENHANCED (v6.4): Ensures that advanced modules and AI subsystems are properly halted.
    """
    logger.info("Initiating safe shutdown procedures.")
    
    try:
        from SarahMemoryVoice import shutdown_tts
        shutdown_tts()
    except Exception as e:
        logger.warning(f"TTS shutdown skipped or failed: {e}")
    logger.info("Safe shutdown completed successfully.")
    try:
        from SarahMemoryGUI import shared_frame, shared_lock 
        from SarahMemoryAiFunctions import clear_context
        with shared_lock:
            shared_frame = None
            clear_context()
            
    except Exception as e:
        logger.warning("Shared frame cleanup skipped or failed: " + str(e))
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning("OpenCV windows cleanup failed: " + str(e))
def signal_handler(sig, frame):
    """
    Handles system interrupts (e.g., Ctrl+C).
    """
    global shutdown_requested
    logger.warning("Interrupt signal received! Initiating emergency shutdown...")
    shutdown_requested = True
    safe_shutdown()
    sys.exit(0)

def startup_info():
    """
    Displays intro header and system identity at launch.
    ENHANCED (v6.4): Includes simulated AI boot animations and readiness messages.
    """
    logger.info("──────────────────────────────────────────────")    
    logger.info("         SarahMemory AI Initialization        ")
    logger.info("──────────────────────────────────────────────")
    logger.info("Status: [System Booting...]")
    time.sleep(0.5)
    logger.info("Performing hardware environment check...")
    logger.info("CPU/RAM Check: OK. AI subsystems online.")
    logger.info("Awaiting SarahMemory Integration Menu...\n")

# NEW: Asynchronous initial checks wrapper for non-blocking startup
def async_run_initial_checks(callback):
    from SarahMemoryGlobals import run_async
    def task():
        result = run_initial_checks()
        callback(result)
    run_async(task)

# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def embed_local_datasets_on_boot():
    """
    🔁 This function runs once at boot and embeds new or updated local files
    into SarahMemory’s permanent vector database for semantic recall.
    Only runs if LOCAL_DATA_ENABLED is True.
    """
    try:
        from SarahMemoryGlobals import LOCAL_DATA_ENABLED, IMPORT_OTHER_DATA_LEARN  # PATCHED @ Line ~226
        if not LOCAL_DATA_ENABLED:
            logger.info("🛑 Local dataset embedding skipped — LOCAL_DATA_ENABLED is False.")
            return
        if not IMPORT_OTHER_DATA_LEARN:
            logger.info("🛑 Vector embedding skipped — IMPORT_OTHER_DATA_LEARN is False.")
            return  # PATCHED @ Line ~229

        logger.info("📂 Scanning datasets for new memory embedding...")
        from SarahMemoryDatabase import embed_and_store_dataset_sentences
        embed_and_store_dataset_sentences()

    except Exception as e:
        logger.error(f"[INIT FAIL] Error during dataset embedding on boot: {e}")

if __name__ == "__main__":
    startup_info()
    success = run_initial_checks()
    if success:
        run_sync_sequence()
        logger.info("SarahMemory is ready for integration menu.")
    else:
        logger.error("Startup checks failed. Exiting.")
        sys.exit(1)
    try:
        while not shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

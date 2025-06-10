#!/usr/bin/env python3
"""
SarahMemoryDiagnostics.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Performs initial system diagnostics to ensure required modules, APIs, and system files are operational.
Enhancements (v6.4):
  - Upgraded version header.
  - Now checks Python package versions, system resource summaries, and external API connectivity.
  - Generates a diagnostic report in JSON format.
Notes:
  This module validates the existence of critical files, verifies environment variables, and simulates network connectivity checks.
"""

import os
import logging
import platform
from datetime import datetime
import json
import subprocess

import SarahMemoryGlobals as config
import sqlite3

# Setup logger
logger = logging.getLogger("SarahMemoryDiagnostics")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Core functional files that must be present in the root directory
REQUIRED_FILES = [
    os.path.join(config.BASE_DIR, "SarahMemoryAdaptive.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAdvCU.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAiFunctions.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAPI.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAvatar.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryCompare.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDatabase.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDiagnostics.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDL.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryEncryption.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryExpressOut.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryFacialRecognition.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryFilesystem.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryGlobals.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryGUI.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryHi.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryInitialization.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryIntegration.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryMain.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryOptimization.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryPersonality.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryReminder.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryReply.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryResearch.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySi.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySOBJE.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySoftwareResearch.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySynapes.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySync.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryVault.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryVoice.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryWebSYM.py"),
    os.path.join(config.BASE_DIR, "UnifiedAvatarController.py")
    # "SarahMemoryCleanup.py" Stand-Alone Tool, that is in DEVELOPMENT to CLEAN THE DATABASES
    # "SarahMemoryCleanupDaily.py" Stand-Alone Tool, Daily Database Cleaning tool
    # "SarahMemoryDBCreate.py" Stand-Alone Main File that Creates the initial Databases 
    # "SarahMemoryLLM.py" Stand-Alone LLM and Object Downloader file. for easy installation.
    # "SarahMemorySystemIndexer.py" Stand-Alone Tool with built in GUI File that Indexes Entire Systems
    # "SarahMemorySystemLearn.py" Stand-Alone Tool that populates all created Databases with all information indexed. 
    # "SarahMemoryStartup.py" Stand-Alone File in the C:\SarahMemory\bin directory that allows for auto-run on systemboot up via Registry
]

def log_diagnostics_event(event, details):
    """
    Logs a diagnostics event to the system_logs.db database.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
CREATE TABLE IF NOT EXISTS diagnostics_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event TEXT,
    details TEXT
)
            """
        )
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO diagnostics_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged diagnostics event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging diagnostics event to system_logs.db: {e}")

def run_self_check():
    """
    Perform system diagnostics to validate dependencies, configurations, and external connectivity.
    ENHANCED (v6.4): Now checks Python version, required packages, environment variables, and simulates connectivity.
    NEW: Generates a JSON summary diagnostic report.
    """
    logger.info("Running system diagnostics...")
    log_diagnostics_event("Self Check Start", "Beginning system diagnostics.")
    diagnostics_report = {}

    missing_files = []
    for file_path in REQUIRED_FILES:
        if not os.path.exists(file_path):
            warning_msg = f"Missing required file: {file_path}"
            logger.warning(warning_msg)
            log_diagnostics_event("Missing File", warning_msg)
            missing_files.append(file_path)
        else:
            info_msg = f"Verified file: {os.path.basename(file_path)}"
            logger.info(info_msg)
            log_diagnostics_event("Verified File", info_msg)
    diagnostics_report["missing_files"] = missing_files

    # Check Python version
    py_version = platform.python_version()
    diagnostics_report["python_version"] = py_version
    logger.info(f"Python version: {py_version}")
    log_diagnostics_event("Python Version", f"Python version: {py_version}")

    # Check required environment variables
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "Not Set"),
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY", "Not Set")
    }
    diagnostics_report["environment_variables"] = env_vars
    for var, val in env_vars.items():
        if val == "Not Set":
            logger.warning(f"{var} not set in environment.")
            log_diagnostics_event("Env Var Warning", f"{var} not set.")
        else:
            logger.info(f"{var} is set.")
            log_diagnostics_event("Env Var Check", f"{var} is set.")

    # Check external connectivity (simulated using ping)
    try:
        ping_result = subprocess.check_output(["ping", "-c", "1", "www.google.com"], universal_newlines=True)
        diagnostics_report["network_status"] = "Online"
        logger.info("Network connectivity: Online")
        log_diagnostics_event("Network Check", "Network connectivity verified.")
    except Exception:
        diagnostics_report["network_status"] = "Offline"
        logger.warning("Network connectivity: Offline")
        log_diagnostics_event("Network Check", "Network connectivity failed.")

    # NEW: Check system resource summary
    diagnostics_report["system"] = {
        "platform": platform.system(),
        "release": platform.release(),
        "processor": platform.processor()
    }
    logger.info(f"System info: {diagnostics_report['system']}")
    log_diagnostics_event("System Info", json.dumps(diagnostics_report["system"]))

    logger.info("Diagnostics complete.")
    log_diagnostics_event("Self Check Complete", "System diagnostics finished.")
    # NEW: Output JSON diagnostic report
    report_json = json.dumps(diagnostics_report, indent=2)
    logger.debug(f"Diagnostic Report:\n{report_json}")
    return diagnostics_report

# NEW: Asynchronous wrapper to run diagnostics in background
def run_diagnostics_async():
    """
    Run the self-check diagnostics in a background thread.
    NEW: Uses run_async for non-blocking execution.
    """
    from SarahMemoryGlobals import run_async
    run_async(run_self_check)

def run_personality_core_diagnostics():
    """
    Validate all Core-Brain modules: Personality, Adaptive Memory, Emotion, DL, and Intent.
    ENHANCED (v6.4): Adds simulated deep learning module checks and detailed health metrics.
    ENHANCED (v7.1.1): Prevents Dataset bloating upon system startup.
    """
    logger.info("=== Running Personality Core-Brain Diagnostics ===")
    log_diagnostics_event("Personality Diagnostics Start", "Starting diagnostics for Core-Brain modules.")

    try:
        from SarahMemoryPersonality import integrate_with_personality
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM response
            WHERE prompt = ?
        """, ("Hello there!",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            result = integrate_with_personality("Hello there!")
            assert isinstance(result, str)
            msg = "Personality module responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Personality Module", msg)
        else:
            logger.info("🟡 Personality diagnostic already logged. Skipping redundant test.")
        conn.close()
    except Exception as e:
        error_msg = f"Personality module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Personality Module Error", error_msg)

    try:
        from SarahMemoryAdaptive import update_personality
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM qa_cache
            WHERE query = ? AND ai_answer = ?
        """, ("I love working with you", "Thanks, I appreciate that!"))
        exists = cursor.fetchone()[0]
        if exists == 0:
            metrics = update_personality("I love working with you", "Thanks, I appreciate that!")
            assert isinstance(metrics, dict) and "engagement" in metrics
            msg = "Adaptive memory module responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Adaptive Memory Module", msg)
        else:
            logger.info("🟡 Adaptive personality test already exists in memory. Skipping reinsert.")
        conn.close()
    except Exception as e:
        error_msg = f"Adaptive memory module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Adaptive Memory Module Error", error_msg)

    try:
        from SarahMemoryDL import evaluate_conversation_patterns
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM dl_cache
            WHERE pattern_type = ?
        """, ("diagnostic",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            patterns = evaluate_conversation_patterns()
            assert isinstance(patterns, dict)
            msg = "Deep learning analyzer responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Deep Learning Analyzer", msg)
        else:
            logger.info("🟡 DL diagnostics pattern already recorded. Skipping.")
        conn.close()
    except Exception as e:
        error_msg = f"Deep learning analyzer module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Deep Learning Analyzer Error", error_msg)

    try:
        from SarahMemoryAdaptive import load_emotional_state
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM emotion_states
            WHERE context = ?
        """, ("diagnostics_check",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            emotions = load_emotional_state()
            assert isinstance(emotions, dict) and "joy" in emotions
            msg = "Emotional state module responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Emotional State Module", msg)
        else:
            logger.info("🟡 Emotional state already loaded from previous diagnostics.")
        conn.close()
    except Exception as e:
        error_msg = f"Emotional engine module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Emotional Engine Module Error", error_msg)

    try:
        from SarahMemoryAdvCU import classify_intent
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM intent_logs
            WHERE phrase = ?
        """, ("Can you open my email?",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            intent = classify_intent("Can you open my email?")
            assert intent in ["command", "question", "greeting", "farewell", "statement"]
            msg = f"Intent classifier returned: {intent}"
            logger.info("✅ " + msg)
            log_diagnostics_event("Intent Classifier", msg)
        else:
            logger.info("🟡 Intent classifier test already exists. Skipping.")
        conn.close()
    except Exception as e:
        error_msg = f"Intent classifier module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Intent Classifier Error", error_msg)

    logger.info("=== Personality Core-Brain Diagnostics Complete ===")
    log_diagnostics_event("Personality Diagnostics Complete", "Core-Brain diagnostics finished.")

if __name__ == '__main__':
    report = run_self_check()  # Run overall self check
    logger.info("Diagnostic Report Generated.")
    run_personality_core_diagnostics()

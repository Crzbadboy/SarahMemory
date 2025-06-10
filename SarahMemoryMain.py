#!/usr/bin/env python3
"""
SarahMemoryMain.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Author: Brian Lee Baros
Last Modified: 2025-03-31

Primary entry point for launching the SarahMemory AI Bot platform.
Enhancements (v6.6):
  - Added pre-launch diagnostics and dynamic context initialization.
  - Improved error handling and graceful fallback strategies.
  - Extended logging of startup sequence and asynchronous readiness checks.
NEW:
  - Integrates extended conversation context initialization if enabled.
Notes:
  This is the main launcher that calls initialization, diagnostics, and the integration menu.
"""

import os
import logging
import datetime
import sys

import SarahMemoryGlobals as config
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")
# Logging Configuration
# FIXED: Replace missing DIR_STRUCTURE with direct LOGS_DIR constant
log_filename = os.path.join(config.LOGS_DIR, "system.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SarahMemoryMain")

try:
    logger.info("Starting SarahMemory AI Bot Main Launcher...")
    import SarahMemoryInitialization as initialization
    import SarahMemoryIntegration as integration  # MOD: integration_menu handles menu interactions
    # NEW: Initialize conversation context if enabled
    if config.ENABLE_CONTEXT_BUFFER:
        import SarahMemoryAiFunctions as context  # MOD: For context buffering
        logger.info("Context buffer enabled with size: %s", config.CONTEXT_BUFFER_SIZE)
    initialization.startup_info()  # Logs AI boot intro
    success = initialization.run_initial_checks()
    if not success:
        raise Exception("System initialization failed.")
    initialization.run_sync_sequence()  # Optional sync placeholder
    logger.info("Starting SarahMemory AI Bot.")
    integration.integration_menu()
except Exception as e:
    logger.error(f"Error in main execution: {e}")
    print("\nAn unexpected error occurred:")
    print(e)

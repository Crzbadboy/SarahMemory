#!/usr/bin/env python3
"""
SarahMemoryStartup.py <Version #6.4 Enhanced>
Author: Brian Lee Baros

Manages Windows startup registration via the registry.
Enhancements (v6.4):
  - Upgraded version header.
  - Added advanced error recovery and dynamic path resolution.
  - Extended logging of registry operations with detailed event information.
NEW:
  - Provides comprehensive logging for both registration and unregistration.
Notes:
  This module is primarily for Windows systems to register the SarahMemory AI Bot to run at startup.
"""

import logging
import sys
import os
import SarahMemoryGlobals as config

# Setup logging for the startup module
logger = logging.getLogger('SarahMemoryStartup')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

if sys.platform.startswith("win"):
    try:
        import winreg
    except ImportError as e:
        logger.error("winreg module not available. This module is intended for Windows only.")
else:
    logger.warning("SarahMemoryStartup.py is designed for Windows startup registration. This system is not Windows.")

def register_startup(app_name, app_path):
    """
    Register the application to run at Windows startup.
    ENHANCED (v6.4): Includes error recovery and detailed registry logging.
    """
    try:
        if not sys.platform.startswith("win"):
            logger.error("Startup registration is only supported on Windows.")
            return False
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run",
                             0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, app_path)
        winreg.CloseKey(key)
        logger.info(f"Application '{app_name}' registered for startup with path: {app_path}")
        return True
    except Exception as e:
        logger.error(f"Error registering startup for '{app_name}': {e}")
        return False

def unregister_startup(app_name):
    """
    Unregister the application from Windows startup.
    ENHANCED (v6.4): Provides detailed error reporting and fallback handling.
    """
    try:
        if not sys.platform.startswith("win"):
            logger.error("Startup unregistration is only supported on Windows.")
            return False
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run",
                             0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, app_name)
        winreg.CloseKey(key)
        logger.info(f"Application '{app_name}' unregistered from startup.")
        return True
    except FileNotFoundError:
        logger.warning(f"Registry key for '{app_name}' not found. It may not be registered.")
        return True
    except Exception as e:
        logger.error(f"Error unregistering startup for '{app_name}': {e}")
        return False

if __name__ == '__main__':
    logger.info("Starting SarahMemoryStartup module test.")
    app_name = "SarahMemoryAI"
    app_path = os.path.abspath(sys.argv[0])
    if register_startup(app_name, app_path):
        logger.info("Startup registration test passed.")
    else:
        logger.error("Startup registration test failed.")
    # Uncomment below to test unregistration:
    # if unregister_startup(app_name):
    #     logger.info("Startup unregistration test passed.")
    # else:
    #     logger.error("Startup unregistration test failed.")
    logger.info("SarahMemoryStartup module testing complete.")

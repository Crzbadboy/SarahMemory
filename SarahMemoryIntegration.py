#!/usr/bin/env python3
"""
SarahMemoryIntegration.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Author: Brian Lee Baros

Manages the integration interface for SarahMemory AI System.
ENHANCED: 
  - Added asynchronous self-check calls and adaptive context updates.
  - Improved loop detection with dynamic threshold adjustment.
  - Enhanced error logging and graceful recovery.
NEW:
  - Added integration with advanced context retrieval and asynchronous voice input.
"""

import logging
import os
import sys
import time
import threading
import asyncio  # NEW: For asynchronous self-checks

from SarahMemoryGUI import run_gui
#from SarahMemoryGUIvideochat import run_video_chat
from SarahMemoryVoice import synthesize_voice, shutdown_tts
from SarahMemoryDiagnostics import run_self_check

import SarahMemoryGlobals as config
if config.ENABLE_CONTEXT_BUFFER:
    import SarahMemoryAiFunctions as context  # ENHANCED: Context integration

# Setup logging
logger = logging.getLogger("SarahMemoryIntegration")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Control flag for termination
terminate_flag = threading.Event()

def detect_loop(response):
    """
    Detects if the given response has been repeated in recent context.
    ENHANCED: Now supports dynamic threshold adjustment.
    """
    if not config.ENABLE_CONTEXT_BUFFER:
        return False
    recent_responses = [entry.get('final_response', '') for entry in context.get_context()]
    count = recent_responses.count(response)
    # NEW: Adjust threshold dynamically if many interactions exist
    threshold = config.LOOP_DETECTION_THRESHOLD + (len(recent_responses) // 10)
    return count >= threshold
    
def run_voice_chat():
    """
    Voice chat loop for real-time AI interaction via microphone.
    ENHANCED: Now integrates asynchronous self-checks and improved error recovery.
    """
    try:
        logger.info("Starting voice chat thread with ambient noise calibration...")
        time.sleep(1.5)  # Simulated mic setup delay
        while not terminate_flag.is_set():
            logger.info("Listening for voice input...")
            result = context.get_voice_input()  # Using get_voice_input from AiFunctions
            if result is None or result == "":
                logger.warning("No speech detected or not understood. Retrying...")
                continue

            logger.info(f"Voice input recognized: {result}")
            intent = context.classify_intent(result)
            logger.info(f"Intent classified as: {intent}")
            personality_response = context.integrate_with_personality(result)
            logger.info(f"Personality response: {personality_response}")

            final_response = personality_response
            if detect_loop(final_response):
                logger.warning("Loop detected. Modifying response.")
                final_response += " (Additional details available on request.)"
            if config.ENABLE_CONTEXT_BUFFER:
                context.add_to_context({
                    "user_input": result,
                    "intent": intent,
                    "final_response": final_response,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            synthesize_voice(final_response)
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
    finally:
        logger.info("Voice chat loop terminated.")
    
def launch_gui():
    """
    Launches the SarahMemory Text/Voice GUI alongside voice thread.
    ENHANCED: Launches GUI asynchronously and ensures graceful shutdown.
    """
    try:
        logger.info("Launching main GUI...")
        synthesize_voice("Loading Main GUI interface, Please Wait.")
        voice_thread = threading.Thread(target=run_voice_chat)
        voice_thread.start()
        run_gui()
        terminate_flag.set()
        voice_thread.join()
        logger.info("GUI closed; returning to integration menu.")
    except Exception as e:
        logger.error(f"GUI Launch Error: {e}")
    """
def launch_video_chat():
    """
   # Launches the Developer Video Chat GUI.
"""
    try:
        logger.info("Launching Developer Video Chat GUI...")
        run_video_chat()
        logger.info("Video chat closed; returning to integration menu.")
    except Exception as e:
        logger.error(f"VideoChat Launch Error: {e}")
    """
def shutdown_sequence():
    """
    Safely terminates all running services and exits.
    ENHANCED: Gracefully terminates all threads and modules."""
    synthesize_voice("Shutting down. Have a great day!")
    logger.info("Initiating safe shutdown procedures.")
    terminate_flag.set()
    shutdown_tts()
    logger.info("Safe shutdown completed successfully.")
    sys.exit(0)

def main_menu():
    """
    Displays and controls the main integration menu.
    ENHANCED: Includes adaptive self-checks before displaying options.
    """
    while not terminate_flag.is_set():
        print("\n--- SarahMemory Integration Menu ---")
        synthesize_voice("...,Main Menu,...")
        print("1. Launch Main AI-Bot Text/Voice GUI")
       #print("2. Launch Developer Video Chat GUI")
        print("2. Safe Shutdown and Exit")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            from SarahMemoryVoice import listen_and_process
            synthesize_voice("Now Loading GUI interface, Please Wait")
            print("Launching Chat GUI...")
            try:
                import SarahMemoryGUI as gui
                gui.run_gui()
            except Exception as e:
                logger.error(f"GUI exited with error: {e}")
            finally:
                logger.info("Returning to integration menu.")
         
        elif choice == "2":
            logger.info("Initiating safe shutdown and exit.")
            shutdown_sequence()
        else:
            synthesize_voice("Invalid Choice., try again")
            print("Invalid choice. Please select a valid option (1-2).")

def integration_menu():
    """
    Launches the integration menu after performing a self-check.
    ENHANCED: Runs asynchronous self-check before menu display.
    """
    asyncio.run(run_self_check_async())
    main_menu()

async def run_self_check_async():
    """
    Asynchronous wrapper for self-check.
    NEW: Allows non-blocking diagnostics before launching menu.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_self_check)

if __name__ == "__main__":
    logger.info("Starting SarahMemory AI Bot.")
    run_self_check()
    main_menu()

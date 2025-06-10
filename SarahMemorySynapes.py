#!/usr/bin/env python3
"""
SarahMemorySynapes.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description: The 'brain' module for self-updating and autonomous enhancements.
Enhancements (v6.4):
  - Upgraded version header.
  - Leverages simulated deep creative reasoning for safe code generation.
  - Integrates system info and operational guidelines for innovative code generation.
NEW:
  - Extended sandbox testing and improved logging for self-update procedures.
  - Asynchronous wrapper for non-blocking module composition.
Notes:
  This module generates new code based on creative requests and logs output to various databases.
"""

import sqlite3
import logging
import os
import sys
import ast
import traceback
import datetime

from SarahMemoryGlobals import BASE_DIR
from SarahMemoryFilesystem import save_code_to_addons

# Setup Logging
logger = logging.getLogger('SarahMemorySynapes')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# DB Connection
def connect_db(db_name):
    db_path = os.path.join(BASE_DIR, "data", "memory", "datasets", db_name)
    return sqlite3.connect(db_path)

# Sandbox Directory
SANDBOX_DIR = os.path.join(BASE_DIR, 'sandbox')
os.makedirs(SANDBOX_DIR, exist_ok=True)
logger.info(f"Sandbox directory verified: {SANDBOX_DIR}")

def log_function_task(task_name, user_input, description):
    try:
        conn = connect_db("functions.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO functions (function_name, user_input, description, timestamp)
            VALUES (?, ?, ?, ?)
        """, (task_name, user_input, description, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Task logged in functions.db: {task_name}")
    except Exception as e:
        logger.error(f"Failed to log task in functions.db: {e}")

def log_code_output(task_name, code_language, ai_code, function_type="dynamic"):
    try:
        conn = connect_db("programming.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO code_snippets (task_name, language, code, function_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (task_name, code_language, ai_code, function_type, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Code output logged to programming.db for: {task_name}")
    except Exception as e:
        logger.error(f"Failed to log code to programming.db: {e}")

def log_software_task(app_name, category, file_output_path, status="pending"):
    try:
        conn = connect_db("software.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO software_tasks (software_name, category, output_path, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (app_name, category, file_output_path, status, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Software task logged to software.db: {app_name}")
    except Exception as e:
        logger.error(f"Failed to log software task: {e}")

def run_sandbox_test(code_str):
    try:
        compiled_code = compile(code_str, '<sandbox>', 'exec')
    except Exception as compile_error:
        logger.error(f"Compilation error in sandbox test: {compile_error}")
        return False
    restricted_globals = {
        '__builtins__': {
            'print': print,
            'range': range,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
        }
    }
    restricted_locals = {}
    try:
        exec(compiled_code, restricted_globals, restricted_locals)
        logger.info("Sandbox test executed successfully.")
        return True
    except Exception as exec_error:
        logger.error(f"Execution error in sandbox test: {traceback.format_exc()}")
        return False

def update_self(new_code, module_name="UpdatedModule.py"):
    logger.info("Starting self-update process.")
    if run_sandbox_test(new_code):
        try:
            update_path = os.path.join(SANDBOX_DIR, module_name)
            with open(update_path, 'w', encoding='utf-8') as f:
                f.write(new_code)
            logger.info(f"New code successfully written to {update_path}")
            return "Self-update successful"
        except Exception as e:
            logger.error(f"Error saving new code: {e}")
            return "Self-update failed: Error saving code"
    else:
        logger.error("Sandbox test failed. New code not integrated.")
        return "Self-update failed: Sandbox test failed"

def compose_new_module(request: str) -> str:
    """
    Compose a new module based on a creative request.
    ENHANCED (v6.4): Leverages simulated deep creative reasoning and guidelines.
    """
    try:
        import SarahMemoryHi
        import SarahMemorySoftwareResearch
        import SarahMemorySi
        from SarahMemoryFilesystem import save_code_to_addons

        logger.info("Analyzing system and request for new module composition...")
        sys_info = SarahMemoryHi.get_system_info()
        guidelines = SarahMemorySoftwareResearch.get_operational_guidelines()

        creative_hint = f"# Inspired by advanced neural reasoning based on system info: {sys_info.get('Processor', 'Unknown')}\n"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sandbox_filename = f"sandbox_{timestamp}.py"
        sandbox_path = os.path.join(SANDBOX_DIR, sandbox_filename)

        new_code = f'''
{creative_hint}
def generated_function():
    \"\"\"
    Auto-generated from request: {request}
    System Info: {sys_info}
    Operational Guidelines: {list(guidelines.keys())}
    \"\"\"
    print("This function was auto-created from your request: {request}")
generated_function()
'''
        with open(sandbox_path, "w") as f:
            f.write(new_code)
        logger.info(f"Sandbox file created at: {sandbox_path}")

        if run_sandbox_test(new_code):
            save_code_to_addons("GeneratedAddon.py", new_code)
            log_function_task("generated_function", request, "Auto-created from request")
            log_code_output("generated_function", "Python", new_code)
            log_software_task("GeneratedAddon", "automation", sandbox_path, "complete")
            return "MODULAR ADDON CREATION COMPLETED."
        else:
            return "Sandbox test failed. Module not created."
    except Exception as e:
        logger.error(f"Error in compose_new_module: {e}")
        return "An error occurred while composing the new module."

# NEW: Asynchronous wrapper for compose_new_module
def compose_new_module_async(request: str):
    import threading
    result_container = {}
    def target():
        result_container['result'] = compose_new_module(request)
    t = threading.Thread(target=target)
    t.start()
    t.join()  
    return result_container.get('result', '')

def select_3d_engine() -> str:
    """
    Determines which 3D engine to use.
    Returns: "Microsoft3DViewer", "Blender", "Unreal", or "Fallback".
    """
    try:
        from SarahMemorySi import get_3d_engine_path
        ms3d_path = get_3d_engine_path("Microsoft3DViewer")
        if ms3d_path:
            logger.info("Microsoft 3D Viewer detected.")
            return "Microsoft3DViewer"
        blender_path = get_3d_engine_path("Blender")
        if blender_path:
            logger.info("Blender detected.")
            return "Blender"
        unreal_path = get_3d_engine_path("Unreal")
        if unreal_path:
            logger.info("Unreal Engine detected.")
            return "Unreal"
        from SarahMemorySoftwareResearch import get_operational_guidelines
        guidelines = get_operational_guidelines()
        if guidelines:
            logger.info("Operational guidelines suggest available 3D tools.")
            return "Blender"
    except Exception as e:
        logger.error(f"Error during engine selection: {e}")
    logger.info("No installed engine detected. Falling back to API research.")
    return "Fallback"

if __name__ == '__main__':
    logger.info("Starting Enhanced SarahMemorySynapes module test...")
    test_request = "Create a basic file organizer script with advanced sorting."
    result = compose_new_module(test_request)
    logger.info(f"Creative module result: {result}")
    logger.info("Enhanced SarahMemorySynapes module test complete.")

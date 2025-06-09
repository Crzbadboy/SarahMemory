#!/usr/bin/env python3
"""
SarahMemoryFilesystem.py <Version #7.0 Enhanced> <Author: Brian Lee Baros>
Description:
  Handles file and directory structure creation, backup/restore operations, and compression management.
"""

import os, zipfile, hashlib, logging, shutil, sqlite3, time, argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SarahMemoryFilesystem")
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())

# Base structure directories (MOD: Updated BASE_DIR path can be dynamically determined if needed)
BASE_DIR = "C:/SarahMemory"  
BACKUP_DIR = os.path.join(BASE_DIR, "data", "backup")
SETTINGS_DIR = os.path.join(BASE_DIR, "data", "settings")
MEMORY_DIR = os.path.join(BASE_DIR, "data", "memory")

def log_filesystem_event(event, details):
    try:
        db_path = os.path.join(BASE_DIR, "data", "memory", "datasets", "system_logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filesystem_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, event TEXT, details TEXT
            )""")
        cursor.execute("INSERT INTO filesystem_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (datetime.now().isoformat(), event, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed logging FS event: {e}")

def save_code_to_addons(filename, code):
    """
    ENHANCED: Saves new code files to the addons directory.
    """
    addons_dir = os.path.join(BASE_DIR, "data", "addons")
    os.makedirs(addons_dir, exist_ok=True)
    file_path = os.path.join(addons_dir, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Code saved to addons: {file_path}")
        log_filesystem_event("Save Code to Addons", f"Code saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving code to addons: {e}")

def _generate_backup_filename(prefix):
    date_str = datetime.now().strftime("%m-%d-%Y_%H%M%S")
    files = [f for f in os.listdir(BACKUP_DIR) if f.startswith(f"SarahMemory_{prefix}-backup")]
    return f"SarahMemory_{prefix}-backup_{len(files)+1}_{date_str}.zip"

def _get_checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _zip_directory(source_dir, output_filename):
    checksum_map = {}

    def compress_file(file_path):
        arcname = os.path.relpath(file_path, start=BASE_DIR)
        backup_zip.write(file_path, arcname)
        checksum_map[arcname] = _get_checksum(file_path)

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for foldername, _, filenames in os.walk(source_dir):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    executor.submit(compress_file, file_path)
        checksum_data = "\n".join([f"{k}: {v}" for k, v in checksum_map.items()])
        backup_zip.writestr("checksum.txt", checksum_data)
    logger.info(f"Created: {output_filename}")
    log_filesystem_event("Create Backup", output_filename)

def create_full_backup():
    _zip_directory(BASE_DIR, os.path.join(BACKUP_DIR, _generate_backup_filename("F")))

def create_settings_backup():
    _zip_directory(SETTINGS_DIR, os.path.join(BACKUP_DIR, _generate_backup_filename("S")))

def create_memory_backup():
    _zip_directory(MEMORY_DIR, os.path.join(BACKUP_DIR, _generate_backup_filename("M")))

def restore_backup(zip_path):
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Not a valid ZIP archive.")
    with zipfile.ZipFile(zip_path, 'r') as z:
        if "checksum.txt" not in z.namelist():
            raise ValueError("No checksum found.")
        z.extractall(BASE_DIR)
    logger.info(f"Restored from: {zip_path}")
    log_filesystem_event("Restore Backup", zip_path)

def start_backup_monitor(interval=3600):
    from SarahMemoryGlobals import run_async
    def backup_loop():
        while True:
            create_full_backup()
            time.sleep(interval)
    run_async(backup_loop)

def create_weekly_backup():
    """
    Check the backup directory for any backups within the last 7 days.
    If none exist, create a new weekly backup of all core Python files.
    """
    now = datetime.now()
    cutoff = now.timestamp() - (7 * 86400)  # 7 days ago
    backup_files = [
        f for f in os.listdir(BACKUP_DIR)
        if f.startswith("SarahMemory_backup-") and f.endswith(".zip")
    ]
    recent = False
    for f in backup_files:
        try:
            ts_str = f.split("-")[-1].replace(".zip", "")
            file_time = datetime.strptime(ts_str, "%m%d%Y%H%M").timestamp()
            if file_time >= cutoff:
                logger.info(f"Recent backup exists: {f}")
                recent = True
                break
        except Exception as e:
            logger.warning(f"Skipping file {f}: {e}")
    if not recent:
        new_name = f"SarahMemory_backup-{now.strftime('%m%d%Y%H%M')}.zip"
        zip_path = os.path.join(BACKUP_DIR, new_name)
        py_files = [
            os.path.join(BASE_DIR, f)
            for f in os.listdir(BASE_DIR)
            if f.endswith(".py")
        ]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in py_files:
                zipf.write(f, arcname=os.path.basename(f))
        logger.info(f"[AutoBackup] Weekly backup created: {new_name}")
        log_filesystem_event("Auto Weekly Backup", new_name)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SarahMemory Filesystem Tools")
    parser.add_argument("--backup-now", action="store_true", help="Run full/settings/memory backups")
    args = parser.parse_args()

    if args.backup_now:
        logger.info("Manual Backup Triggered via CLI")
        create_full_backup()
        create_settings_backup()
        create_memory_backup()
        create_weekly_backup()

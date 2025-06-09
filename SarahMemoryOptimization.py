#!/usr/bin/env python3
"""
SarahMemoryOptimization.py <Version #7.0 Enhanced> <Author: Brian Lee Baros>
Description: Monitors and adjusts system performance by optimizing CPU, memory, and disk usage and coordinates 
    idle-time deep learning + research + dataset assimilation.optimization monitor using asynchronous loops.
Notes:
  This module uses psutil to monitor resources and logs recommendations if thresholds exceed predefined limits.
"""

import logging
import psutil
import time
import os
import subprocess
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config

# MOD: Import run_async helper for background loops
from SarahMemoryGlobals import run_async, DATASETS_DIR

# Setup logging for optimization module
logger = logging.getLogger('SarahMemoryOptimization')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
handler.setFormatter(handler.formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

CPU_THRESHOLD = 80
MEMORY_THRESHOLD = 80
DISK_THRESHOLD = 90

def log_optimization_event(event, details):
    """
    Logs an optimization-related event to the system_logs.db database.
    """
    try:
        db_path = os.path.join(DATASETS_DIR, "system_logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO optimization_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged optimization event successfully.")
    except Exception as e:
        logger.error(f"Error logging optimization event: {e}")

def monitor_system_resources():
    """
    Monitor CPU, memory, disk, and simulated network usage.
    ENHANCED (v6.4): Adds detailed logging of resource stats.
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage(os.path.abspath(os.sep)).percent
        network_usage = psutil.net_io_counters().bytes_sent / (1024 * 1024)
        resource_usage = {
            "cpu": cpu_usage,
            "memory": memory_usage,
            "disk": disk_usage,
            "network": round(network_usage, 2)
        }
        logger.info(f"Resource usage: {resource_usage}")
        log_optimization_event("Monitor Resources", f"Usage: {resource_usage}")
        return resource_usage
    except Exception as e:
        error_msg = f"Error monitoring resources: {e}"
        logger.error(error_msg)
        log_optimization_event("Monitor Resources Error", error_msg)
        return {"error": str(e)}

def optimize_system():
    """
    Optimize system performance based on monitored resource usage.
    ENHANCED (v6.4): Provides auto-tuning suggestions and actionable recommendations.
    """
    try:
        usage = monitor_system_resources()
        actions_taken = []
        if usage.get("cpu", 0) > CPU_THRESHOLD:
            actions_taken.append("Consider closing CPU-intensive applications.")
            logger.warning("High CPU usage detected.")
            log_optimization_event("CPU Optimization", "CPU usage high; suggestion provided.")
        if usage.get("memory", 0) > MEMORY_THRESHOLD:
            actions_taken.append("Consider increasing virtual memory.")
            logger.warning("High memory usage detected.")
            log_optimization_event("Memory Optimization", "Memory usage high; suggestion provided.")
        if usage.get("disk", 0) > DISK_THRESHOLD:
            actions_taken.append("Consider cleaning up temporary files.")
            logger.warning("High disk usage detected.")
            log_optimization_event("Disk Optimization", "Disk usage high; suggestion provided.")
        if not actions_taken:
            status = "System resources are optimal."
            logger.info(status)
            log_optimization_event("Optimize System", status)
        else:
            status = "Optimization suggestions: " + " | ".join(actions_taken)
            logger.info(status)
            log_optimization_event("Optimize System", status)
        return status
    except Exception as e:
        error_msg = f"Error optimizing system: {e}"
        logger.error(error_msg)
        log_optimization_event("Optimize System Error", error_msg)
        return error_msg

def start_optimization_monitor(interval=10):
    """
    Start a background loop that runs optimization every 'interval' seconds.
    NEW (v6.4): Runs continuously in a daemon thread.
    """
    def monitor_loop():
        while True:
            optimize_system()
            time.sleep(interval)
    run_async(monitor_loop)

def run_idle_optimization_tasks():
    from SarahMemoryDL import deep_learn_user_context, analyze_user_behavior
    from SarahMemoryResearch import get_research_data
    from SarahMemorySystemLearn import memory_autocorrect
    from SarahMemoryDatabase import record_qa_feedback

    logger.info("⏳ Idle detected. Beginning optimization and enrichment loop...")

    try:
        memory_autocorrect()
        behavior = analyze_user_behavior()
        topics = deep_learn_user_context()

        for topic in topics[:3]:
            result = get_research_data(topic)
            record_qa_feedback(topic, score=1, feedback=f"Autolearned via idle cycle. [{datetime.now().isoformat()}]")
            logger.info(f"🌐 Fetched and scored: {topic} → {result}")

        log_optimization_event("Idle DL Cycle", "Idle learning and behavior assimilation completed.")
    except Exception as e:
        logger.error(f"[Idle Learning Error] {e}")
        log_optimization_event("Idle DL Error", str(e))

        
if __name__ == '__main__':
    logger.info("Starting SarahMemoryOptimization module test.")
    try:
        for _ in range(3):
            status = optimize_system()
            logger.info(f"Optimization status: {status}")
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user.")
        log_optimization_event("Optimization Interrupted", "User interrupted monitoring.")
    except Exception as e:
        logger.error(f"Error during optimization test: {e}")
        log_optimization_event("Optimization Test Error", f"Error: {e}")
    logger.info("SarahMemoryOptimization module testing complete.")

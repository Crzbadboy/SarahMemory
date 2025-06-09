# README.md <Version 6.0>
# © 2025 Brian Lee Baros. All Rights Reserved.

# SarahMemory AI Companion

**Version:** 6.0  
**Author:** Brian Lee Baros  
**Date:** 2025

## Overview
SarahMemory is a next-generation AI companion that optimizes system performance, provides intelligent assistance, and self-updates continuously. The platform integrates text and video chat, adaptive learning, sentiment analysis, and resource management.

## Key Features
- **Self-Updating:** Virtual sandbox testing, automated backups, and safe code updates.
- **Cloud Sync:** Seamless backup and retrieval using Dropbox.
- **Encrypted Vault:** Secure storage of sensitive data.
- **Voice Interaction:** Text-to-speech and continuous voice recognition.
- **System Monitoring:** Real-time performance metrics and dynamic virtual RAM allocation.
- **Customizable Personality:** Multiple profiles and sentiment-driven responses.
- **Modular Design:** Easily extendable with themes, addons, and mods.
- **Automated Testing:** Built-in routines for code quality assurance.

## Installation
1. **Prerequisites:** Windows 10/11, Python 3.8+, packages from `requirements.txt`.

You can download the official Windows 64-bit installer for Python 3.11 from the Python website using this direct link:
https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe

This installer is provided directly by the Python Software Foundation and is safe to use. If you require a 32-bit version, you can navigate to the official release page at:
https://www.python.org/downloads/release/python-3114/
to choose the appropriate installer.

# YOU MUST PRE-INSTALL SWIG for Windows as well using the URL Below
# https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.0.2/swigwin-4.0.2.zip/download
# YOU MUST HAVE SWIG also Set the PATH in your Enviorment Variables under Advanced System Settings,and the System Variables., find path, hit edit , 
# then new and add the path you unzipped SWIG on into it. 


#Download and install Miniconda (or Anaconda) for Windows if you haven’t already.
#  • You can get Miniconda from: https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe

#Open a new Command Prompt (or the Anaconda Prompt) and create a new Conda environment with Python 3.11:   conda create -n sarahmemory #python=3.11
#
#Activate your new environment:   conda activate sarahmemory
#


2. **Setup Virtual Environment:**
   
   python -m venv venv
   venv\Scripts\activate
Install Dependencies:

pip install -r requirements.txt

Run SarahMemory:
python SarahMemoryMain.py


Directory Structure
Root: C:\SarahMemory

Database Storage: C:\SarahMemory\StoredMemory

Backup: C:\SarahMemory\data\backup

Documents: C:\SarahMemory\documents

Themes: C:\SarahMemory\data\themes

Addons/Mods: C:\SarahMemory\data\addons and C:\SarahMemory\data\mods

Configuration
All settings (paths, API keys, thresholds, etc.) are defined in SarahMemoryParameters.py.

Technical Guide
For in-depth details on functionality, error-handling, API integration, and more, refer to the Technical Guide.

License
© 2025 Brian Lee Baros. All Rights Reserved.

---

Each module now includes robust error handling, improved logging, asynchronous operations, enhanced docstrings with technical guide references, and centralized configuration. The new unified API client and GUI enhancements (with theme and addon menus) further streamline the overall design. 

Feel free to adjust or expand these modules further as needed for your project.





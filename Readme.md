# SarahMemory AI Companion
![License](https://img.shields.io/badge/license-Custom%20%7C%2010%25%20Royalty-blue?style=flat-square)
![Author](https://img.shields.io/badge/Author-Brian%20Lee%20Baros-orange)
![Open Source](https://img.shields.io/badge/Open%20Source-Personal%20Use%20Only-yellow)

> Version 7.1.2 · © 2025 Brian Lee Baros. All Rights Reserved.

# SarahMemory AI Companion
# README.md <Version 7.1.2>
# © 2025 Brian Lee Baros. All Rights Reserved.

# SarahMemory AI Companion

**Version:** 7.1.2  
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

LEGAL NOTICE  
© 2025 Brian Lee Baros. All Rights Reserved.

WARNING:  
This Software is an experimental and evolving development of the SarahMemory AI-Bot Companion Platform. Although the Software is functional, unforeseen programming errors, unpredictable behavior, and potential system damage or instability may occur. The Software is provided "as is" without any warranties, express or implied.

By using this Software, you assume all risks associated with its use. The author, Brian Lee Baros, assumes no responsibility for any adverse effects, malfunctions, or damage that may result from its use or misuse.

---

LICENSE TERMS:

This Software and its associated documentation (collectively, the “Software”) are the exclusive intellectual property of **Brian Lee Baros**.

The Software is released under the following terms:

1. **OPEN-SOURCE USAGE (NON-COMMERCIAL):**  
   - You are permitted to download, use, modify, and distribute this Software for personal, educational, and research purposes.
   - Attribution must be maintained in all derivative works or redistributions by clearly stating:  
     _“Originally developed by Brian Lee Baros as part of the SarahMemory Project (2025).”_

2. **COMMERCIAL USE CLAUSE:**  
   - If you use this Software or any portion thereof, directly or indirectly, in a product, platform, service, or solution that **generates revenue**, you are **legally required to pay a 10% royalty fee** on net profits derived from such usage.
   - A formal commercial license must be requested via direct written agreement with Brian Lee Baros.
   - Failure to comply may result in legal action.

3. **PROHIBITED USES WITHOUT WRITTEN CONSENT:**  
   - Selling, sublicensing, leasing, or rebranding the Software.
   - Using the Software to train commercial AI platforms without explicit permission.
   - Including the Software in proprietary systems without honoring royalty terms.

---

INTELLECTUAL PROPERTY & ENFORCEMENT:

This Software is protected by applicable copyright, trademark, and international intellectual property laws.

All rights not expressly granted herein are reserved by **Brian Lee Baros**.  
Unauthorized reproduction, modification, or redistribution may result in severe civil or criminal penalties and will be prosecuted to the fullest extent of the law.

---

TRADEMARKS:

All trademarks, logos, or service marks used within this Software are either property of Brian Lee Baros or their respective owners. No license or right to use any of these marks is granted or implied by this License.

---

LEGAL AGREEMENT:

By downloading, accessing, or using the Software, you acknowledge and agree to these terms.  
If you do not accept these terms in full, you are not authorized to use this Software.

This License constitutes the entire legal agreement regarding the Software and supersedes any prior agreements, communications, or representations.

—

To request a commercial license or report violations, contact:  
**Brian Lee Baros**  
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
bbaros1977@gmail.com
sir_badboy@hotmail.com







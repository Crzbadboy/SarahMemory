#!/usr/bin/env python3
"""
SarahMemoryVoice.py <Version #6.4 Enhanced> <Author: Brian Lee Baros>
Description:
  Unified Voice module that handles both speech synthesis and speech recognition.
  - For synthesis, it uses pyttsx3 with support for dynamic rate, pitch, bass, and treble (simulated).
  - For recognition, it uses the SpeechRecognition library, with optional noise reduction (noisereduce)
    and multi-voice detection (using pydub if available).
  - Also includes saving and loading of voice settings to/from settings.json.
  
Notes:
  To use this file, update the import references in the other modules to import from this file.
  For production, handle SSL verification and external library dependencies securely.
"""

import pyttsx3
import speech_recognition as sr
recognizer = sr.Recognizer()
import logging
import os
import sqlite3
from datetime import datetime
import time
import json
import numpy as np
import SarahMemoryGlobals as config

# Additional libraries for noise reduction and audio splitting
try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    from pydub import AudioSegment
    AudioSegment.converter = r"C:\SarahMemory\bin\ffmpeg\bin\ffmpeg.exe"
    from pydub.silence import split_on_silence
except ImportError:
    AudioSegment = None
    split_on_silence = None

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryVoice")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# Voice Synthesis Functions (Previously in SarahMemoryVoiceSynthesis.py)
# -----------------------------------------------------------------------------
# Initialize the speech recognizer
enumerator = sr.Recognizer()
enumerator.dynamic_energy_threshold = True
recognizer = enumerator

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 185)  # Default rate
engine.setProperty('volume', 1.0)  # Max volume

# A simple lock flag (if needed)
engine_lock = False
active_voice_profile = "Default"
AVATAR_IS_SPEAKING = True

# Define voice profiles mapping (for gender or explicit names)
VOICE_PROFILES = {
    "Default": "female",
    "Female": "female",
    "Male": "male"
}

# Retrieve available voices from the engine
available_voices = engine.getProperty('voices')

# Store custom audio settings (simulated parameters)
custom_audio_settings = {
    "pitch": 1.0,
    "bass": 1.0,
    "treble": 1.0
}

# Current settings for logging/export
current_settings = {
    "speech_rate": "Normal",
    "voice_profile": "female",
    "pitch": 1.0,
    "bass": 1.0,
    "treble": 1.0,
    "reverb": 3,
}


def synthesize_voice(text):
    """Synthesize speech from text using pyttsx3."""
    if not text.strip():
        logger.warning("No text provided for synthesis; skipping TTS.")
        return
    try:
        logger.debug(f"Current audio settings: {custom_audio_settings}")
        config.AVATAR_IS_SPEAKING = True  # ✅ Prevent overlap during speaking
        engine.say(text)
        engine.runAndWait()
        config.AVATAR_IS_SPEAKING = False  # ✅ Allow mic to listen again
        logger.info("Voice synthesis completed successfully.")
    except RuntimeError as e:
        logger.error(f"TTS Runtime Error: {e}")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        config.AVATAR_IS_SPEAKING = False  # ✅ Ensure it resets even on failure

def shutdown_tts():
    """Shut down the TTS engine."""
    try:
        engine.stop()
        logger.info("TTS engine shut down successfully.")
    except Exception as e:
        logger.error(f"Failed to shut down TTS engine: {e}")

def import_custom_voice_profile(filepath):
    """Import a custom voice profile (placeholder for future logic)."""
    if os.path.exists(filepath):
        logger.info(f"Custom voice profile loaded from {filepath}")
        # Placeholder: Implement custom profile loading logic here
    else:
        logger.warning(f"Custom voice profile path invalid: {filepath}")
    try:
        # If additional exception handling is needed:
        pass
    except Exception as e:
        logger.error(f"❌ Failed to load voice settings: {e}")

# -----------------------------------------------------------------------------
# Database Logging Function
# -----------------------------------------------------------------------------
def log_voice_event(event, details):
    """Log voice events to the system_logs.db database."""
    try:
        db_path = os.path.abspath(os.path.join(config.BASE_DIR, "data", "memory", "datasets", "system_logs.db"))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
CREATE TABLE IF NOT EXISTS voice_recognition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event TEXT,
    details TEXT
)
                """
            )
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO voice_recognition_events (timestamp, event, details) VALUES (?, ?, ?)",
                           (timestamp, event, details))
            conn.commit()
        logger.info("Logged voice event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging voice event to system_logs.db: {e}")

# -----------------------------------------------------------------------------
# Voice Recognition Functions (Previously in SarahMemoryVoiceRecognition.py)
# -----------------------------------------------------------------------------

mic = None
def get_voice_profiles():
    """Return a list of available voice profile names."""
    return list(VOICE_PROFILES.keys()) + [v.name for v in available_voices if v.name not in VOICE_PROFILES]

def initialize_microphone():
    """Initializes and returns a Microphone object."""
    global mic
    if mic is None:
        try:
            mic = sr.Microphone()
            logger.info("Microphone initialized.")
            log_voice_event("Microphone Initialized", "Microphone object created successfully.")
        except Exception as e:
            logger.error(f"Microphone init failed: {e}")
            log_voice_event("Microphone Initialization Error", f"Error: {e}")
            mic = None
    return mic

def reduce_background_noise(audio):
    """
    Applies noise reduction to the AudioData if the noisereduce library is available.
    Returns new AudioData with reduced noise.
    """
    if nr is None:
        logger.warning("noisereduce library not available. Skipping noise reduction.")
        return audio
    try:
        raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        reduced = nr.reduce_noise(y=raw, sr=audio.sample_rate)
        new_raw = reduced.astype(np.int16).tobytes()
        logger.info("Background noise reduction applied successfully.")
        return sr.AudioData(new_raw, audio.sample_rate, audio.sample_width)
    except Exception as e:
        logger.error(f"Error during noise reduction: {e}")
        return audio


def listen_and_process():
    """
    Integrates ambient noise calibration, listens for voice input, applies background noise reduction,
    and processes the audio for recognition.
    It attempts multi-voice detection using pydub if available; otherwise it returns single voice input.
    Optimized to listen IMMEDIATELY after Sarah finishes speaking.
    """
    initialize_microphone()
    if mic is None:
        log_voice_event("Listen and Process", "Microphone not available.")
        return None

    # 🔄 Wait until speaking ends before activating mic
    while config.AVATAR_IS_SPEAKING:
        time.sleep(0.1)

    try:
        with mic as source:
            logger.info("Adjusting for ambient noise...")
            log_voice_event("Ambient Noise Adjustment", "Starting ambient noise adjustment.")

            recognizer = sr.Recognizer()
            # ⚡ SPEED OPTIMIZATION: Reduce ambient calibration time
            duration = getattr(config, "AMBIENT_NOISE_DURATION", 0.2)
            recognizer.adjust_for_ambient_noise(source, duration=duration)
            
            logger.info("Listening for user input...")
            audio = recognizer.listen(
                source,
                timeout=config.LISTEN_TIMEOUT,
                phrase_time_limit=config.PHRASE_TIME_LIMIT
            )
    except sr.WaitTimeoutError:
        logger.warning("Listening timed out; no speech detected.")
        log_voice_event("Listen Timeout", "No speech detected within timeout period.")
        return None
    except Exception as e:
        logger.error(f"Error during listening: {e}")
        log_voice_event("Listening Exception", f"Exception: {e}")
        return None

    # ✅ Background noise reduction (if enabled)
    audio = reduce_background_noise(audio)

    # 🔊 Attempt multi-voice detection (if supported)
    if AudioSegment is not None and split_on_silence is not None:
        try:
            audio_segment = AudioSegment(
                data=audio.get_raw_data(),
                sample_width=audio.sample_width,
                frame_rate=audio.sample_rate,
                channels=1
            )
            chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=audio_segment.dBFS - 14)
            if len(chunks) > 1:
                voices = {}
                for i, chunk in enumerate(chunks, start=1):
                    try:
                        chunk_audio = sr.AudioData(chunk.raw_data, audio.sample_rate, audio.sample_width)
                        recognized_text = recognizer.recognize_google(chunk_audio).strip()
                        voices[f"speaker_{i}"] = recognized_text
                    except Exception as e:
                        logger.warning(f"Could not recognize segment {i}: {e}")
                if voices:
                    logger.info(f"Multi-voice recognition successful: {voices}")
                    log_voice_event("Multi-Voice Captured", f"Captured voices: {voices}")
                    return voices
                else:
                    logger.warning("No voices recognized in multi-voice input.")
        except Exception as e:
            logger.error(f"Error during multi-voice processing: {e}")
            log_voice_event("Multi-Voice Recognition Exception", f"Exception: {e}")

    # 🗣️ Fallback: Single voice recognition
    try:
        text = recognizer.recognize_google(audio).strip()
        logger.info(f"Voice input recognized: {text}")
        log_voice_event("Voice Input Recognized", f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("Could not understand audio.")
        log_voice_event("Voice Input Unknown", "Audio not understood.")
    except sr.RequestError as e:
        logger.error(f"Speech recognition error: {e}")
        log_voice_event("Voice Input Error", f"RequestError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in recognition: {e}")
        log_voice_event("Voice Recognition Exception", f"Exception: {e}")
    return None

# -----------------------------------------------------------------------------
# Settings Saving/Loading Functions
# -----------------------------------------------------------------------------
def save_voice_settings():
    """
    Save the voice configuration portion to the shared settings.json file.
    This is called from the GUI as a modular voice update.
    """
    try:
        settings_path = os.path.join(config.SETTINGS_DIR, 'settings.json')
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        data = {}
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("settings.json was corrupt or empty. Rebuilding from scratch.")
        # Update only the voice settings
        data["voice_profile"] = active_voice_profile
        data["pitch"] = custom_audio_settings["pitch"]
        data["bass"] = custom_audio_settings["bass"]
        data["treble"] = custom_audio_settings["treble"]
        with open(settings_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info("✅ Voice settings saved to settings.json successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save voice settings: {e}")

def load_voice_settings():
    """
    Load voice configuration from settings.json and apply them to the voice system.
    Triggered during startup (e.g. by SarahMemoryInitialization.py).
    """
    try:
        settings_path = os.path.join(config.SETTINGS_DIR, 'settings.json')
        if not os.path.exists(settings_path):
            logger.warning("Voice settings file not found. Skipping voice configuration load.")
            return
        with open(settings_path, 'r') as f:
            data = json.load(f)
        if "voice_profile" in data:
            # Assume set_voice_profile is defined elsewhere and imported
            from SarahMemoryVoice import set_voice_profile
            set_voice_profile(data["voice_profile"])
        if "pitch" in data:
            from SarahMemoryVoice import set_pitch
            set_pitch(data["pitch"])
        if "bass" in data:
            from SarahMemoryVoice import set_bass
            set_bass(data["bass"])
        if "treble" in data:
            from SarahMemoryVoice import set_treble
            set_treble(data["treble"])
        logger.info("✅ Voice settings loaded from settings.json successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load voice settings: {e}")

# -----------------------------------------------------------------------------
# Functions for Setting Individual Parameters (Placeholders)
# -----------------------------------------------------------------------------
def set_voice_profile(profile_name):
    """Set the voice profile using full match first, then fallback to gender-based."""
    global active_voice_profile
    selected_voice = None

    # First try full name match (case-insensitive)
    for voice in available_voices:
        if profile_name.lower() == voice.name.lower():
            selected_voice = voice.id
            break

    # If not found, try gender mapping (fallback)
    if not selected_voice:
        voice_gender = VOICE_PROFILES.get(profile_name, "female")
        for voice in available_voices:
            if voice_gender.lower() in voice.name.lower():
                selected_voice = voice.id
                break

    if selected_voice:
        engine.setProperty('voice', selected_voice)
        logger.info(f"✅ Voice profile set to: {profile_name}")
    else:
        logger.warning(f"❌ Voice profile '{profile_name}' not found. Using default.")
    active_voice_profile = profile_name

def set_pitch(value):
    """Set simulated pitch."""
    custom_audio_settings["pitch"] = value
    logger.info(f"Pitch set to: {value} (simulated)")

def set_bass(value):
    """Set simulated bass."""
    custom_audio_settings["bass"] = value
    logger.info(f"Bass set to: {value} (simulated)")

def set_treble(value):
    """Set simulated treble."""
    custom_audio_settings["treble"] = value
    logger.info(f"Treble set to: {value} (simulated)")

def set_speech_rate(rate_label):
    """Set the speech rate using preset values."""
    rates = {
        "Slow": 135,
        "Normal": 185,
        "Fast": 230
    }
    rate_value = rates.get(rate_label, 185)
    engine.setProperty('rate', rate_value)
    logger.info(f"Speech rate set to: {rate_label} ({rate_value} wpm)")
    current_settings['speech_rate'] = rate_label
# -----------------------------------------------------------------------------
# Module Main Test (Optional)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # For testing purposes only; in production the functions will be imported.
    set_voice_profile("Female")
    logger.info("Starting SarahMemoryVoice module test (Combined Synthesis and Recognition).")
    
    # Test synthesis
    synthesize_voice("Hello, this is the unified voice module speaking with a natural tone.")
    
    # Test recognition (single voice)
    result = listen_and_process()
    if result:
        print(f"Recognized voice input: {result}")
    else:
        print("No valid voice input recognized.")
    
    # Test saving and loading voice settings
    save_voice_settings()
    load_voice_settings()
    
    logger.info("SarahMemoryVoice module test complete.")
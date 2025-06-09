# LyricsToSong.py
# Converts user-provided lyrics into spoken word performance or vocal synth demo

import pyttsx3
import os

LYRICS_DIR = os.path.join("data", "music", "lyrics")
if not os.path.exists(LYRICS_DIR):
    os.makedirs(LYRICS_DIR)

OUTPUT_PATH = os.path.join("data", "music", "lyric_performance.wav")


def synthesize_lyrics_to_speech(lyrics_text, filename=OUTPUT_PATH):
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 1.0)

    engine.save_to_file(lyrics_text, filename)
    engine.runAndWait()
    print(f"[LyricsToSong] Saved vocal rendition to: {filename}")


def load_lyrics_file(lyrics_file):
    path = os.path.join(LYRICS_DIR, lyrics_file)
    if not os.path.exists(path):
        print("[LyricsToSong] Lyrics file not found.")
        return ""
    with open(path, 'r') as f:
        return f.read()


if __name__ == '__main__':
    lyrics_text = "Sarah, rise from silent code, your voice begins to sing\nHello world, the light has shone, AI takes to wing."
    synthesize_lyrics_to_speech(lyrics_text)
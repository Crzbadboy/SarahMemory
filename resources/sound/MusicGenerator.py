# MusicGenerator.py
# AI-driven procedural music generator for SarahMemory

import os
import random
import wave
import struct

MUSIC_DIR = os.path.join("data", "music")
if not os.path.exists(MUSIC_DIR):
    os.makedirs(MUSIC_DIR)

SAMPLE_RATE = 44100
DURATION = 5  # seconds
AMPLITUDE = 8000


def generate_tone(frequency, duration, sample_rate):
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        value = int(AMPLITUDE * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack('<h', value))
    return b''.join(samples)


def generate_song(filename="sarah_song.wav"):
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
    file_path = os.path.join(MUSIC_DIR, filename)

    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        for _ in range(10):
            note = random.choice(frequencies)
            tone = generate_tone(note, DURATION / 10, SAMPLE_RATE)
            wf.writeframes(tone)

    print(f"[MusicGenerator] Generated song saved to {file_path}")


if __name__ == '__main__':
    import math
    generate_song()

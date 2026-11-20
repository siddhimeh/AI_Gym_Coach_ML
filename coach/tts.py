"""
Text-to-Speech (TTS) Coaching Module
Generates lightweight spoken coaching audio cues using gTTS.
Features:
- Strict utterance throttling (minimum 5s cooldown) so audio does not spam the user.
- On-disk MP3 audio caching using message hashing to avoid repeated gTTS network calls.
- Silent, non-blocking fallback if offline or if audio generation encounters an error.
"""

import os
import time
import hashlib
from io import BytesIO

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Minimum seconds between spoken audio cues
TTS_COOLDOWN_SECONDS = 5.0


class TTSCoach:
    """Manages audio generation, caching, and rate limiting for live speech feedback."""

    def __init__(self, cooldown: float = TTS_COOLDOWN_SECONDS):
        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""

    def speak(self, text: str, force: bool = False) -> str | None:
        """
        Synthesizes text into an MP3 audio file using gTTS.
        Returns the file path to the audio clip, or None if skipped or on failure.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        now = time.time()

        # Check cooldown and repetition
        if not force:
            if (now - self.last_spoken_time) < self.cooldown:
                return None

        # Check cache
        text_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{text_hash}.mp3")

        if os.path.exists(cache_file):
            self.last_spoken_time = time.time()
            self.last_spoken_text = clean_text
            return cache_file

        # Generate audio using gTTS
        try:
            from gtts import gTTS
            tts_engine = gTTS(text=clean_text, lang="en", slow=False)
            tts_engine.save(cache_file)
            self.last_spoken_time = time.time()
            self.last_spoken_text = clean_text
            return cache_file
        except Exception:
            # If offline or rate-limited by Google TTS, fail gracefully
            return None

    def get_audio_bytes(self, text: str) -> bytes | None:
        """Retrieves raw audio bytes for direct in-memory playback."""
        filepath = self.speak(text)
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return f.read()
        return None

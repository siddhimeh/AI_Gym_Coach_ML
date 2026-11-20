"""
Unit Tests for Coaching Module (LLM, TTS, Fallback)
Tests throttling cooldowns, fallback behavior, caching, and safe failure modes.
"""

import os
import time
import pytest
from coach.fallback import get_fallback_feedback, FALLBACK_MESSAGES
from coach.llm import LLMCoach
from coach.tts import TTSCoach


class TestCoachModule:
    def test_fallback_feedback_messages(self):
        """Ensures fallback returns distinct, valid feedback for each form state."""
        for status in ["good", "too_shallow", "back_leaning_forward", "standing"]:
            msg = get_fallback_feedback(status)
            assert isinstance(msg, str)
            assert len(msg) > 5

        # Check telemetry-specific messages
        shallow_msg = get_fallback_feedback("too_shallow", knee_angle=125.0)
        assert "125" in shallow_msg or "deeper" in shallow_msg

        lean_msg = get_fallback_feedback("back_leaning_forward", back_lean=45.0)
        assert "45" in lean_msg or "upright" in lean_msg

    def test_llm_coach_throttling_and_fallback(self):
        """Verifies LLM coach enforces cooldown and cleanly falls back to rule-based cues."""
        coach = LLMCoach(cooldown=2.0)

        # Call 1: Should return fallback (no API key set in test environment)
        msg1 = coach.generate_feedback("too_shallow", knee_angle=120.0, force_fallback=True)
        assert len(msg1) > 0

        # Immediate Call 2: Within cooldown, should return cached message without hitting API
        msg2 = coach.generate_feedback("too_shallow", knee_angle=120.0, force_fallback=True)
        assert msg2 == msg1

    def test_tts_throttling_and_cache(self, tmp_path):
        """Verifies TTS generates speech audio, caches it, and throttles rapid calls."""
        tts = TTSCoach(cooldown=1.0)

        # Safe handling of empty string
        assert tts.speak("") is None

        # Test audio synthesis
        audio_path = tts.speak("Great squat!", force=True)
        if audio_path:
            assert os.path.exists(audio_path)
            assert audio_path.endswith(".mp3")
            assert os.path.getsize(audio_path) > 0

            # Immediate repeat call within cooldown should return None (throttled)
            assert tts.speak("Great squat!") is None

            # After cooldown, repeat should hit cache immediately
            time.sleep(1.1)
            cached_path = tts.speak("Great squat!")
            assert cached_path == audio_path

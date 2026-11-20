"""
Coach package for AI Gym Coach.
Handles rule-based fallback, LLM natural language feedback, and gTTS audio coaching.
"""
from .fallback import get_fallback_feedback, FALLBACK_MESSAGES
from .llm import LLMCoach
from .tts import TTSCoach

__all__ = ["get_fallback_feedback", "FALLBACK_MESSAGES", "LLMCoach", "TTSCoach"]

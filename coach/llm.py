"""
LLM Coaching Module
Generates personalized, natural-language coaching feedback using an optional LLM API.
Features:
- Strict 6-second cooldown throttling to avoid rate-limiting and frame drops.
- Deduplication to avoid repeating the exact same cue.
- Automatic fallback to rule-based messages if no API key is provided or if network fails.
- Secrets secured via environment variables or Streamlit secrets (never hardcoded).
"""

import os
import time
import urllib.request
import json
from coach.fallback import get_fallback_feedback

# Minimum interval between LLM calls (seconds)
COOLDOWN_SECONDS = 6.0


class LLMCoach:
    """Manages throttled natural-language coaching generation with rule fallback."""

    def __init__(self, cooldown: float = COOLDOWN_SECONDS):
        self.cooldown = cooldown
        self.last_call_time = 0.0
        self.last_status = None
        self.last_message = ""

    def _get_api_key(self) -> str | None:
        """Retrieves API key from environment variable or Streamlit secrets."""
        # 1. Check environment variables
        key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if key:
            return key

        # 2. Check Streamlit secrets if running inside Streamlit
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
            if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

        return None

    def generate_feedback(
        self,
        status: str,
        knee_angle: float | None = None,
        back_lean: float | None = None,
        rep_count: int = 0,
        force_fallback: bool = False,
    ) -> str:
        """
        Generates short, actionable coaching feedback.
        Enforces cooldown and deduplication.
        """
        now = time.time()

        # Check if status has not changed and cooldown hasn't expired
        if status == self.last_status and (now - self.last_call_time) < self.cooldown:
            return self.last_message or get_fallback_feedback(status, knee_angle, back_lean)

        api_key = self._get_api_key()

        # Fallback if no API key or forced fallback
        if force_fallback or not api_key:
            msg = get_fallback_feedback(status, knee_angle, back_lean)
            self.last_call_time = now
            self.last_status = status
            self.last_message = msg
            return msg

        # Attempt API call if cooldown has elapsed
        if (now - self.last_call_time) >= self.cooldown:
            try:
                # Example lightweight prompt for LLM
                prompt = (
                    f"You are an expert strength coach. Give a 1-sentence concise, encouraging coaching cue "
                    f"for a squatter. Form status: {status}, Knee Angle: {knee_angle} deg (target <= 95 deg), "
                    f"Back Lean: {back_lean} deg (target <= 38 deg), Current Reps: {rep_count}. "
                    f"Do not mention degrees or numbers, speak directly to the athlete like a live trainer."
                )

                # Call Google Gemini REST endpoint using built-in urllib
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                data = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}]
                }).encode("utf-8")

                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    cand = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    msg = cand.replace("\n", " ")

                self.last_call_time = now
                self.last_status = status
                self.last_message = msg
                return msg
            except Exception as e:
                # Network or parsing failure -> seamless rule fallback
                msg = get_fallback_feedback(status, knee_angle, back_lean)
                self.last_call_time = now
                self.last_status = status
                self.last_message = msg
                return msg

        return self.last_message or get_fallback_feedback(status, knee_angle, back_lean)

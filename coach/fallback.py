"""
Rule-Based Fallback Coaching Module
Provides explainable, predefined feedback cues when an LLM API is unavailable,
unconfigured, or offline.
"""

import random

FALLBACK_MESSAGES = {
    "good": [
        "Great rep! Perfect depth and upright posture.",
        "Excellent depth! Keep that strong posture.",
        "Solid repetition! Clean lockout at the top.",
    ],
    "too_shallow": [
        "Try to squat a little deeper.",
        "Lower your hips to break parallel with your knees.",
        "Rep counted, but too shallow. Push for full depth next time.",
    ],
    "back_leaning_forward": [
        "Keep your back more upright.",
        "Keep your chest proud and eyes forward.",
        "Avoid excessive forward torso lean to protect your lower back.",
    ],
    "standing": [
        "Ready — begin your descent.",
        "Stand tall and brace your core.",
    ],
    "descending": [
        "Lowering smoothly...",
        "Control your descent.",
    ],
    "bottom": [
        "Good depth! Drive back up through your midfoot.",
        "Hit parallel! Power upward.",
    ],
}


def get_fallback_feedback(status: str, knee_angle: float | None = None, back_lean: float | None = None) -> str:
    """
    Returns an appropriate rule-based coaching message based on form status and joint kinematics.
    """
    status_key = status.lower() if status else "standing"
    options = FALLBACK_MESSAGES.get(status_key, FALLBACK_MESSAGES["standing"])

    # If telemetry details are available, give specific bio-cues
    if status_key == "too_shallow" and knee_angle is not None:
        return f"Try to squat a little deeper. Your knee reached {knee_angle:.0f}°, target is 95°."
    elif status_key == "back_leaning_forward" and back_lean is not None:
        return f"Keep your back more upright. Torso leaned {back_lean:.0f}°, limit is 38°."
    elif status_key == "good" and knee_angle is not None:
        return f"Great rep! Clean {knee_angle:.0f}° depth with upright posture."

    return random.choice(options)

"""
AI Gym Coach Configuration & Global Settings
Centralized parameters for computer vision thresholds, exercise targets, and UI themes.
"""

import os

# Base Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
DB_PATH = os.path.join(ROOT_DIR, "db", "gym_coach.db")

# Camera & Performance Defaults
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Biomechanical Angle Thresholds (Calibrated from Phase 4 Kinematic Analysis)
UP_THRESHOLD = 160.0       # Knee angle (deg) to complete standing lockout
DOWN_INITIATE = 150.0      # Knee angle (deg) to initiate descent (hysteresis)
DEPTH_THRESHOLD = 95.0     # Knee angle (deg) required for parallel/full depth
SHALLOW_DEPTH_MAX = 140.0  # Max knee angle (deg) to qualify as a genuine shallow squat (rejects standing sway)
BACK_LEAN_MAX = 38.0       # Max allowable torso forward lean (deg from vertical)
MIN_DETECTION_CONFIDENCE = 0.50
MIN_REP_FRAMES = 4         # Minimum duration (frames) to complete a valid rep (rejects 1-2 frame spikes)
STOP_AT_TARGET_DEFAULT = True  # Stop accepting additional repetitions once planned target is reached
SIDE_SWITCH_HYSTERESIS = 0.10  # Minimum confidence delta required to switch active side

# Supported Exercises
EXERCISES = {
    "squats": {
        "name": "Barbell / Bodyweight Squats",
        "view": "Side View",
        "description": "Lower-body compound exercise targeting quadriceps, glutes, and hamstrings.",
        "primary_angle": "Knee Flexion (Hip-Knee-Ankle)",
        "secondary_angle": "Torso Posture (Hip-Shoulder vs Vertical)",
        "target_depth": "<= 95°",
        "upright_angle": ">= 160°",
    }
}

# Educational Viva Disclaimer
APP_DISCLAIMER = (
    "Educational Prototype: This application was developed as an academic Machine Learning "
    "course project at SRM Institute of Science and Technology. It is an educational tool for "
    "biomechanical posture awareness, not a medical or physical therapy diagnostic device."
)

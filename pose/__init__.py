"""
Pose estimation and landmark detection package.
"""
from .detector import PoseDetector
from .angles import (
    calculate_3point_angle,
    calculate_knee_angle,
    calculate_hip_angle,
    calculate_back_lean,
)

__all__ = [
    "PoseDetector",
    "calculate_3point_angle",
    "calculate_knee_angle",
    "calculate_hip_angle",
    "calculate_back_lean",
]

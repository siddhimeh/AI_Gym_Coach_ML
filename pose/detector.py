"""
MediaPipe Pose Detection Module
Encapsulates pose landmark detection, keypoint extraction, and visualization.
Uses the lightweight MediaPipe Tasks PoseLandmarker model optimized for CPU.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Target MediaPipe Landmark indices for lower-body exercise analysis
LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose_landmarker_lite.task")


class PoseDetector:
    """Detects human pose landmarks and extracts key lower-body joints."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, min_detection_confidence: float = 0.5):
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.active_side = None
        self.side_switch_margin = 0.10

    def reset(self):
        """Resets active tracking side and internal filter state."""
        self.active_side = None

    def _ensure_model_exists(self):
        """Downloads the lightweight pose landmarker model if not present locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Downloading pose model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Pose model download complete.")

    def find_pose(self, frame_bgr: np.ndarray):
        """
        Runs pose detection on a single BGR video frame.
        
        Returns:
            result: The raw MediaPipe PoseLandmarkerResult object.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.detector.detect(mp_image)

    def extract_landmarks(self, result, frame_shape: tuple) -> dict | None:
        """
        Extracts relevant lower-body joints, pixel coordinates, and visibility.
        Determines whether the left or right side is facing the camera in a side view.
        Applies hysteresis to prevent rapid side-switching jitter.
        Rejects corrupted coordinates (NaN, Inf) and impossible joint geometry.

        Args:
            result: PoseLandmarkerResult from find_pose()
            frame_shape: (height, width, channels) of the frame

        Returns:
            Dictionary with side, keypoints, and visibility, or None if no person detected.
        """
        if not result or not result.pose_landmarks:
            return None

        h, w = frame_shape[:2]
        raw_lms = result.pose_landmarks[0]  # First detected person

        # Reject corrupted landmarks with non-finite coordinates or out-of-frame values
        for idx in [
            LANDMARKS["LEFT_SHOULDER"], LANDMARKS["RIGHT_SHOULDER"],
            LANDMARKS["LEFT_HIP"], LANDMARKS["RIGHT_HIP"],
            LANDMARKS["LEFT_KNEE"], LANDMARKS["RIGHT_KNEE"],
            LANDMARKS["LEFT_ANKLE"], LANDMARKS["RIGHT_ANKLE"],
        ]:
            if idx >= len(raw_lms):
                return None
            lm = raw_lms[idx]
            if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                return None

        def get_pt(idx: int):
            lm = raw_lms[idx]
            vis = getattr(lm, "visibility", 1.0)
            pres = getattr(lm, "presence", 1.0)
            v_val = float(vis) if (vis is not None and np.isfinite(vis)) else 1.0
            p_val = float(pres) if (pres is not None and np.isfinite(pres)) else 1.0
            conf = min(v_val, p_val)
            return {
                "x_norm": float(lm.x),
                "y_norm": float(lm.y),
                "z_norm": float(lm.z) if hasattr(lm, "z") else 0.0,
                "x_px": int(round(lm.x * w)),
                "y_px": int(round(lm.y * h)),
                "visibility": float(conf),
                "presence": p_val,
            }

        # Left side keypoints
        left_shoulder = get_pt(LANDMARKS["LEFT_SHOULDER"])
        left_hip = get_pt(LANDMARKS["LEFT_HIP"])
        left_knee = get_pt(LANDMARKS["LEFT_KNEE"])
        left_ankle = get_pt(LANDMARKS["LEFT_ANKLE"])
        left_heel = get_pt(LANDMARKS["LEFT_HEEL"])
        left_foot = get_pt(LANDMARKS["LEFT_FOOT_INDEX"])

        # Right side keypoints
        right_shoulder = get_pt(LANDMARKS["RIGHT_SHOULDER"])
        right_hip = get_pt(LANDMARKS["RIGHT_HIP"])
        right_knee = get_pt(LANDMARKS["RIGHT_KNEE"])
        right_ankle = get_pt(LANDMARKS["RIGHT_ANKLE"])
        right_heel = get_pt(LANDMARKS["RIGHT_HEEL"])
        right_foot = get_pt(LANDMARKS["RIGHT_FOOT_INDEX"])

        # Compare side visibilities to pick the active camera-facing side with hysteresis
        left_vis = (left_shoulder["visibility"] + left_hip["visibility"] + left_knee["visibility"] + left_ankle["visibility"]) / 4.0
        right_vis = (right_shoulder["visibility"] + right_hip["visibility"] + right_knee["visibility"] + right_ankle["visibility"]) / 4.0

        if self.active_side is None:
            self.active_side = "left" if left_vis >= right_vis else "right"
        else:
            if self.active_side == "left" and (right_vis - left_vis) > self.side_switch_margin:
                self.active_side = "right"
            elif self.active_side == "right" and (left_vis - right_vis) > self.side_switch_margin:
                self.active_side = "left"

        active_side = self.active_side

        active_points = {
            "shoulder": left_shoulder if active_side == "left" else right_shoulder,
            "hip": left_hip if active_side == "left" else right_hip,
            "knee": left_knee if active_side == "left" else right_knee,
            "ankle": left_ankle if active_side == "left" else right_ankle,
            "heel": left_heel if active_side == "left" else right_heel,
            "foot": left_foot if active_side == "left" else right_foot,
        }

        # Reject impossible joint geometry (e.g. ankle above knee or zero leg length)
        ankle_y = active_points["ankle"]["y_px"]
        knee_y = active_points["knee"]["y_px"]
        if ankle_y <= knee_y:
            return None

        return {
            "side": active_side,
            "confidence": max(left_vis, right_vis),
            "points": active_points,
            "left_points": {
                "shoulder": left_shoulder, "hip": left_hip, "knee": left_knee,
                "ankle": left_ankle, "heel": left_heel, "foot": left_foot,
            },
            "right_points": {
                "shoulder": right_shoulder, "hip": right_hip, "knee": right_knee,
                "ankle": right_ankle, "heel": right_heel, "foot": right_foot,
            },
            "raw_landmarks": raw_lms,
        }

    def draw_skeleton(self, frame_bgr: np.ndarray, keypoint_data: dict | None) -> np.ndarray:
        """
        Draws the active side skeleton connections (shoulder -> hip -> knee -> ankle -> foot).
        """
        annotated = frame_bgr.copy()
        if not keypoint_data:
            return annotated

        pts = keypoint_data["points"]
        side = keypoint_data["side"]

        # Color scheme: Teal lines (#00e5ff) and Amber joints (#ffb300)
        line_color = (255, 229, 0)   # BGR format: Yellow/Teal
        joint_color = (0, 179, 255)  # BGR format: Amber/Orange
        head_color = (0, 255, 128)

        connections = [
            ("shoulder", "hip"),
            ("hip", "knee"),
            ("knee", "ankle"),
            ("ankle", "heel"),
            ("heel", "foot"),
            ("ankle", "foot"),
        ]

        # Draw bone connections
        for start_k, end_k in connections:
            p1 = (pts[start_k]["x_px"], pts[start_k]["y_px"])
            p2 = (pts[end_k]["x_px"], pts[end_k]["y_px"])
            cv2.line(annotated, p1, p2, line_color, 3, cv2.LINE_AA)

        # Draw joint nodes
        for joint_k in ["shoulder", "hip", "knee", "ankle", "heel", "foot"]:
            p = (pts[joint_k]["x_px"], pts[joint_k]["y_px"])
            cv2.circle(annotated, p, 6, joint_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, p, 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw side indicator badge on top-left
        badge_text = f"Active Side: {side.upper()} (vis: {keypoint_data['confidence']:.2f})"
        cv2.putText(annotated, badge_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

        return annotated

    def close(self):
        """Releases the detector resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()

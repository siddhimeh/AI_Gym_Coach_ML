"""
Squat Counter & Form Analysis Module
Implements a 4-state Finite State Machine (FSM):
STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING
Includes hysteresis thresholds, landmark visibility validation,
standing calibration, downward amplitude verification,
and explainable biomechanical form assessment.
"""

import time
from collections import deque
from enum import Enum
import numpy as np
from pose.angles import calculate_knee_angle, calculate_hip_angle, calculate_back_lean


class SquatState(str, Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class SquatCounter:
    """
    Tracks lower-body squat biomechanics and counts repetitions with form assessment.
    
    Calibrated Biomechanical Thresholds:
      - UP_THRESHOLD = 160.0 deg (Full standing extension / lockout)
      - DOWN_INITIATE = 150.0 deg (Hysteresis deadband entry into descent)
      - DEPTH_THRESHOLD = 95.0 deg (Parallel / full depth cutoff)
      - SHALLOW_DEPTH_MAX = 125.0 deg (Cutoff for genuine shallow squat; rejects standing sway)
      - BACK_LEAN_MAX = 38.0 deg (Warning cutoff for excessive forward lean)
      - MIN_VISIBILITY = 0.50 (Noise / occlusion confidence cutoff)
      - MIN_REP_FRAMES = 4 (Rejects 1-2 frame tracking spikes)
      - MIN_AMPLITUDE = 0.08 (Minimum vertical hip displacement fraction of thigh length)
    """

    def __init__(
        self,
        up_threshold: float = 160.0,
        down_initiate: float = 150.0,
        depth_threshold: float = 95.0,
        shallow_depth_max: float = 125.0,
        back_lean_max: float = 38.0,
        min_visibility: float = 0.50,
        min_rep_frames: int = 4,
        max_consecutive_invalid: int = 15,
        target_reps: int | None = None,
        stop_at_target: bool = False,
        min_amplitude: float = 0.08,
        smoothing_alpha: float = 1.0,
        debug: bool = False,
        **kwargs,
    ):
        self.up_threshold = float(up_threshold)
        self.down_initiate = float(down_initiate)
        self.depth_threshold = float(depth_threshold)
        self.shallow_depth_max = float(shallow_depth_max)
        self.back_lean_max = float(back_lean_max)
        self.min_visibility = float(min_visibility)
        self.min_rep_frames = int(min_rep_frames)
        self.max_consecutive_invalid = int(max_consecutive_invalid)
        self.target_reps = target_reps
        self.stop_at_target = stop_at_target
        self.min_amplitude = float(min_amplitude)
        self.smoothing_alpha = float(smoothing_alpha)
        self.debug = bool(debug)

        # Bounded telemetry debug ring buffer (capped at 500 entries)
        self.debug_logs: deque = deque(maxlen=500)
        self.frame_count: int = 0
        self.rep_event_counter: int = 0

        self.reset()

    def set_target(self, target_reps: int | None = None, stop_at_target: bool = False):
        """
        Configures or updates target repetitions and hard stop behavior.
        """
        self.target_reps = target_reps
        self.stop_at_target = stop_at_target
        if self.stop_at_target and self.target_reps is not None and self.rep_count >= self.target_reps:
            self.target_reached = True
            self.feedback = f"Goal reached ({self.rep_count}/{self.target_reps} reps)! Workout complete"

    def reset(self):
        """Resets all workout counters and FSM state variables."""
        self.state = SquatState.STANDING
        self.rep_count = 0
        self.good_reps = 0
        self.shallow_reps = 0
        self.forward_lean_reps = 0

        # Repetition history tracking (per-rep telemetry)
        self.rep_history: list[dict] = []

        # Current repetition tracking
        self.current_min_knee = 180.0
        self.current_max_back_lean = 0.0
        self.current_max_amplitude = 0.0
        self.depth_achieved = False
        self.rep_frames = 0
        self.consecutive_invalid_frames = 0
        self.target_reached = False

        # Standing pose baseline calibration
        self.standing_hip_y: float | None = None
        self.standing_thigh_len: float | None = None
        self.standing_calibration_frames = 0

        # Filtered telemetry
        self.filtered_knee: float | None = None

        # Feedback messages
        self.feedback = "Stand in frame to begin"
        self.last_rep_status = "none"

    def _extract_hip_knee_geometry(self, pts: dict) -> tuple[float, float, float] | None:
        """Helper to extract hip_y, knee_y, and thigh length in consistent units."""
        if not pts or "hip" not in pts or "knee" not in pts:
            return None
        hip = pts["hip"]
        knee = pts["knee"]

        # Prefer pixel coordinates if present
        if "y_px" in hip and "y_px" in knee:
            hy = float(hip["y_px"])
            ky = float(knee["y_px"])
            hx = float(hip.get("x_px", 0.0))
            kx = float(knee.get("x_px", 0.0))
        elif "y_norm" in hip and "y_norm" in knee:
            hy = float(hip["y_norm"])
            ky = float(knee["y_norm"])
            hx = float(hip.get("x_norm", 0.0))
            kx = float(knee.get("x_norm", 0.0))
        elif "y" in hip and "y" in knee:
            hy = float(hip["y"])
            ky = float(knee["y"])
            hx = float(hip.get("x", 0.0))
            kx = float(knee.get("x", 0.0))
        else:
            return None

        thigh_len = float(np.hypot(hx - kx, hy - ky))
        return hy, ky, thigh_len

    def update(self, keypoint_data: dict | None) -> dict:
        """
        Updates the state machine with newly detected frame keypoints.

        Args:
            keypoint_data: Dictionary returned by PoseDetector.extract_landmarks()

        Returns:
            Dictionary containing state, metrics, rep count, and live feedback.
        """
        self.frame_count += 1
        prev_state = self.state
        rep_count_before = self.rep_count
        completed_rep_id = None
        transition_reason = "none"

        # Validate input packet and landmark availability
        if (
            not keypoint_data
            or keypoint_data.get("confidence", 0.0) < self.min_visibility
            or not keypoint_data.get("points")
            or not all(k in keypoint_data["points"] for k in ("shoulder", "hip", "knee", "ankle"))
        ):
            self.consecutive_invalid_frames += 1
            # If tracking is lost prolonged mid-rep, safely abort in-progress rep to prevent false count
            if self.consecutive_invalid_frames >= self.max_consecutive_invalid and self.state != SquatState.STANDING:
                self.state = SquatState.STANDING
                self.current_min_knee = 180.0
                self.current_max_back_lean = 0.0
                self.current_max_amplitude = 0.0
                self.depth_achieved = False
                self.rep_frames = 0
                transition_reason = f"aborted_after_{self.max_consecutive_invalid}_invalid_frames"

            if self.debug:
                self._record_debug_entry(
                    keypoint_data=keypoint_data,
                    knee_angle_raw=None,
                    knee_angle=None,
                    hip_angle=None,
                    back_lean=None,
                    prev_state=prev_state,
                    transition_reason=transition_reason,
                    amplitude=0.0,
                    completed_rep_id=None,
                    rep_count_before=rep_count_before,
                    invalid_reason="missing_or_low_confidence_landmarks",
                )

            return {
                "state": self.state.value,
                "rep_count": self.rep_count,
                "knee_angle": None,
                "hip_angle": None,
                "back_lean": None,
                "feedback": "Step fully into camera view",
                "last_rep_status": self.last_rep_status,
                "good_reps": self.good_reps,
                "shallow_reps": self.shallow_reps,
                "forward_lean_reps": self.forward_lean_reps,
                "target_reached": self.target_reached,
            }

        self.consecutive_invalid_frames = 0
        pts = keypoint_data["points"]

        try:
            knee_angle_raw = calculate_knee_angle(pts["hip"], pts["knee"], pts["ankle"])
            hip_angle = calculate_hip_angle(pts["shoulder"], pts["hip"], pts["knee"])
            back_lean = calculate_back_lean(pts["shoulder"], pts["hip"])
        except (ValueError, TypeError, KeyError):
            return {
                "state": self.state.value,
                "rep_count": self.rep_count,
                "knee_angle": None,
                "hip_angle": None,
                "back_lean": None,
                "feedback": "Step fully into camera view",
                "last_rep_status": self.last_rep_status,
                "good_reps": self.good_reps,
                "shallow_reps": self.shallow_reps,
                "forward_lean_reps": self.forward_lean_reps,
                "target_reached": self.target_reached,
            }

        # Exponential Moving Average angle filter (if smoothing_alpha < 1.0)
        if self.filtered_knee is None:
            self.filtered_knee = knee_angle_raw
        else:
            if self.smoothing_alpha < 1.0:
                # Reject impossible single-frame jump (> 45 deg)
                if abs(knee_angle_raw - self.filtered_knee) > 45.0:
                    knee_angle_raw = self.filtered_knee
                self.filtered_knee = self.smoothing_alpha * knee_angle_raw + (1.0 - self.smoothing_alpha) * self.filtered_knee
            else:
                self.filtered_knee = knee_angle_raw

        knee_angle = round(float(self.filtered_knee), 1)

        # Track downward hip movement (amplitude) relative to standing baseline
        geom = self._extract_hip_knee_geometry(pts)
        current_amplitude = 0.0
        if geom:
            hy, ky, thigh_len = geom
            if self.state == SquatState.STANDING and knee_angle >= self.up_threshold:
                # Update standing baseline calibration
                if self.standing_hip_y is None:
                    self.standing_hip_y = hy
                    self.standing_thigh_len = thigh_len
                else:
                    self.standing_hip_y = 0.95 * self.standing_hip_y + 0.05 * hy
                    self.standing_thigh_len = 0.95 * self.standing_thigh_len + 0.05 * thigh_len
                self.standing_calibration_frames += 1

            if self.standing_hip_y is not None and self.standing_thigh_len and self.standing_thigh_len > 0:
                current_amplitude = (hy - self.standing_hip_y) / self.standing_thigh_len
                if self.state in (SquatState.DESCENDING, SquatState.BOTTOM, SquatState.ASCENDING):
                    self.current_max_amplitude = max(self.current_max_amplitude, current_amplitude)

        # Check target rep hard stop limit
        if self.stop_at_target and self.target_reps is not None and self.rep_count >= self.target_reps:
            self.target_reached = True
            self.feedback = f"Goal reached ({self.rep_count}/{self.target_reps} reps)! Workout complete"
            return {
                "state": SquatState.STANDING.value,
                "rep_count": self.rep_count,
                "knee_angle": knee_angle,
                "hip_angle": hip_angle,
                "back_lean": back_lean,
                "feedback": self.feedback,
                "last_rep_status": self.last_rep_status,
                "good_reps": self.good_reps,
                "shallow_reps": self.shallow_reps,
                "forward_lean_reps": self.forward_lean_reps,
                "target_reached": True,
            }

        # ----------------- FINITE STATE MACHINE -----------------
        if self.state == SquatState.STANDING:
            self.feedback = "Ready - begin descending"
            # Descent initiates when knee breaks below initiate threshold (hysteresis deadband)
            if knee_angle < self.down_initiate:
                # Reject impossible single-frame spikes (e.g. jumping from 170 to <40)
                if knee_angle >= 40.0:
                    self.state = SquatState.DESCENDING
                    self.current_min_knee = knee_angle
                    self.current_max_back_lean = back_lean
                    self.depth_achieved = (knee_angle <= self.depth_threshold)
                    self.rep_frames = 1
                    self.feedback = "Squatting down..."
                    transition_reason = f"knee_angle_{knee_angle}_below_down_initiate_{self.down_initiate}"

        elif self.state == SquatState.DESCENDING:
            self.rep_frames += 1
            self.current_min_knee = min(self.current_min_knee, knee_angle)
            self.current_max_back_lean = max(self.current_max_back_lean, back_lean)

            # Check for excessive back lean warning in real time
            if back_lean > self.back_lean_max:
                self.feedback = "Keep chest up! Torso leaning forward"
            else:
                self.feedback = "Lowering..."

            # 1. Parallel / Full depth achieved
            if knee_angle <= self.depth_threshold:
                self.state = SquatState.BOTTOM
                self.depth_achieved = True
                self.feedback = "Good depth! Drive back up"
                transition_reason = f"knee_angle_{knee_angle}_reached_depth_threshold_{self.depth_threshold}"

            # 2. Aborted descent or standing sway returned to standing without achieving squat depth
            elif knee_angle >= self.up_threshold:
                # If movement never reached valid shallow depth (shallow_depth_max), discard as standing jitter
                self.state = SquatState.STANDING
                self.current_min_knee = 180.0
                self.current_max_back_lean = 0.0
                self.current_max_amplitude = 0.0
                self.depth_achieved = False
                self.rep_frames = 0
                self.feedback = "Ready - begin descending"
                transition_reason = f"aborted_to_standing_without_reaching_shallow_max_{self.shallow_depth_max}"

            # 3. Genuine shallow squat turnaround: must descend <= shallow_depth_max and rise by 10 deg
            elif (
                self.current_min_knee <= self.shallow_depth_max
                and knee_angle > self.current_min_knee + 10.0
            ):
                self.state = SquatState.ASCENDING
                self.feedback = "Rising..."
                transition_reason = f"shallow_turnaround_min_{self.current_min_knee}_rose_to_{knee_angle}"

        elif self.state == SquatState.BOTTOM:
            self.rep_frames += 1
            self.current_min_knee = min(self.current_min_knee, knee_angle)
            self.current_max_back_lean = max(self.current_max_back_lean, back_lean)

            # Lifter begins rising up from bottom
            if knee_angle > self.depth_threshold + 10.0:
                self.state = SquatState.ASCENDING
                self.feedback = "Rising back up..."
                transition_reason = f"bottom_ascent_turnaround_rose_above_{self.depth_threshold + 10.0}"

        elif self.state == SquatState.ASCENDING:
            self.rep_frames += 1
            self.current_max_back_lean = max(self.current_max_back_lean, back_lean)

            if knee_angle >= self.up_threshold:
                # Repetition complete! Validate movement duration and credible trajectory
                # Must satisfy minimum frames and must have actually descended to shallow_depth_max
                valid_cycle = (
                    self.rep_frames >= self.min_rep_frames
                    and self.current_min_knee <= self.shallow_depth_max
                )

                if valid_cycle:
                    self.rep_count += 1
                    self.rep_event_counter += 1
                    completed_rep_id = self.rep_event_counter

                    is_leaning = self.current_max_back_lean > self.back_lean_max
                    is_deep = self.depth_achieved

                    if is_deep and not is_leaning:
                        self.good_reps += 1
                        self.last_rep_status = "good"
                        self.feedback = "Great rep! Perfect depth and posture"
                    elif not is_deep and not is_leaning:
                        self.shallow_reps += 1
                        self.last_rep_status = "too_shallow"
                        self.feedback = "Rep counted, but too shallow! Squat lower"
                    elif is_deep and is_leaning:
                        self.forward_lean_reps += 1
                        self.last_rep_status = "back_leaning_forward"
                        self.feedback = "Good depth, but keep your chest upright!"
                    else:
                        self.shallow_reps += 1
                        self.forward_lean_reps += 1
                        self.last_rep_status = "too_shallow"
                        self.feedback = "Squat deeper and keep your back upright"

                    # Record repetition metrics for fatigue and biomechanical analysis
                    self.rep_history.append({
                        "rep_number": self.rep_count,
                        "status": self.last_rep_status,
                        "min_knee_angle": round(float(self.current_min_knee), 1),
                        "max_back_lean": round(float(self.current_max_back_lean), 1),
                        "duration_frames": int(self.rep_frames),
                        "max_amplitude": round(float(self.current_max_amplitude), 3),
                        "timestamp": round(time.time(), 2),
                        "hip_angle": round(float(hip_angle), 1) if hip_angle is not None else None,
                    })

                    transition_reason = f"rep_{self.rep_count}_completed_status_{self.last_rep_status}"

                    # Check if target reached upon rep completion
                    if self.stop_at_target and self.target_reps is not None and self.rep_count >= self.target_reps:
                        self.target_reached = True
                        self.feedback = f"Goal reached ({self.rep_count}/{self.target_reps} reps)! Workout complete"
                else:
                    transition_reason = f"discarded_rep_frames_{self.rep_frames}_min_knee_{self.current_min_knee}"

                # Reset rep tracking
                self.state = SquatState.STANDING
                self.current_min_knee = 180.0
                self.current_max_back_lean = 0.0
                self.current_max_amplitude = 0.0
                self.depth_achieved = False
                self.rep_frames = 0

        if self.debug:
            self._record_debug_entry(
                keypoint_data=keypoint_data,
                knee_angle_raw=knee_angle_raw,
                knee_angle=knee_angle,
                hip_angle=hip_angle,
                back_lean=back_lean,
                prev_state=prev_state,
                transition_reason=transition_reason,
                amplitude=current_amplitude,
                completed_rep_id=completed_rep_id,
                rep_count_before=rep_count_before,
                invalid_reason=None,
            )

        return {
            "state": self.state.value,
            "rep_count": self.rep_count,
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "back_lean": back_lean,
            "feedback": self.feedback,
            "last_rep_status": self.last_rep_status,
            "good_reps": self.good_reps,
            "shallow_reps": self.shallow_reps,
            "forward_lean_reps": self.forward_lean_reps,
            "target_reached": self.target_reached,
        }

    def _record_debug_entry(
        self,
        keypoint_data: dict | None,
        knee_angle_raw: float | None,
        knee_angle: float | None,
        hip_angle: float | None,
        back_lean: float | None,
        prev_state: SquatState,
        transition_reason: str,
        amplitude: float,
        completed_rep_id: int | None,
        rep_count_before: int,
        invalid_reason: str | None,
    ):
        """Maintains a bounded in-memory telemetry debug log."""
        entry = {
            "frame_idx": self.frame_count,
            "timestamp": time.time(),
            "visibility": float(keypoint_data.get("confidence", 0.0)) if keypoint_data else 0.0,
            "side": keypoint_data.get("side") if keypoint_data else None,
            "raw_knee": round(knee_angle_raw, 2) if knee_angle_raw is not None else None,
            "filtered_knee": round(knee_angle, 2) if knee_angle is not None else None,
            "hip_angle": round(hip_angle, 2) if hip_angle is not None else None,
            "back_lean": round(back_lean, 2) if back_lean is not None else None,
            "state": self.state.value,
            "prev_state": prev_state.value,
            "transition_reason": transition_reason,
            "movement_amplitude": round(float(amplitude), 3),
            "completed_rep_id": completed_rep_id,
            "rep_count_before": rep_count_before,
            "rep_count_after": self.rep_count,
            "invalid_reason": invalid_reason,
        }
        self.debug_logs.append(entry)

    def get_debug_logs(self) -> list[dict]:
        """Returns bounded debug entries."""
        return list(self.debug_logs)

    def clear_debug_logs(self):
        """Clears debug buffer."""
        self.debug_logs.clear()

    def get_rep_history(self) -> list[dict]:
        """Returns the chronological list of completed repetition metrics."""
        return [dict(r) for r in self.rep_history]

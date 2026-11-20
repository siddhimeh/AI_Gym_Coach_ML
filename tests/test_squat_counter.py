"""
Unit Tests for Squat Counter State Machine
Tests hysteresis, depth assessment, forward lean detection, and noise robustness.
"""

import pytest
import numpy as np
from exercises.squat import SquatCounter, SquatState


def make_dummy_keypoints(knee_angle: float, back_lean: float, confidence: float = 0.95) -> dict:
    """Creates a mock keypoint dictionary that produces the exact target knee_angle and back_lean."""
    # Knee at (0.5, 0.7), Ankle at (0.5, 1.0)
    rad = np.radians(180.0 - knee_angle)
    hip_x = 0.5 + 0.3 * np.sin(rad)
    hip_y = 0.7 - 0.3 * np.cos(rad)

    lean_rad = np.radians(back_lean)
    shoulder_y = hip_y - 0.4
    shoulder_x = hip_x + 0.4 * np.tan(lean_rad)

    return {
        "side": "right",
        "confidence": confidence,
        "points": {
            "shoulder": {"x_norm": float(shoulder_x), "y_norm": float(shoulder_y), "visibility": 0.98},
            "hip": {"x_norm": float(hip_x), "y_norm": float(hip_y), "visibility": 0.98},
            "knee": {"x_norm": 0.5, "y_norm": 0.7, "visibility": 0.98},
            "ankle": {"x_norm": 0.5, "y_norm": 1.0, "visibility": 0.98},
            "heel": {"x_norm": 0.5, "y_norm": 1.05, "visibility": 0.95},
            "foot": {"x_norm": 0.55, "y_norm": 1.05, "visibility": 0.95},
        }
    }


class TestSquatCounter:
    def test_good_squat_cycle(self):
        counter = SquatCounter()

        # 1. Standing frames
        for _ in range(3):
            res = counter.update(make_dummy_keypoints(knee_angle=175.0, back_lean=10.0))
            assert res["state"] == SquatState.STANDING

        # 2. Descent frames (initiates below 150)
        for k in [145.0, 130.0, 115.0, 100.0]:
            res = counter.update(make_dummy_keypoints(knee_angle=k, back_lean=15.0))
            assert res["state"] == SquatState.DESCENDING

        # 3. Bottom position (reaches <=95)
        for k in [90.0, 75.0, 80.0]:
            res = counter.update(make_dummy_keypoints(knee_angle=k, back_lean=18.0))
            assert res["state"] == SquatState.BOTTOM

        # 4. Ascent frames
        for k in [110.0, 130.0, 150.0]:
            res = counter.update(make_dummy_keypoints(knee_angle=k, back_lean=15.0))
            assert res["state"] == SquatState.ASCENDING

        # 5. Lockout standing (>=160)
        res = counter.update(make_dummy_keypoints(knee_angle=165.0, back_lean=10.0))
        assert res["rep_count"] == 1
        assert res["good_reps"] == 1
        assert res["shallow_reps"] == 0
        assert res["last_rep_status"] == "good"

    def test_shallow_squat_cycle(self):
        counter = SquatCounter()

        # Standing
        counter.update(make_dummy_keypoints(knee_angle=175.0, back_lean=10.0))
        # Descend only to 125 degrees (half squat, does not reach 95)
        for k in [145.0, 135.0, 128.0, 125.0, 125.0, 125.0]:
            counter.update(make_dummy_keypoints(knee_angle=k, back_lean=12.0))
        # Turn around and rise (+10 deg above min 125)
        for k in [138.0, 148.0, 155.0]:
            counter.update(make_dummy_keypoints(knee_angle=k, back_lean=12.0))
        # Finish lockout
        res = counter.update(make_dummy_keypoints(knee_angle=165.0, back_lean=10.0))

        assert res["rep_count"] == 1
        assert res["shallow_reps"] == 1
        assert res["good_reps"] == 0
        assert res["last_rep_status"] == "too_shallow"

    def test_excessive_forward_lean(self):
        counter = SquatCounter()

        # Standing
        counter.update(make_dummy_keypoints(knee_angle=175.0, back_lean=10.0))
        # Descend with 45 degree back lean (> 38 max)
        for k in [140.0, 110.0, 85.0]:
            counter.update(make_dummy_keypoints(knee_angle=k, back_lean=46.0))
        # Bottom
        counter.update(make_dummy_keypoints(knee_angle=80.0, back_lean=46.0))
        # Ascend
        counter.update(make_dummy_keypoints(knee_angle=130.0, back_lean=40.0))
        # Standing lockout
        res = counter.update(make_dummy_keypoints(knee_angle=165.0, back_lean=15.0))

        assert res["rep_count"] == 1
        assert res["forward_lean_reps"] >= 1
        assert res["last_rep_status"] == "back_leaning_forward"

    def test_hysteresis_prevents_false_reps(self):
        """Small knee wobbles between 152 and 170 deg should never trigger a rep."""
        counter = SquatCounter()
        for k in [170.0, 165.0, 155.0, 152.0, 155.0, 168.0, 172.0]:
            res = counter.update(make_dummy_keypoints(knee_angle=k, back_lean=10.0))

        assert res["rep_count"] == 0
        assert res["state"] == SquatState.STANDING

    def test_occlusion_and_low_confidence(self):
        """Frames with low confidence or None should be ignored safely."""
        counter = SquatCounter()
        res1 = counter.update(None)
        assert res1["knee_angle"] is None
        assert "Step fully into camera" in res1["feedback"]

        res2 = counter.update(make_dummy_keypoints(knee_angle=170.0, back_lean=10.0, confidence=0.3))
        assert res2["knee_angle"] is None

    def test_constructor_compatibility(self):
        """Verifies default, targeted, and kwargs initialization."""
        # 1. Default call without parameters
        c1 = SquatCounter()
        assert c1.target_reps is None
        assert c1.stop_at_target is False
        assert c1.rep_count == 0

        # 2. Hard target call
        c2 = SquatCounter(target_reps=4, stop_at_target=True)
        assert c2.target_reps == 4
        assert c2.stop_at_target is True

        # 3. Soft target call
        c3 = SquatCounter(target_reps=4, stop_at_target=False)
        assert c3.target_reps == 4
        assert c3.stop_at_target is False

        # 4. Unknown kwargs forward-compatibility
        c4 = SquatCounter(arbitrary_extra_param="test", target_reps=5)
        assert c4.target_reps == 5

        # 5. set_target method
        c1.set_target(target_reps=6, stop_at_target=True)
        assert c1.target_reps == 6
        assert c1.stop_at_target is True

    def test_target_limit_enforcement(self):
        """Verifies that counter stops incrementing once target_reps is reached when stop_at_target=True."""
        counter = SquatCounter(target_reps=2, stop_at_target=True)

        # Rep 1
        counter.update(make_dummy_keypoints(175.0, 10.0))
        for k in [140.0, 88.0, 130.0, 165.0]:
            counter.update(make_dummy_keypoints(k, 10.0))
        assert counter.rep_count == 1
        assert counter.target_reached is False

        # Rep 2
        counter.update(make_dummy_keypoints(175.0, 10.0))
        for k in [140.0, 88.0, 130.0, 165.0]:
            res = counter.update(make_dummy_keypoints(k, 10.0))
        assert counter.rep_count == 2
        assert counter.target_reached is True
        assert res["target_reached"] is True

        # Attempt Rep 3 (should be rejected because target reached)
        counter.update(make_dummy_keypoints(175.0, 10.0))
        for k in [140.0, 88.0, 130.0, 165.0]:
            res = counter.update(make_dummy_keypoints(k, 10.0))
        assert res["rep_count"] == 2
        assert counter.rep_count == 2

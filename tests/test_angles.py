"""
Unit Tests for Joint Angle Calculations
Tests knee, hip, and back-lean angle functions with known geometric coordinates.
"""

import pytest
import numpy as np
from pose.angles import (
    calculate_3point_angle,
    calculate_knee_angle,
    calculate_hip_angle,
    calculate_back_lean,
)


class TestJointAngles:
    def test_straight_line_180_degrees(self):
        """Points lying on a straight line should form an interior angle of 180.0 degrees."""
        a = (0.0, 0.0)
        b = (0.0, 1.0)
        c = (0.0, 2.0)
        angle = calculate_3point_angle(a, b, c)
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_right_angle_90_degrees(self):
        """Orthogonal vectors should form a 90.0 degree angle."""
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        c = (0.0, 1.0)
        angle = calculate_3point_angle(a, b, c)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_acute_angle_45_degrees(self):
        """Known 45-degree triangle should evaluate to 45.0 degrees."""
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        c = (1.0, 1.0)
        angle = calculate_3point_angle(a, b, c)
        assert angle == pytest.approx(45.0, abs=0.1)

    def test_knee_angle_standing_vs_squat(self):
        """
        Simulate standing leg (hip directly above knee directly above ankle)
        vs bottom of parallel squat (femur horizontal, tibia vertical).
        """
        # Standing straight upright:
        standing_hip = (0.5, 0.4)
        standing_knee = (0.5, 0.7)
        standing_ankle = (0.5, 1.0)
        standing_knee_angle = calculate_knee_angle(standing_hip, standing_knee, standing_ankle)
        assert standing_knee_angle == pytest.approx(180.0, abs=0.1)

        # Deep/parallel squat (90 degrees at knee):
        squat_hip = (0.2, 0.7)    # Hip back at same height as knee
        squat_knee = (0.5, 0.7)   # Knee joint
        squat_ankle = (0.5, 1.0)  # Foot on floor
        squat_knee_angle = calculate_knee_angle(squat_hip, squat_knee, squat_ankle)
        assert squat_knee_angle == pytest.approx(90.0, abs=0.1)

    def test_hip_angle_standing_vs_squat(self):
        """
        Hip hinge should close from ~180 degrees standing to acute angle during squat.
        """
        # Standing straight:
        shoulder = (0.5, 0.1)
        hip = (0.5, 0.4)
        knee = (0.5, 0.7)
        hip_standing = calculate_hip_angle(shoulder, hip, knee)
        assert hip_standing == pytest.approx(180.0, abs=0.1)

        # Bottom squat:
        squat_shoulder = (0.4, 0.4)
        squat_hip = (0.2, 0.7)
        squat_knee = (0.5, 0.7)
        hip_squat = calculate_hip_angle(squat_shoulder, squat_hip, squat_knee)
        assert 45.0 <= hip_squat <= 90.0

    def test_back_lean_upright_and_inclined(self):
        """
        Back lean measures deviation from vertical axis (0 degrees is vertical).
        """
        # Torso perfectly vertical (0 degrees back lean):
        upright_hip = (0.5, 0.8)
        upright_shoulder = (0.5, 0.3)
        assert calculate_back_lean(upright_shoulder, upright_hip) == pytest.approx(0.0, abs=0.1)

        # Torso leaned forward at 45 degrees:
        # dx = 0.5, dy = 0.5 -> atan2(0.5, 0.5) = 45 degrees
        leaned_hip = (0.5, 0.8)
        leaned_shoulder = (1.0, 0.3)
        assert calculate_back_lean(leaned_shoulder, leaned_hip) == pytest.approx(45.0, abs=0.1)

    def test_dictionary_input_format(self):
        """
        Verify that passing keypoint dictionaries from PoseDetector works transparently.
        """
        pt_a = {"x_norm": 0.5, "y_norm": 0.4, "visibility": 0.99}
        pt_b = {"x_norm": 0.5, "y_norm": 0.7, "visibility": 0.98}
        pt_c = {"x_norm": 0.5, "y_norm": 1.0, "visibility": 0.95}

        angle = calculate_knee_angle(pt_a, pt_b, pt_c)
        assert angle == pytest.approx(180.0, abs=0.1)

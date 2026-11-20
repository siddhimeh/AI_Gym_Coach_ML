"""
Unit Tests for Workout Metrics & Analytics Utility
Validates Form Quality Score, Fatigue Detection, Consistency Metrics, and Summary Aggregations.
"""

import pytest
from utils.metrics import (
    calculate_form_score,
    analyze_fatigue,
    calculate_consistency_metrics,
    calculate_workout_summary,
)


class TestFormQualityScore:
    def test_perfect_reps(self):
        score = calculate_form_score(good_reps=10, shallow_reps=0, forward_lean_reps=0, total_reps=10)
        assert score == 100.0

    def test_zero_reps(self):
        score = calculate_form_score(good_reps=0, shallow_reps=0, forward_lean_reps=0, total_reps=0)
        assert score == 0.0

    def test_weighted_partial_credit(self):
        # 10 reps total: 6 good (6.0) + 2 shallow (0.8) + 2 forward lean (1.0) = 7.8 / 10 = 78.0%
        score = calculate_form_score(good_reps=6, shallow_reps=2, forward_lean_reps=2, total_reps=10)
        assert score == 78.0

    def test_all_shallow_weighted(self):
        # 10 shallow reps get 0.4 credit each = 40.0%
        score = calculate_form_score(good_reps=0, shallow_reps=10, forward_lean_reps=0, total_reps=10)
        assert score == 40.0

    def test_strict_clean_ratio(self):
        # Strict mode gives zero partial credit
        score_strict = calculate_form_score(
            good_reps=8, shallow_reps=2, forward_lean_reps=0, total_reps=10, method="strict"
        )
        assert score_strict == 80.0

        all_shallow_strict = calculate_form_score(
            good_reps=0, shallow_reps=10, forward_lean_reps=0, total_reps=10, method="strict"
        )
        assert all_shallow_strict == 0.0


class TestFatigueAnalysis:
    def test_insufficient_data(self):
        res1 = analyze_fatigue([])
        assert res1["fatigue_detected"] is False
        assert res1["level"] == "INSUFFICIENT_DATA"

        res2 = analyze_fatigue([{"duration_frames": 30, "min_knee_angle": 90.0}])
        assert res2["fatigue_detected"] is False

    def test_stable_biomechanics_no_fatigue(self):
        reps = [
            {"rep_number": 1, "duration_frames": 30, "min_knee_angle": 88.0, "max_back_lean": 20.0},
            {"rep_number": 2, "duration_frames": 30, "min_knee_angle": 88.5, "max_back_lean": 21.0},
            {"rep_number": 3, "duration_frames": 31, "min_knee_angle": 89.0, "max_back_lean": 21.0},
            {"rep_number": 4, "duration_frames": 31, "min_knee_angle": 88.0, "max_back_lean": 22.0},
        ]
        res = analyze_fatigue(reps, fps=15.0)
        assert res["fatigue_detected"] is False
        assert res["level"] == "LOW"
        assert abs(res["tempo_change_pct"]) < 10.0

    def test_tempo_deceleration_fatigue(self):
        # Early reps fast (20 frames), late reps slowed down (32 frames) = +60% slowdown
        reps = [
            {"rep_number": 1, "duration_frames": 20, "min_knee_angle": 88.0, "max_back_lean": 20.0},
            {"rep_number": 2, "duration_frames": 20, "min_knee_angle": 88.0, "max_back_lean": 20.0},
            {"rep_number": 3, "duration_frames": 32, "min_knee_angle": 89.0, "max_back_lean": 22.0},
            {"rep_number": 4, "duration_frames": 32, "min_knee_angle": 89.0, "max_back_lean": 22.0},
        ]
        res = analyze_fatigue(reps, fps=15.0)
        assert res["fatigue_detected"] is True
        assert res["level"] in ("MODERATE", "SIGNIFICANT")
        assert res["tempo_change_pct"] >= 50.0

    def test_depth_decay_fatigue(self):
        # Depth deteriorates from 85 deg to 98 deg (+13 deg drift shallower)
        reps = [
            {"rep_number": 1, "duration_frames": 25, "min_knee_angle": 85.0, "max_back_lean": 20.0},
            {"rep_number": 2, "duration_frames": 25, "min_knee_angle": 85.0, "max_back_lean": 20.0},
            {"rep_number": 3, "duration_frames": 26, "min_knee_angle": 98.0, "max_back_lean": 24.0},
            {"rep_number": 4, "duration_frames": 26, "min_knee_angle": 99.0, "max_back_lean": 25.0},
        ]
        res = analyze_fatigue(reps, fps=15.0)
        assert res["fatigue_detected"] is True
        assert res["depth_drift_deg"] >= 10.0


class TestConsistencyAndSummary:
    def test_consistency_metrics(self):
        identical_reps = [
            {"duration_frames": 30, "min_knee_angle": 90.0},
            {"duration_frames": 30, "min_knee_angle": 90.0},
            {"duration_frames": 30, "min_knee_angle": 90.0},
        ]
        res = calculate_consistency_metrics(identical_reps, fps=15.0)
        assert res["consistency_score"] == 100.0
        assert res["rating"] == "Elite Consistency"

    def test_calculate_workout_summary_integration(self):
        reps = [
            {"rep_number": 1, "duration_frames": 25, "min_knee_angle": 88.0, "max_back_lean": 20.0},
            {"rep_number": 2, "duration_frames": 32, "min_knee_angle": 96.0, "max_back_lean": 22.0},
        ]
        summary = calculate_workout_summary(
            total_reps=2,
            good_reps=1,
            shallow_reps=1,
            forward_lean_reps=0,
            duration_sec=3.8,
            avg_fps=15.0,
            rep_history=reps,
        )

        assert summary["total_reps"] == 2
        assert summary["form_score"] == 70.0  # (1.0 + 0.4) / 2 = 70.0%
        assert summary["strict_score"] == 50.0  # 1 / 2 = 50.0%
        assert "fatigue" in summary
        assert "consistency" in summary

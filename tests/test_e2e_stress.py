"""
Phase 13: End-to-End Stress & Edge-Case Test Suite
Validates system robustness under extreme conditions:
1. Occlusions, blackouts, and malformed landmark packets.
2. Rapid hysteresis boundary oscillations and false-rep rejection.
3. Mid-workout camera side-switching (left vs right).
4. Extreme joint angles (hyperextension, ATG squats, extreme forward lean).
5. SQL injection resistance and database bulk concurrency.
6. LLM/TTS network drop resilience and cooldown enforcement.
7. Full end-to-end workout session lifecycle (Pose -> Counter -> Metrics -> SQLite).
"""

import os
import time
import pytest
import numpy as np

from exercises.squat import SquatCounter, SquatState
from utils.metrics import (
    calculate_form_score,
    analyze_fatigue,
    calculate_consistency_metrics,
    calculate_workout_summary,
)
from db.database import (
    init_db,
    get_or_create_user,
    save_workout,
    get_user_workouts,
    get_user_stats,
    clear_user_workouts,
)
from coach.llm import LLMCoach
from coach.tts import TTSCoach


def make_mock_kps(
    knee_angle: float, back_lean: float, side: str = "right", confidence: float = 0.95
) -> dict:
    """Generates synthetic keypoint dict producing target knee angle and torso lean."""
    rad = np.radians(180.0 - knee_angle)
    hip_x = 0.5 + 0.3 * np.sin(rad)
    hip_y = 0.7 - 0.3 * np.cos(rad)

    lean_rad = np.radians(back_lean)
    shoulder_y = hip_y - 0.4
    shoulder_x = hip_x + 0.4 * np.tan(lean_rad)

    return {
        "side": side,
        "confidence": confidence,
        "points": {
            "shoulder": {"x_norm": float(shoulder_x), "y_norm": float(shoulder_y), "visibility": 0.98},
            "hip": {"x_norm": float(hip_x), "y_norm": float(hip_y), "visibility": 0.98},
            "knee": {"x_norm": 0.5, "y_norm": 0.7, "visibility": 0.98},
            "ankle": {"x_norm": 0.5, "y_norm": 1.0, "visibility": 0.98},
            "heel": {"x_norm": 0.5, "y_norm": 1.05, "visibility": 0.95},
            "foot": {"x_norm": 0.55, "y_norm": 1.05, "visibility": 0.95},
        },
    }


class TestOcclusionAndBoundaryRobustness:
    def test_occlusion_and_blackout_bursts(self):
        """Squat counter must handle 50 consecutive frames of None without throwing exceptions."""
        counter = SquatCounter()
        for _ in range(50):
            res = counter.update(None)
            assert res["state"] == SquatState.STANDING
            assert res["rep_count"] == 0
            assert res["knee_angle"] is None
            assert "Step fully into camera view" in res["feedback"]

    def test_malformed_keypoint_packets(self):
        """Counter must safely reject incomplete or malformed landmark dictionaries."""
        counter = SquatCounter()
        malformed_inputs = [
            {},
            {"confidence": 0.95},
            {"confidence": 0.95, "points": {}},
            {"confidence": 0.95, "points": {"knee": {"x_norm": 0.5, "y_norm": 0.5}}},
            {"confidence": 0.20, "points": {"shoulder": {}, "hip": {}, "knee": {}, "ankle": {}}},
        ]
        for pkt in malformed_inputs:
            res = counter.update(pkt)
            assert res["knee_angle"] is None
            assert res["rep_count"] == 0

    def test_rapid_hysteresis_oscillation_stress(self):
        """Wobbling around 150 deg (descent threshold) should never count a rep."""
        counter = SquatCounter()
        # Rapidly bounce between 148 and 153 degrees 30 times
        for _ in range(30):
            counter.update(make_mock_kps(knee_angle=148.0, back_lean=15.0))
            counter.update(make_mock_kps(knee_angle=153.0, back_lean=15.0))

        # Still incomplete, never reached bottom (<=95) or lockout (>=160)
        assert counter.rep_count == 0

    def test_mid_workout_side_switching(self):
        """Lifter faces left camera side, then turns around to right camera side."""
        counter = SquatCounter()
        # Rep 1 on left side
        for k in [175.0, 140.0, 85.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=15.0, side="left"))
        assert counter.rep_count == 1
        assert counter.good_reps == 1

        # Rep 2 on right side
        for k in [175.0, 140.0, 85.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=15.0, side="right"))
        assert counter.rep_count == 2
        assert counter.good_reps == 2

    def test_extreme_kinematics(self):
        """Handles deep ATG squats (45 deg) and extreme forward leans (65 deg) safely."""
        counter = SquatCounter()
        # Stand
        counter.update(make_mock_kps(knee_angle=185.0, back_lean=-5.0))
        # Drop to extreme deep squat with heavy forward lean
        counter.update(make_mock_kps(knee_angle=140.0, back_lean=30.0))
        counter.update(make_mock_kps(knee_angle=48.0, back_lean=65.0))
        counter.update(make_mock_kps(knee_angle=130.0, back_lean=50.0))
        res = counter.update(make_mock_kps(knee_angle=168.0, back_lean=15.0))

        assert res["rep_count"] == 1
        assert res["forward_lean_reps"] == 1
        assert res["last_rep_status"] == "back_leaning_forward"


class TestDatabaseSecurityAndStress:
    def test_sql_injection_resistance(self, tmp_path):
        """Tests parameterized security against SQL injection payloads."""
        db_path = os.path.join(tmp_path, "security_test.db")
        init_db(db_path)

        injections = [
            "admin' OR '1'='1",
            "'; DROP TABLE workouts; --",
            "<script>alert('xss')</script>",
            "athlete_🚀_unicode",
            "   ",  # Empty/whitespace fallback
        ]

        for user_payload in injections:
            user = get_or_create_user(user_payload, db_path)
            assert user["id"] > 0
            assert isinstance(user["username"], str)

            w_id = save_workout({
                "username": user_payload,
                "exercise_name": "Squats",
                "planned_sets": 1,
                "planned_reps": 10,
                "completed_reps": 10,
                "good_reps": 9,
                "shallow_reps": 1,
                "forward_lean_reps": 0,
                "form_score": 94.0,
                "duration_sec": 45.0,
                "avg_fps": 15.0,
            }, db_path)
            assert w_id > 0

            history = get_user_workouts(user_payload, db_path)
            assert len(history) >= 1

    def test_database_bulk_volume_stress(self, tmp_path):
        """Inserts 50 sessions in sequence and verifies mathematical aggregate stats."""
        db_path = os.path.join(tmp_path, "bulk_test.db")
        init_db(db_path)

        user = "stress_athlete"
        for i in range(1, 51):
            save_workout({
                "username": user,
                "exercise_name": "Squats",
                "planned_sets": 3,
                "planned_reps": 10,
                "completed_reps": 10,
                "good_reps": 8,
                "shallow_reps": 2,
                "forward_lean_reps": 0,
                "form_score": 88.0,
                "duration_sec": 40.0,
                "avg_fps": 15.0,
            }, db_path)

        stats = get_user_stats(user, db_path)
        assert stats["total_workouts"] == 50
        assert stats["total_reps"] == 500
        assert stats["clean_reps"] == 400
        assert stats["clean_ratio"] == 80.0
        assert stats["avg_form_score"] == 88.0

        # Test clear workouts
        deleted = clear_user_workouts(user, db_path)
        assert deleted == 50
        assert len(get_user_workouts(user, db_path)) == 0


class TestCoachResilience:
    def test_llm_coach_graceful_fallback(self):
        """LLM coach must return rule fallback cleanly without crashing when forced or offline."""
        coach = LLMCoach(cooldown=0.1)
        # Forced fallback
        msg1 = coach.generate_feedback(status="too_shallow", knee_angle=110.0, back_lean=20.0, force_fallback=True)
        assert isinstance(msg1, str)
        assert len(msg1) > 5

        # With no API key
        msg2 = coach.generate_feedback(status="good", knee_angle=85.0, back_lean=22.0)
        assert isinstance(msg2, str)
        assert "depth" in msg2 and "upright" in msg2

    def test_tts_throttling_and_offline_safety(self):
        """TTS coach must enforce cooldown and never raise an uncaught exception."""
        coach = TTSCoach(cooldown=5.0)
        assert coach.speak("") is None
        assert coach.speak("   ") is None

        # First call might create or check cache
        res = coach.speak("Test audio cue", force=True)
        # Second call within 0.1s should be throttled (return None)
        throttled = coach.speak("Another audio cue")
        assert throttled is None


class TestFullWorkoutLifecycleE2E:
    def test_full_session_pipeline_e2e(self, tmp_path):
        """
        Simulates an entire workout session from vision stream to SQLite storage:
        - Rep 1: Clean squat
        - Rep 2: Clean squat
        - Rep 3: Shallow squat
        - Glitch: 4 frames of blackout/occlusion
        - Rep 4: Forward lean squat
        - Rep 5: Clean squat
        """
        counter = SquatCounter()
        test_db = os.path.join(tmp_path, "e2e_lifecycle.db")

        # Rep 1: Clean
        for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=20.0))

        # Rep 2: Clean
        for k in [175.0, 140.0, 90.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=22.0))

        # Rep 3: Shallow (stops at 118 deg)
        for k in [175.0, 140.0, 125.0, 118.0, 118.0, 118.0, 130.0, 150.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=20.0))

        # Blackout glitch
        for _ in range(4):
            counter.update(None)

        # Rep 4: Forward lean (reaches 86 deg depth, but 44 deg torso lean)
        for k in [175.0, 140.0, 86.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=44.0))

        # Rep 5: Clean
        for k in [175.0, 140.0, 89.0, 130.0, 165.0]:
            counter.update(make_mock_kps(knee_angle=k, back_lean=22.0))

        # Assert counter metrics
        assert counter.rep_count == 5
        assert counter.good_reps == 3
        assert counter.shallow_reps == 1
        assert counter.forward_lean_reps == 1

        rep_history = counter.get_rep_history()
        assert len(rep_history) == 5
        assert rep_history[0]["status"] == "good"
        assert rep_history[2]["status"] == "too_shallow"
        assert rep_history[3]["status"] == "back_leaning_forward"

        # Calculate workout summary
        summary = calculate_workout_summary(
            total_reps=counter.rep_count,
            good_reps=counter.good_reps,
            shallow_reps=counter.shallow_reps,
            forward_lean_reps=counter.forward_lean_reps,
            duration_sec=35.0,
            avg_fps=15.0,
            rep_history=rep_history,
        )

        # Weighted: (3*1.0 + 1*0.4 + 1*0.5) / 5 * 100 = 3.9 / 5 = 78.0%
        assert summary["form_score"] == 78.0
        # Strict: 3 / 5 = 60.0%
        assert summary["strict_score"] == 60.0
        assert summary["duration_sec"] == 35.0
        assert "fatigue" in summary
        assert "consistency" in summary

        # Persist to SQLite
        workout_rec = {
            "username": "e2e_athlete",
            "exercise_name": "Squats",
            "planned_sets": 1,
            "planned_reps": 5,
            "completed_reps": summary["total_reps"],
            "good_reps": summary["good_reps"],
            "shallow_reps": summary["shallow_reps"],
            "forward_lean_reps": summary["forward_lean_reps"],
            "form_score": summary["form_score"],
            "duration_sec": summary["duration_sec"],
            "avg_fps": summary["avg_fps"],
        }
        sid = save_workout(workout_rec, test_db)
        assert sid > 0

        # Verify retrieval
        saved_workouts = get_user_workouts("e2e_athlete", test_db)
        assert len(saved_workouts) == 1
        s = saved_workouts[0]
        assert s["reps"] == 5
        assert s["good"] == 3
        assert s["shallow"] == 1
        assert s["forward_lean"] == 1
        assert s["score"] == 78.0

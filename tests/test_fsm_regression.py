"""
Automated Regression Test Suite for Squat Telemetry & FSM Logic
Covers all 15 scenarios specified in Step 8:
 1. Standing still for 60 seconds -> 0 reps.
 2. Standing angle fluctuations -> 0 reps.
 3. One-frame knee-angle spike -> 0 reps.
 4. Temporary landmark loss -> no false rep.
 5. Side-selection switching -> no phantom rep.
 6. One complete deep squat -> exactly 1 rep (100% clean).
 7. One genuine shallow squat -> exactly 1 shallow rep.
 8. Incomplete squat -> 0 completed reps.
 9. Three valid squats with hard target of 3 -> exactly 3 reps.
10. Fourth squat after hard stop -> count remains 3.
11. Duplicate completion event -> no extra count.
12. New workout -> all counters reset.
13. Zero completed reps -> Clean = 0, Shallow = 0, Form Score = 0.
14. Streamlit rerun -> no duplicate counts.
15. SQLite saved totals match the workout summary.
"""

import os
import pytest
import numpy as np

from exercises.squat import SquatCounter, SquatState
from utils.metrics import calculate_workout_summary, calculate_form_score
from db.database import init_db, save_workout, get_user_workouts
from tests.test_squat_counter import make_dummy_keypoints


def make_kps(knee_angle: float, back_lean: float = 10.0, side: str = "right", confidence: float = 0.95) -> dict:
    """Helper returning synthetic keypoints for testing."""
    return make_dummy_keypoints(knee_angle=knee_angle, back_lean=back_lean, confidence=confidence)


class TestFSMRegression:
    def test_1_standing_still_60s(self):
        """1. Standing still for 60 seconds (1800 frames @ 30 FPS) must produce zero reps."""
        counter = SquatCounter()
        for _ in range(1800):
            res = counter.update(make_kps(knee_angle=172.0, back_lean=8.0))

        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0
        assert counter.forward_lean_reps == 0
        assert counter.state == SquatState.STANDING
        assert res["rep_count"] == 0

    def test_2_standing_angle_fluctuations(self):
        """2. Standing posture sway fluctuating between 146° and 175° must produce zero reps."""
        counter = SquatCounter()
        # Simulate subtle standing shifts: wobbling near descent initiate threshold
        fluctuations = [170.0, 165.0, 155.0, 148.0, 146.0, 152.0, 162.0, 172.0, 168.0, 149.0, 158.0, 170.0]
        for _ in range(25):
            for k in fluctuations:
                counter.update(make_kps(knee_angle=k, back_lean=10.0))

        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0
        assert counter.state == SquatState.STANDING

    def test_3_one_frame_knee_angle_spike(self):
        """3. An isolated single-frame angle glitch (e.g. 170 -> 80 -> 170) must not register a rep."""
        counter = SquatCounter()
        # Normal standing
        for _ in range(5):
            counter.update(make_kps(knee_angle=170.0))

        # Single frame glitch
        counter.update(make_kps(knee_angle=80.0))

        # Back to standing
        for _ in range(10):
            counter.update(make_kps(knee_angle=170.0))

        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0

    def test_4_temporary_landmark_loss_no_false_rep(self):
        """4. Descent initiated followed by tracking loss (20 frames of None) must abort safely."""
        counter = SquatCounter()
        # Start descent
        counter.update(make_kps(knee_angle=170.0))
        counter.update(make_kps(knee_angle=145.0))

        # 20 frames of occlusion / landmark loss
        for _ in range(20):
            counter.update(None)

        # Person reappears standing
        for _ in range(10):
            res = counter.update(make_kps(knee_angle=170.0))

        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0
        assert res["state"] == SquatState.STANDING

    def test_5_side_selection_switching_no_phantom_rep(self):
        """5. Rapid side-selection changes while standing must produce zero reps."""
        counter = SquatCounter()
        for i in range(100):
            side = "left" if i % 2 == 0 else "right"
            k = 168.0 if side == "left" else 164.0
            kp = make_kps(knee_angle=k, back_lean=10.0)
            kp["side"] = side
            counter.update(kp)

        assert counter.rep_count == 0
        assert counter.state == SquatState.STANDING

    def test_6_one_complete_deep_squat(self):
        """6. Exactly one complete deep squat cycle produces exactly 1 clean repetition."""
        counter = SquatCounter()
        # Standing
        counter.update(make_kps(knee_angle=175.0, back_lean=10.0))
        # Descent
        for k in [145.0, 130.0, 115.0, 100.0]:
            counter.update(make_kps(knee_angle=k, back_lean=18.0))
        # Bottom (<= 95)
        for k in [90.0, 80.0, 85.0]:
            counter.update(make_kps(knee_angle=k, back_lean=22.0))
        # Ascent
        for k in [110.0, 130.0, 150.0]:
            counter.update(make_kps(knee_angle=k, back_lean=18.0))
        # Standing lockout
        res = counter.update(make_kps(knee_angle=165.0, back_lean=10.0))

        assert res["rep_count"] == 1
        assert res["good_reps"] == 1
        assert res["shallow_reps"] == 0
        assert res["forward_lean_reps"] == 0
        assert res["last_rep_status"] == "good"

        score = calculate_form_score(res["good_reps"], res["shallow_reps"], res["forward_lean_reps"], res["rep_count"])
        assert score == 100.0

    def test_7_one_genuine_shallow_squat(self):
        """7. One genuine shallow squat (descends to 125°) produces exactly 1 shallow repetition."""
        counter = SquatCounter()
        # Standing
        counter.update(make_kps(knee_angle=175.0, back_lean=10.0))
        # Descent to 125° (valid shallow range <= 140°)
        for k in [145.0, 135.0, 128.0, 125.0, 125.0]:
            counter.update(make_kps(knee_angle=k, back_lean=12.0))
        # Turnaround and rising
        for k in [138.0, 148.0, 155.0]:
            counter.update(make_kps(knee_angle=k, back_lean=12.0))
        # Lockout
        res = counter.update(make_kps(knee_angle=165.0, back_lean=10.0))

        assert res["rep_count"] == 1
        assert res["good_reps"] == 0
        assert res["shallow_reps"] == 1
        assert res["last_rep_status"] == "too_shallow"

    def test_8_incomplete_squat_no_count(self):
        """8. An aborted squat (drops to 144°, never reaches 140°) produces 0 completed reps."""
        counter = SquatCounter()
        # Standing
        counter.update(make_kps(knee_angle=175.0))
        # Slight knee dip to 144°
        for k in [148.0, 144.0, 145.0]:
            counter.update(make_kps(knee_angle=k))
        # Returns directly to standing
        res = counter.update(make_kps(knee_angle=165.0))

        assert res["rep_count"] == 0
        assert res["good_reps"] == 0
        assert res["shallow_reps"] == 0
        assert res["state"] == SquatState.STANDING

    def test_9_three_valid_squats_hard_target_three(self):
        """9. Performing 3 valid squats with a hard target of 3 records exactly 3 repetitions."""
        counter = SquatCounter(target_reps=3, stop_at_target=True)

        for rep in range(3):
            # Stand
            counter.update(make_kps(knee_angle=175.0))
            # Descend
            for k in [140.0, 115.0, 88.0]:
                counter.update(make_kps(knee_angle=k))
            # Ascend
            for k in [120.0, 145.0]:
                counter.update(make_kps(knee_angle=k))
            # Lockout
            res = counter.update(make_kps(knee_angle=165.0))

        assert res["rep_count"] == 3
        assert res["good_reps"] == 3
        assert res["target_reached"] is True

    def test_10_fourth_squat_after_hard_stop_remains_three(self):
        """10. Attempting a 4th squat after reaching a hard target of 3 keeps count locked at 3."""
        counter = SquatCounter(target_reps=3, stop_at_target=True)

        # Complete 3 reps
        for _ in range(3):
            counter.update(make_kps(knee_angle=175.0))
            for k in [140.0, 88.0, 130.0, 165.0]:
                counter.update(make_kps(knee_angle=k))

        assert counter.rep_count == 3

        # Attempt 4th squat
        for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
            res = counter.update(make_kps(knee_angle=k))

        assert res["rep_count"] == 3
        assert res["good_reps"] == 3
        assert res["target_reached"] is True

    def test_11_duplicate_completion_event_no_extra_count(self):
        """11. Remaining standing at 165° after rep completion must not duplicate counts."""
        counter = SquatCounter()
        # Complete 1 rep
        for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
            counter.update(make_kps(knee_angle=k))

        assert counter.rep_count == 1

        # Send 30 additional standing lockout frames
        for _ in range(30):
            res = counter.update(make_kps(knee_angle=168.0))

        assert res["rep_count"] == 1
        assert counter.rep_count == 1

    def test_12_new_workout_resets_all_counters(self):
        """12. Resetting the counter clears all repetition states for a fresh workout."""
        counter = SquatCounter()
        # Perform 1 rep
        for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
            counter.update(make_kps(knee_angle=k))
        assert counter.rep_count == 1

        # Reset
        counter.reset()
        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0
        assert counter.forward_lean_reps == 0
        assert counter.rep_history == []
        assert counter.state == SquatState.STANDING
        assert counter.target_reached is False

    def test_13_zero_completed_reps_metrics(self):
        """13. A standing-only workout (0 reps) evaluates to Clean=0, Shallow=0, Score=0."""
        summary = calculate_workout_summary(
            total_reps=0,
            good_reps=0,
            shallow_reps=0,
            forward_lean_reps=0,
            duration_sec=25.0,
            avg_fps=16.0,
            rep_history=[],
        )
        assert summary["total_reps"] == 0
        assert summary["good_reps"] == 0
        assert summary["shallow_reps"] == 0
        assert summary["forward_lean_reps"] == 0
        assert summary["form_score"] == 0.0
        assert summary["strict_score"] == 0.0

    def test_14_streamlit_rerun_no_duplicate_database_writes(self, tmp_path):
        """14. Multiple Streamlit script reruns must not insert duplicate database records."""
        db_path = os.path.join(tmp_path, "rerun_test.db")
        init_db(db_path)

        user = "rerun_athlete"
        session_state_sim = {"session_saved": False}

        # Simulated save guard logic from Live_Workout.py
        def simulate_session_end(rep_count, good_reps):
            if rep_count > 0:
                if not session_state_sim.get("session_saved", False):
                    save_workout({
                        "username": user,
                        "exercise_name": "Squats",
                        "planned_sets": 1,
                        "planned_reps": 3,
                        "completed_reps": rep_count,
                        "good_reps": good_reps,
                        "shallow_reps": 0,
                        "forward_lean_reps": 0,
                        "form_score": 100.0,
                        "duration_sec": 30.0,
                        "avg_fps": 15.0,
                    }, db_path)
                    session_state_sim["session_saved"] = True

        # First exit: saves to SQLite
        simulate_session_end(rep_count=3, good_reps=3)
        assert session_state_sim["session_saved"] is True

        # Subsequent Streamlit reruns: must be blocked by session_saved guard
        simulate_session_end(rep_count=3, good_reps=3)
        simulate_session_end(rep_count=3, good_reps=3)

        records = get_user_workouts(user, db_path)
        assert len(records) == 1

    def test_15_sqlite_saved_totals_match_workout_summary(self, tmp_path):
        """15. Persisted SQLite workout totals exactly match the in-memory summary."""
        db_path = os.path.join(tmp_path, "summary_match.db")
        init_db(db_path)

        counter = SquatCounter()
        for _ in range(3):
            for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
                counter.update(make_kps(knee_angle=k))

        summary = calculate_workout_summary(
            total_reps=counter.rep_count,
            good_reps=counter.good_reps,
            shallow_reps=counter.shallow_reps,
            forward_lean_reps=counter.forward_lean_reps,
            duration_sec=28.5,
            avg_fps=15.0,
            rep_history=counter.get_rep_history(),
        )

        w_id = save_workout({
            "username": "parity_athlete",
            "exercise_name": "Squats",
            "planned_sets": 1,
            "planned_reps": 3,
            "completed_reps": summary["total_reps"],
            "good_reps": summary["good_reps"],
            "shallow_reps": summary["shallow_reps"],
            "forward_lean_reps": summary["forward_lean_reps"],
            "form_score": summary["form_score"],
            "duration_sec": summary["duration_sec"],
            "avg_fps": summary["avg_fps"],
        }, db_path)

        db_records = get_user_workouts("parity_athlete", db_path)
        assert len(db_records) == 1
        rec = db_records[0]

        assert rec["reps"] == summary["total_reps"] == 3
        assert rec["good"] == summary["good_reps"] == 3
        assert rec["shallow"] == summary["shallow_reps"] == 0
        assert rec["forward_lean"] == summary["forward_lean_reps"] == 0
        assert rec["score"] == summary["form_score"] == 100.0

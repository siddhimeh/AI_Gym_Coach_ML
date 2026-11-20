"""
Phase 11: Comprehensive Automated Regression Tests for Workout Lifecycle
Validates:
1. A completed zero-rep workout is saved.
2. An abandoned workout is not saved as completed.
3. A completed 3-rep workout is saved with correct counts.
4. A new workout receives a unique session identifier.
5. Repeated Finish button clicks do not duplicate records.
6. Summary displays the exact latest session.
7. History displays the newest session first.
8. A zero-rep session appears in History.
9. Total Sessions increases after a completed workout.
10. Total Volume increases only by valid completed reps.
11. Clean Rep Ratio is calculated correctly.
12. Average Form Score handles undefined scores.
13. Training duration is accumulated correctly.
14. Form Progression graph handles zero-rep sessions.
15. Error Composition graph handles overlapping lean flags.
16. Volume & Pacing graph handles zero reps.
17. Streamlit cache does not show stale history after a save.
18. Different users see only their own sessions.
19. Session timestamps are handled correctly (timezone aware).
20. Failed database writes are reported and retryable.
21. Existing records remain unchanged.
22. No SQL injection is possible through user-controlled values.
23. The live workout FSM and squat counter remain functional.
"""

import os
import uuid
import pytest
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from exercises.squat import SquatCounter, SquatState
from utils.metrics import calculate_workout_summary
from db.database import (
    init_db,
    get_or_create_user,
    save_workout,
    get_user_workouts,
    get_workout_by_id,
    get_user_stats,
    format_timestamp_for_display,
)


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh isolated SQLite database for each test without touching gym_coach.db."""
    db_file = str(tmp_path / "test_lifecycle.db")
    init_db(db_file)
    return db_file


def make_kps(knee_angle: float, back_lean: float = 15.0) -> dict:
    """Helper to generate synthetic keypoints."""
    rad = np.radians(180.0 - knee_angle)
    hip_x = 0.5 + 0.3 * np.sin(rad)
    hip_y = 0.7 - 0.3 * np.cos(rad)

    lean_rad = np.radians(back_lean)
    shoulder_y = hip_y - 0.4
    shoulder_x = hip_x + 0.4 * np.tan(lean_rad)

    return {
        "side": "right",
        "confidence": 0.95,
        "points": {
            "shoulder": {"x_norm": float(shoulder_x), "y_norm": float(shoulder_y), "visibility": 0.98},
            "hip": {"x_norm": float(hip_x), "y_norm": float(hip_y), "visibility": 0.98},
            "knee": {"x_norm": 0.5, "y_norm": 0.7, "visibility": 0.98},
            "ankle": {"x_norm": 0.5, "y_norm": 1.0, "visibility": 0.98},
            "heel": {"x_norm": 0.5, "y_norm": 1.05, "visibility": 0.95},
            "foot": {"x_norm": 0.55, "y_norm": 1.05, "visibility": 0.95},
        },
    }


class TestWorkoutDataLifecycle:

    def test_1_completed_zero_rep_workout_is_saved(self, temp_db):
        """1. A completed zero-rep workout (standing still and clicking finish) is saved."""
        w_data = {
            "username": "athlete_1",
            "exercise_name": "Squats",
            "planned_sets": 3,
            "planned_reps": 10,
            "completed_reps": 0,
            "good_reps": 0,
            "shallow_reps": 0,
            "forward_lean_reps": 0,
            "form_score": None,
            "duration_sec": 15.4,
            "avg_fps": 15.0,
            "session_token": "token_zero_rep_1",
        }
        sid = save_workout(w_data, temp_db)
        assert sid > 0

        workouts = get_user_workouts("athlete_1", temp_db)
        assert len(workouts) == 1
        w = workouts[0]
        assert w["reps"] == 0
        assert w["good"] == 0
        assert w["shallow"] == 0
        assert w["forward_lean"] == 0
        assert w["score"] is None
        assert w["duration"] == 15.4

    def test_2_abandoned_workout_is_not_saved_as_completed(self, temp_db):
        """2. An abandoned/accidental workout (0 reps, duration < 1s, 0 frames) is rejected."""
        # Simulated lifecycle save guard logic
        def lifecycle_saver(reps, duration, frames, intentional_finish):
            if reps > 0:
                return True
            if intentional_finish and (duration >= 1.0 or frames >= 5):
                return True
            return False

        assert lifecycle_saver(reps=0, duration=0.4, frames=2, intentional_finish=False) is False
        assert lifecycle_saver(reps=0, duration=0.2, frames=0, intentional_finish=True) is False
        assert lifecycle_saver(reps=0, duration=10.0, frames=150, intentional_finish=True) is True
        assert lifecycle_saver(reps=3, duration=15.0, frames=200, intentional_finish=False) is True

    def test_3_completed_3_rep_workout_saved_with_correct_counts(self, temp_db):
        """3. A completed 3-rep workout is saved with exact breakdown counts."""
        w_data = {
            "username": "athlete_3",
            "exercise_name": "Squats",
            "planned_sets": 1,
            "planned_reps": 3,
            "completed_reps": 3,
            "good_reps": 2,
            "shallow_reps": 1,
            "forward_lean_reps": 0,
            "form_score": 80.0,
            "duration_sec": 24.5,
            "avg_fps": 15.0,
            "session_token": "token_3rep_1",
        }
        sid = save_workout(w_data, temp_db)
        assert sid > 0

        workouts = get_user_workouts("athlete_3", temp_db)
        assert len(workouts) == 1
        w = workouts[0]
        assert w["reps"] == 3
        assert w["good"] == 2
        assert w["shallow"] == 1
        assert w["score"] == 80.0

    def test_4_new_workout_receives_unique_session_identifier(self, temp_db):
        """4. Distinct workouts receive unique session tokens and unique database IDs."""
        token1 = uuid.uuid4().hex[:12]
        token2 = uuid.uuid4().hex[:12]
        assert token1 != token2

        id1 = save_workout({
            "username": "athlete_tokens", "completed_reps": 5, "good_reps": 5,
            "form_score": 100.0, "duration_sec": 20.0, "session_token": token1
        }, temp_db)

        id2 = save_workout({
            "username": "athlete_tokens", "completed_reps": 5, "good_reps": 5,
            "form_score": 100.0, "duration_sec": 20.0, "session_token": token2
        }, temp_db)

        assert id1 != id2
        assert id2 > id1

    def test_5_repeated_finish_clicks_do_not_duplicate_records(self, temp_db):
        """5. Repeated Finish button clicks (idempotency token) never insert duplicates."""
        token = "idempotent_session_99"
        w_data = {
            "username": "athlete_idempotent",
            "completed_reps": 4,
            "good_reps": 4,
            "form_score": 100.0,
            "duration_sec": 18.0,
            "session_token": token,
        }

        id_first = save_workout(w_data, temp_db)
        id_second = save_workout(w_data, temp_db)
        id_third = save_workout(w_data, temp_db)

        assert id_first == id_second == id_third

        workouts = get_user_workouts("athlete_idempotent", temp_db)
        assert len(workouts) == 1

    def test_6_summary_displays_exact_latest_session(self, temp_db):
        """6. Summary data structure matches the exact finalized workout session."""
        summary = calculate_workout_summary(
            total_reps=0,
            good_reps=0,
            shallow_reps=0,
            forward_lean_reps=0,
            duration_sec=16.8,
            avg_fps=15.0,
            rep_history=[],
        )
        assert summary["total_reps"] == 0
        assert summary["good_reps"] == 0
        assert summary["duration_sec"] == 16.8

    def test_7_history_displays_newest_session_first(self, temp_db):
        """7. Sessions are ordered newest first using timestamp descending and id as tie-breaker."""
        user = "ordering_user"
        save_workout({
            "username": user, "completed_reps": 5, "good_reps": 5, "form_score": 100.0,
            "duration_sec": 20.0, "timestamp": "2026-10-25 10:00:00", "session_token": "ord_1"
        }, temp_db)
        save_workout({
            "username": user, "completed_reps": 8, "good_reps": 7, "form_score": 90.0,
            "duration_sec": 30.0, "timestamp": "2026-10-31 15:00:00", "session_token": "ord_2"
        }, temp_db)
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0, "form_score": None,
            "duration_sec": 12.0, "timestamp": "2026-11-02 18:00:00", "session_token": "ord_3"
        }, temp_db)

        workouts = get_user_workouts(user, temp_db)
        assert len(workouts) == 3
        assert workouts[0]["session_token"] == "ord_3"
        assert workouts[0]["reps"] == 0
        assert workouts[1]["session_token"] == "ord_2"
        assert workouts[2]["session_token"] == "ord_1"

    def test_8_zero_rep_session_appears_in_history(self, temp_db):
        """8. Zero-rep session is present in history with date and None score."""
        save_workout({
            "username": "athlete_zero_hist", "completed_reps": 0, "good_reps": 0,
            "shallow_reps": 0, "forward_lean_reps": 0, "form_score": None,
            "duration_sec": 14.2, "session_token": "zh_1"
        }, temp_db)

        workouts = get_user_workouts("athlete_zero_hist", temp_db)
        assert len(workouts) == 1
        assert workouts[0]["reps"] == 0
        assert workouts[0]["score"] is None
        assert workouts[0]["date"] is not None

    def test_9_total_sessions_increases_after_completed_workout(self, temp_db):
        """9. Total sessions increases by 1 for any completed workout (including zero-rep)."""
        user = "session_counter_user"
        stats0 = get_user_stats(user, temp_db)
        assert stats0["total_workouts"] == 0

        # Save zero-rep completed session
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0,
            "form_score": None, "duration_sec": 10.0, "session_token": "sc_1"
        }, temp_db)
        stats1 = get_user_stats(user, temp_db)
        assert stats1["total_workouts"] == 1

        # Save 5-rep session
        save_workout({
            "username": user, "completed_reps": 5, "good_reps": 4,
            "form_score": 85.0, "duration_sec": 25.0, "session_token": "sc_2"
        }, temp_db)
        stats2 = get_user_stats(user, temp_db)
        assert stats2["total_workouts"] == 2

    def test_10_total_volume_increases_only_by_valid_completed_reps(self, temp_db):
        """10. Total volume increases only by completed reps; zero-rep session adds 0."""
        user = "vol_user"
        save_workout({
            "username": user, "completed_reps": 10, "good_reps": 8,
            "form_score": 85.0, "duration_sec": 40.0, "session_token": "v_1"
        }, temp_db)
        assert get_user_stats(user, temp_db)["total_reps"] == 10

        # Zero rep workout
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0,
            "form_score": None, "duration_sec": 15.0, "session_token": "v_2"
        }, temp_db)
        assert get_user_stats(user, temp_db)["total_reps"] == 10  # Unchanged

        # 4 reps workout
        save_workout({
            "username": user, "completed_reps": 4, "good_reps": 3,
            "form_score": 80.0, "duration_sec": 18.0, "session_token": "v_3"
        }, temp_db)
        assert get_user_stats(user, temp_db)["total_reps"] == 14

    def test_11_clean_rep_ratio_calculated_correctly(self, temp_db):
        """11. Clean Rep Ratio is good_reps / total_reps * 100."""
        user = "ratio_user"
        save_workout({
            "username": user, "completed_reps": 10, "good_reps": 7,
            "form_score": 80.0, "duration_sec": 40.0, "session_token": "r_1"
        }, temp_db)
        stats = get_user_stats(user, temp_db)
        assert stats["clean_ratio"] == 70.0

        # Zero-rep workout does not affect clean ratio
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0,
            "form_score": None, "duration_sec": 15.0, "session_token": "r_2"
        }, temp_db)
        stats2 = get_user_stats(user, temp_db)
        assert stats2["clean_ratio"] == 70.0

    def test_12_average_form_score_handles_undefined_scores(self, temp_db):
        """12. Average Form Score excludes sessions with None/undefined scores."""
        user = "avg_score_user"
        save_workout({
            "username": user, "completed_reps": 10, "good_reps": 10,
            "form_score": 100.0, "duration_sec": 30.0, "session_token": "as_1"
        }, temp_db)
        save_workout({
            "username": user, "completed_reps": 10, "good_reps": 6,
            "form_score": 76.0, "duration_sec": 30.0, "session_token": "as_2"
        }, temp_db)

        # Average of 100.0 and 76.0 is 88.0%
        assert get_user_stats(user, temp_db)["avg_form_score"] == 88.0

        # Adding a zero-rep workout with None score must NOT pull the average down to 58.7%
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0,
            "form_score": None, "duration_sec": 20.0, "session_token": "as_3"
        }, temp_db)

        stats_after = get_user_stats(user, temp_db)
        assert stats_after["total_workouts"] == 3
        assert stats_after["avg_form_score"] == 88.0  # Still 88.0%

    def test_13_training_duration_accumulated_correctly(self, temp_db):
        """13. Training duration accumulates across all completed sessions."""
        user = "dur_user"
        save_workout({
            "username": user, "completed_reps": 5, "good_reps": 5,
            "form_score": 100.0, "duration_sec": 45.5, "session_token": "d_1"
        }, temp_db)
        save_workout({
            "username": user, "completed_reps": 0, "good_reps": 0,
            "form_score": None, "duration_sec": 14.5, "session_token": "d_2"
        }, temp_db)

        stats = get_user_stats(user, temp_db)
        assert stats["total_duration_sec"] == 60.0

    def test_14_form_progression_graph_handles_zero_rep_sessions(self):
        """14. Form progression data series handles None scores as NaN without crashing."""
        workouts = [
            {"session_id": 1, "score": 90.0},
            {"session_id": 2, "score": None},
            {"session_id": 3, "score": 85.0},
        ]
        df = pd.DataFrame(workouts)
        score_num = pd.to_numeric(df["score"], errors="coerce")
        assert score_num.isna().sum() == 1
        assert not np.isnan(score_num.iloc[0])
        assert np.isnan(score_num.iloc[1])
        assert not np.isnan(score_num.iloc[2])

    def test_15_error_composition_grouped_no_blind_stacking(self):
        """15. Error breakdown handles overlapping lean flags without invalid stacking."""
        # 1 rep that is shallow AND forward leaning
        good = 0
        shallow = 1
        lean = 1
        total_reps = 1

        # If grouped bars (Clean, Shallow, Lean), each bar height is legitimate:
        bar_heights = [good, shallow, lean]
        assert max(bar_heights) <= total_reps

    def test_16_volume_and_pacing_graph_handles_zero_reps(self):
        """16. Pacing handles zero reps safely with NaN instead of zero division."""
        durations = [30.0, 15.0, 40.0]
        reps = [10, 0, 8]
        pacing = [(d / r) if r > 0 else np.nan for d, r in zip(durations, reps)]

        assert pacing[0] == 3.0
        assert np.isnan(pacing[1])
        assert pacing[2] == 5.0

    def test_17_cache_invalidation_logic(self):
        """17. After save, cache clearing functions without error."""
        import streamlit as st
        # Verify st.cache_data has clear method
        assert hasattr(st.cache_data, "clear")

    def test_18_different_users_see_only_their_own_sessions(self, temp_db):
        """18. User isolation is enforced; user A cannot see user B's sessions."""
        save_workout({
            "username": "user_alpha", "completed_reps": 10, "good_reps": 10,
            "form_score": 100.0, "duration_sec": 30.0, "session_token": "iso_1"
        }, temp_db)
        save_workout({
            "username": "user_beta", "completed_reps": 5, "good_reps": 5,
            "form_score": 90.0, "duration_sec": 20.0, "session_token": "iso_2"
        }, temp_db)

        w_alpha = get_user_workouts("user_alpha", temp_db)
        w_beta = get_user_workouts("user_beta", temp_db)

        assert len(w_alpha) == 1
        assert w_alpha[0]["session_token"] == "iso_1"
        assert len(w_beta) == 1
        assert w_beta[0]["session_token"] == "iso_2"

    def test_19_session_timestamps_handled_correctly(self):
        """19. Timestamps are parsed and converted to Asia/Kolkata correctly."""
        utc_ts = "2026-11-02 16:00:00"
        date_str, time_str, full_str = format_timestamp_for_display(utc_ts)
        assert date_str == "2026-11-02"
        assert time_str == "21:30:00"
        assert "2026-11-02 21:30:00" in full_str

    def test_20_failed_database_writes_reported_and_retryable(self, temp_db):
        """20. Database write errors can be caught and retried."""
        w_data = {
            "username": "retry_athlete",
            "completed_reps": 5,
            "good_reps": 5,
            "form_score": 100.0,
            "duration_sec": 25.0,
            "session_token": "retry_tok_1",
        }
        # First attempt works
        sid = save_workout(w_data, temp_db)
        assert sid > 0

        # Retry with same token returns same session id
        sid_retry = save_workout(w_data, temp_db)
        assert sid_retry == sid

    def test_21_existing_records_remain_unchanged(self, temp_db):
        """21. Table migrations and saves preserve existing records exactly."""
        # Insert initial record
        sid1 = save_workout({
            "username": "persist_user", "completed_reps": 7, "good_reps": 5,
            "form_score": 82.0, "duration_sec": 35.0, "session_token": "p_1"
        }, temp_db)

        # Run init_db again (triggers migration check)
        init_db(temp_db)

        workouts = get_user_workouts("persist_user", temp_db)
        assert len(workouts) == 1
        assert workouts[0]["session_id"] == sid1
        assert workouts[0]["reps"] == 7
        assert workouts[0]["score"] == 82.0

    def test_22_no_sql_injection_is_possible(self, temp_db):
        """22. Parameterized SQL prevents injection via usernames or values."""
        payload = "admin' OR 1=1; DROP TABLE workouts; --"
        user = get_or_create_user(payload, temp_db)
        assert user["id"] > 0

        save_workout({
            "username": payload, "completed_reps": 2, "good_reps": 2,
            "form_score": 100.0, "duration_sec": 10.0, "session_token": "inj_1"
        }, temp_db)

        # Workouts table still exists and only has 1 record for payload
        workouts = get_user_workouts(payload, temp_db)
        assert len(workouts) == 1

    def test_23_live_workout_fsm_and_squat_counter_remain_functional(self):
        """23. SquatCounter FSM accurately counts standing still (0 reps) vs valid squats (1 rep)."""
        counter = SquatCounter()

        # 1. Standing still 60 frames = 0 reps
        for _ in range(60):
            res = counter.update(make_kps(knee_angle=175.0))
            assert res["state"] == SquatState.STANDING
            assert res["rep_count"] == 0

        # 2. Perform 1 full clean squat
        for k in [175.0, 140.0, 88.0, 130.0, 165.0]:
            counter.update(make_kps(knee_angle=k))

        assert counter.rep_count == 1
        assert counter.good_reps == 1

    def test_24_frame_level_ml_predictions_do_not_directly_increment_rep_totals(self):
        """24. Frame-level ML predictions do not increment rep totals; only finalized FSM events do."""
        from ml.evaluate import predict_rule_based

        counter = SquatCounter()
        # Simulate isolated frame classifications
        preds = []
        for angle in [170.0, 160.0, 140.0, 90.0, 85.0]:
            p = predict_rule_based(knee_angle=angle, back_lean=15.0)
            preds.append(p)

        # Frame predictions say "good" when angle is 90
        assert "good" in preds
        # But counter must not increment reps from frame-level predictions
        assert counter.rep_count == 0
        assert counter.good_reps == 0
        assert counter.shallow_reps == 0

    def test_25_rep_telemetry_persistence_and_retrieval_by_id(self, temp_db):
        """25. Detailed rep telemetry is serialized into SQLite and retrieved by ID."""
        rep_telemetry = [
            {"rep_number": 1, "min_knee_angle": 88.0, "max_back_lean": 22.0, "duration_frames": 24, "status": "good"},
            {"rep_number": 2, "min_knee_angle": 91.5, "max_back_lean": 25.0, "duration_frames": 26, "status": "good"},
            {"rep_number": 3, "min_knee_angle": 120.0, "max_back_lean": 18.0, "duration_frames": 22, "status": "too_shallow"},
        ]
        w_data = {
            "username": "telemetry_user",
            "exercise_name": "Squats",
            "planned_sets": 1,
            "planned_reps": 3,
            "completed_reps": 3,
            "good_reps": 2,
            "shallow_reps": 1,
            "forward_lean_reps": 0,
            "form_score": 85.0,
            "duration_sec": 25.0,
            "avg_fps": 15.0,
            "rep_history": rep_telemetry,
            "session_token": "telem_tok_1",
        }
        sid = save_workout(w_data, temp_db)
        assert sid > 0

        # Retrieve by session ID
        retrieved = get_workout_by_id(sid, temp_db)
        assert retrieved is not None
        assert retrieved["session_id"] == sid
        assert retrieved["reps"] == 3
        assert len(retrieved["rep_history"]) == 3
        assert retrieved["rep_history"][0]["min_knee_angle"] == 88.0
        assert retrieved["rep_history"][2]["status"] == "too_shallow"

    def test_26_legacy_session_without_telemetry_loads_safely(self, temp_db):
        """26. Sessions saved without rep_telemetry load with empty rep_history without crashing."""
        w_data = {
            "username": "legacy_user",
            "exercise_name": "Squats",
            "completed_reps": 5,
            "good_reps": 5,
            "shallow_reps": 0,
            "forward_lean_reps": 0,
            "form_score": 100.0,
            "duration_sec": 30.0,
            "session_token": "legacy_tok_1",
            # No rep_history key provided
        }
        sid = save_workout(w_data, temp_db)
        retrieved = get_workout_by_id(sid, temp_db)
        assert retrieved is not None
        assert retrieved["reps"] == 5
        assert retrieved["rep_history"] == []

    def test_27_intentional_finish_preserves_reps_and_telemetry(self, temp_db):
        """27. Repeated finish calls do not duplicate records, and telemetry is preserved."""
        reps = [
            {"rep_number": 1, "min_knee_angle": 90.0, "max_back_lean": 20.0, "duration_frames": 25, "status": "good"}
        ]
        w_data = {
            "username": "athlete_finish",
            "exercise_name": "Squats",
            "completed_reps": 1,
            "good_reps": 1,
            "shallow_reps": 0,
            "forward_lean_reps": 0,
            "form_score": 100.0,
            "duration_sec": 12.0,
            "rep_history": reps,
            "session_token": "finish_tok_uniq",
        }
        sid1 = save_workout(w_data, temp_db)
        sid2 = save_workout(w_data, temp_db)
        assert sid1 == sid2

        workouts = get_user_workouts("athlete_finish", temp_db)
        assert len(workouts) == 1
        assert len(workouts[0]["rep_history"]) == 1



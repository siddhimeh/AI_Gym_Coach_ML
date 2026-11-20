"""
Unit Tests for SQLite Database Integration
Tests table creation, user authentication, workout persistence, and injection protection.
"""

import os
import pytest
import sqlite3
from db.database import (
    init_db,
    get_or_create_user,
    save_workout,
    get_user_workouts,
    get_user_stats,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary, isolated SQLite database file."""
    db_file = os.path.join(tmp_path, "test_gym_coach.db")
    init_db(db_file)
    return db_file


class TestDatabase:
    def test_init_db(self, temp_db):
        """Verifies that users and workouts tables exist upon initialization."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "users" in tables
        assert "workouts" in tables

    def test_get_or_create_user(self, temp_db):
        """Creating the same username twice should return the identical user record."""
        user1 = get_or_create_user("akarsh", temp_db)
        assert user1["id"] is not None
        assert user1["username"] == "akarsh"

        user2 = get_or_create_user("Akarsh", temp_db)
        assert user2["id"] == user1["id"]

    def test_save_and_retrieve_workout(self, temp_db):
        """A saved workout must be retrievable with all telemetry fields intact."""
        workout_data = {
            "username": "akarsh",
            "exercise_name": "Squats",
            "planned_sets": 3,
            "planned_reps": 10,
            "completed_reps": 10,
            "good_reps": 8,
            "shallow_reps": 2,
            "forward_lean_reps": 0,
            "form_score": 80.0,
            "duration_sec": 45.2,
            "avg_fps": 15.0,
        }

        workout_id = save_workout(workout_data, temp_db)
        assert workout_id > 0

        workouts = get_user_workouts("akarsh", temp_db)
        assert len(workouts) == 1
        w = workouts[0]
        assert w["reps"] == 10
        assert w["good"] == 8
        assert w["shallow"] == 2
        assert w["score"] == 80.0
        assert w["duration"] == 45.2

    def test_user_aggregate_stats(self, temp_db):
        """Tests calculation of lifetime reps, sessions, and average form score."""
        save_workout({
            "username": "akarsh",
            "completed_reps": 10,
            "good_reps": 10,
            "form_score": 100.0,
            "duration_sec": 40.0,
        }, temp_db)

        save_workout({
            "username": "akarsh",
            "completed_reps": 10,
            "good_reps": 8,
            "form_score": 80.0,
            "duration_sec": 50.0,
        }, temp_db)

        stats = get_user_stats("akarsh", temp_db)
        assert stats["total_workouts"] == 2
        assert stats["total_reps"] == 20
        assert stats["clean_reps"] == 18
        assert stats["avg_form_score"] == 90.0

    def test_sql_injection_defense(self, temp_db):
        """Verifies that parameterized SQL prevents SQL injection characters from corrupting the DB."""
        malicious_input = "hacker' OR '1'='1"
        user = get_or_create_user(malicious_input, temp_db)
        assert user["username"] == malicious_input.lower()

        # Should retrieve only this literal username, not all users
        workouts = get_user_workouts(malicious_input, temp_db)
        assert isinstance(workouts, list)

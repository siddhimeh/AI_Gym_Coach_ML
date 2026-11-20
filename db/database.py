"""
SQLite Database Management Module
Implements user tracking, workout session persistence, schema migration,
timezone-aware timestamp formatting, and progress history queries.
All queries use parameterized inputs to prevent SQL injection vulnerabilities.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
import zoneinfo

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(ROOT_DIR, "db", "gym_coach.db")
DISPLAY_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection and sets row_factory to sqlite3.Row."""
    abs_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    conn = sqlite3.connect(abs_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable foreign key support in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def format_timestamp_for_display(ts_str: str | None) -> tuple[str, str, str]:
    """
    Parses a UTC or legacy timestamp and converts to Asia/Kolkata local time.
    Returns: (date_str 'YYYY-MM-DD', time_str 'HH:MM:SS', full_display 'YYYY-MM-DD HH:MM:SS')
    """
    if not ts_str:
        now_local = datetime.now(DISPLAY_TZ)
        return (
            now_local.strftime("%Y-%m-%d"),
            now_local.strftime("%H:%M:%S"),
            now_local.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # Clean raw string
    raw = str(ts_str).strip()
    dt = None

    # Handle standard ISO and SQLite current_timestamp formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            clean_ts = raw.split("+")[0].rstrip("Z")
            dt = datetime.strptime(clean_ts, fmt)
            break
        except ValueError:
            pass

    if dt is None:
        # Fallback for legacy date-only strings (e.g. '2026-10-31')
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d"), "00:00:00", dt.strftime("%Y-%m-%d 00:00:00")
        except ValueError:
            return raw, "--", raw

    # Assume stored database timestamp is UTC
    dt_utc = dt.replace(tzinfo=timezone.utc)
    dt_local = dt_utc.astimezone(DISPLAY_TZ)
    return (
        dt_local.strftime("%Y-%m-%d"),
        dt_local.strftime("%H:%M:%S"),
        dt_local.strftime("%Y-%m-%d %H:%M:%S"),
    )


def init_db(db_path: str = DEFAULT_DB_PATH):
    """
    Initializes the database schema if tables do not exist,
    and performs safe backward-compatible migration for workouts table.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Check existing workouts table schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workouts';")
        has_workouts = cursor.fetchone() is not None

        if not has_workouts:
            cursor.execute("""
                CREATE TABLE workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    exercise_name TEXT NOT NULL,
                    planned_sets INTEGER NOT NULL,
                    planned_reps INTEGER NOT NULL,
                    completed_reps INTEGER NOT NULL,
                    good_reps INTEGER NOT NULL,
                    shallow_reps INTEGER NOT NULL,
                    forward_lean_reps INTEGER NOT NULL,
                    form_score REAL,
                    duration_sec REAL NOT NULL,
                    avg_fps REAL DEFAULT 0.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_token TEXT UNIQUE,
                    rep_telemetry TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
        else:
            # Check if migration is needed (form_score nullable and session_token column present)
            cursor.execute("PRAGMA table_info(workouts);")
            columns = cursor.fetchall()
            col_map = {c["name"]: c for c in columns}

            score_col = col_map.get("form_score")
            token_col = col_map.get("session_token")

            needs_migration = False
            if score_col and score_col["notnull"] == 1:
                needs_migration = True
            if not token_col:
                needs_migration = True

            if needs_migration:
                # Perform safe table migration preserving all existing rows
                cursor.execute("PRAGMA foreign_keys = OFF;")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workouts_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        exercise_name TEXT NOT NULL,
                        planned_sets INTEGER NOT NULL,
                        planned_reps INTEGER NOT NULL,
                        completed_reps INTEGER NOT NULL,
                        good_reps INTEGER NOT NULL,
                        shallow_reps INTEGER NOT NULL,
                        forward_lean_reps INTEGER NOT NULL,
                        form_score REAL,
                        duration_sec REAL NOT NULL,
                        avg_fps REAL DEFAULT 0.0,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_token TEXT UNIQUE,
                        rep_telemetry TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("""
                    INSERT INTO workouts_v2 (
                        id, user_id, exercise_name, planned_sets, planned_reps,
                        completed_reps, good_reps, shallow_reps, forward_lean_reps,
                        form_score, duration_sec, avg_fps, timestamp
                    )
                    SELECT id, user_id, exercise_name, planned_sets, planned_reps,
                           completed_reps, good_reps, shallow_reps, forward_lean_reps,
                           form_score, duration_sec, avg_fps, timestamp
                    FROM workouts;
                """)
                cursor.execute("DROP TABLE workouts;")
                cursor.execute("ALTER TABLE workouts_v2 RENAME TO workouts;")
                cursor.execute("PRAGMA foreign_keys = ON;")

        # Ensure rep_telemetry column exists in workouts
        cursor.execute("PRAGMA table_info(workouts);")
        col_names = [c["name"] for c in cursor.fetchall()]
        if "rep_telemetry" not in col_names:
            cursor.execute("ALTER TABLE workouts ADD COLUMN rep_telemetry TEXT;")

        # Ensure index on session_token
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_workouts_session_token ON workouts(session_token);")
        conn.commit()


def get_or_create_user(username: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Finds an existing user or creates a new user record.
    Uses parameterized queries for classroom security.
    """
    username = (username or "").strip().lower()
    if not username:
        username = "guest"
    init_db(db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Query existing user
        cursor.execute("SELECT id, username, created_at FROM users WHERE username = ?;", (username,))
        row = cursor.fetchone()

        if row:
            return dict(row)

        # Insert new user
        cursor.execute("INSERT INTO users (username) VALUES (?);", (username,))
        conn.commit()
        user_id = cursor.lastrowid

        return {
            "id": user_id,
            "username": username,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def save_workout(workout_data: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Persists a completed workout session linked to a user.
    Idempotent: if session_token is provided and already exists, returns existing session id.
    
    Expected workout_data keys:
        username, exercise_name, planned_sets, planned_reps,
        completed_reps, good_reps, shallow_reps, forward_lean_reps,
        form_score, duration_sec, avg_fps, session_token, timestamp
    """
    init_db(db_path)
    user = get_or_create_user(workout_data["username"], db_path)
    session_token = workout_data.get("session_token")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Idempotency check: prevent duplicate writes
        if session_token:
            cursor.execute("SELECT id FROM workouts WHERE session_token = ?;", (str(session_token),))
            existing = cursor.fetchone()
            if existing:
                return existing[0]

        completed_reps = int(workout_data.get("completed_reps", 0))
        form_score_raw = workout_data.get("form_score")

        # Zero-rep workouts have undefined/NULL form score in SQLite
        if completed_reps == 0 or form_score_raw is None:
            form_score = None
        else:
            form_score = float(form_score_raw)

        utc_ts = workout_data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        rep_telemetry = None
        if workout_data.get("rep_history"):
            try:
                rep_telemetry = json.dumps(workout_data["rep_history"])
            except Exception as e:
                logger.warning(f"Could not serialize rep_history: {e}")

        cursor.execute("""
            INSERT INTO workouts (
                user_id, exercise_name, planned_sets, planned_reps,
                completed_reps, good_reps, shallow_reps, forward_lean_reps,
                form_score, duration_sec, avg_fps, timestamp, session_token,
                rep_telemetry
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            user["id"],
            workout_data.get("exercise_name", "Squats"),
            int(workout_data.get("planned_sets", 3)),
            int(workout_data.get("planned_reps", 10)),
            completed_reps,
            int(workout_data.get("good_reps", 0)),
            int(workout_data.get("shallow_reps", 0)),
            int(workout_data.get("forward_lean_reps", 0)),
            form_score,
            float(workout_data.get("duration_sec", 0.0)),
            float(workout_data.get("avg_fps", 0.0)),
            utc_ts,
            str(session_token) if session_token else None,
            rep_telemetry,
        ))
        conn.commit()
        return cursor.lastrowid


def get_user_workouts(username: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Retrieves all past workout sessions for a given user, ordered newest first."""
    init_db(db_path)
    user = get_or_create_user(username, db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                id AS session_id,
                exercise_name,
                planned_sets,
                planned_reps,
                completed_reps AS reps,
                completed_reps,
                good_reps AS good,
                good_reps,
                shallow_reps AS shallow,
                shallow_reps,
                forward_lean_reps AS forward_lean,
                forward_lean_reps,
                form_score AS score,
                form_score,
                duration_sec AS duration,
                duration_sec,
                avg_fps,
                timestamp,
                session_token,
                rep_telemetry
            FROM workouts
            WHERE user_id = ?
            ORDER BY timestamp DESC, id DESC;
        """, (user["id"],))
        rows = cursor.fetchall()
        
        workouts = []
        for r in rows:
            w = dict(r)
            date_str, time_str, full_str = format_timestamp_for_display(w.get("timestamp"))
            w["date"] = date_str
            w["time"] = time_str
            w["display_time"] = full_str
            if w.get("rep_telemetry"):
                try:
                    w["rep_history"] = json.loads(w["rep_telemetry"])
                except Exception:
                    w["rep_history"] = []
            else:
                w["rep_history"] = []
            workouts.append(w)

        return workouts


def get_workout_by_id(session_id: int, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Retrieves a single workout session by its ID, with deserialized rep telemetry."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                w.id AS session_id,
                u.username,
                w.exercise_name,
                w.planned_sets,
                w.planned_reps,
                w.completed_reps AS reps,
                w.completed_reps,
                w.good_reps AS good,
                w.good_reps,
                w.shallow_reps AS shallow,
                w.shallow_reps,
                w.forward_lean_reps AS forward_lean,
                w.forward_lean_reps,
                w.form_score AS score,
                w.form_score,
                w.duration_sec AS duration,
                w.duration_sec,
                w.avg_fps,
                w.timestamp,
                w.session_token,
                w.rep_telemetry
            FROM workouts w
            JOIN users u ON w.user_id = u.id
            WHERE w.id = ?;
        """, (int(session_id),))
        row = cursor.fetchone()
        if not row:
            return None
        w = dict(row)
        date_str, time_str, full_str = format_timestamp_for_display(w.get("timestamp"))
        w["date"] = date_str
        w["time"] = time_str
        w["display_time"] = full_str
        if w.get("rep_telemetry"):
            try:
                w["rep_history"] = json.loads(w["rep_telemetry"])
            except Exception:
                w["rep_history"] = []
        else:
            w["rep_history"] = []
        return w


def get_user_stats(username: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Computes aggregate workout statistics for a user, handling zero-rep sessions properly."""
    workouts = get_user_workouts(username, db_path)
    if not workouts:
        return {
            "total_workouts": 0,
            "total_reps": 0,
            "clean_reps": 0,
            "clean_ratio": 0.0,
            "avg_form_score": 0.0,
            "total_duration_sec": 0.0,
        }

    total_reps = sum(w["reps"] for w in workouts)
    clean_reps = sum(w["good"] for w in workouts)
    
    # Exclude sessions where score is undefined / None
    valid_scores = [w["score"] for w in workouts if w["score"] is not None]
    avg_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
    
    total_dur = sum(w["duration"] for w in workouts)
    clean_ratio = (clean_reps / total_reps * 100.0) if total_reps > 0 else 0.0

    return {
        "total_workouts": len(workouts),
        "total_reps": total_reps,
        "clean_reps": clean_reps,
        "clean_ratio": round(clean_ratio, 1),
        "avg_form_score": round(avg_score, 1),
        "total_duration_sec": round(total_dur, 1),
    }


def clear_user_workouts(username: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """Deletes all workout records for a user. Useful for test resets."""
    init_db(db_path)
    user = get_or_create_user(username, db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM workouts WHERE user_id = ?;", (user["id"],))
        conn.commit()
        return cursor.rowcount


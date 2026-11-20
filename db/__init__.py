"""
Database package for AI Gym Coach.
Handles SQLite connection, schema initialization, and parameterized CRUD operations.
"""
from .database import (
    init_db,
    get_or_create_user,
    save_workout,
    get_user_workouts,
    get_user_stats,
)

__all__ = [
    "init_db",
    "get_or_create_user",
    "save_workout",
    "get_user_workouts",
    "get_user_stats",
]

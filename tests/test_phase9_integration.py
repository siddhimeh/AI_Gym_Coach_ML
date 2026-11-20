"""
Phase 9 Integration Test
Verifies end-to-end execution of the live camera pipeline components:
1. Video reading & pose detection
2. Angle calculations (knee, hip, back lean)
3. SquatCounter state machine transitions
4. HUD drawing and banner rendering
5. SQLite session storage and retrieval
"""

import os
import cv2
import pytest
from pose.detector import PoseDetector
from exercises.squat import SquatCounter
from utils.video import draw_hud_panel, draw_feedback_banner
from utils.metrics import calculate_workout_summary
from db.database import init_db, save_workout, get_user_workouts

def test_phase9_live_pipeline_components(tmp_path):
    clip_path = "data/clips/squat_sample.webm"
    assert os.path.exists(clip_path), f"Sample clip {clip_path} must exist"

    cap = cv2.VideoCapture(clip_path)
    assert cap.isOpened(), "VideoCapture must open sample clip"

    detector = PoseDetector()
    counter = SquatCounter()
    test_db = os.path.join(tmp_path, "test_phase9.db")
    init_db(test_db)

    frame_count = 0
    # Process 50 frames
    while cap.isOpened() and frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result = detector.find_pose(frame)
        kp_data = detector.extract_landmarks(result, frame.shape)
        annotated = detector.draw_skeleton(frame, kp_data)
        state_data = counter.update(kp_data)

        # Test HUD rendering without errors
        annotated = draw_hud_panel(
            frame=annotated,
            rep_count=state_data["rep_count"],
            state=state_data["state"],
            knee_angle=state_data["knee_angle"],
            back_lean=state_data["back_lean"],
            fps=30.0,
            good_reps=state_data["good_reps"],
        )

        annotated = draw_feedback_banner(
            frame=annotated,
            feedback_text=state_data["feedback"],
            status=state_data["last_rep_status"],
        )

        assert annotated.shape == frame.shape
        assert annotated.dtype == frame.dtype

    cap.release()
    detector.close()

    assert frame_count == 50

    # Test summary and database persistence
    summary = calculate_workout_summary(
        total_reps=counter.rep_count,
        good_reps=counter.good_reps,
        shallow_reps=counter.shallow_reps,
        forward_lean_reps=counter.forward_lean_reps,
        duration_sec=3.5,
        avg_fps=28.0,
    )

    workout_rec = {
        "username": "phase9_tester",
        "exercise_name": "Squats",
        "planned_sets": 1,
        "planned_reps": 10,
        "completed_reps": summary["total_reps"],
        "good_reps": summary["good_reps"],
        "shallow_reps": summary["shallow_reps"],
        "forward_lean_reps": summary["forward_lean_reps"],
        "form_score": summary["form_score"],
        "duration_sec": summary["duration_sec"],
        "avg_fps": summary["avg_fps"],
    }
    session_id = save_workout(workout_rec, test_db)
    assert session_id > 0

    records = get_user_workouts("phase9_tester", test_db)
    assert len(records) == 1
    assert records[0]["exercise_name"] == "Squats"
    print("\nPhase 9 live pipeline integration test PASSED!")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_phase9_live_pipeline_components(td)

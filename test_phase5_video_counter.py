"""
Phase 5: Real Video Repetition Counter & Form Evaluator
Tests SquatCounter state machine on real video clips:
1. data/clips/squat_sample.webm (Contains 2 authentic Good Reps)
2. data/clips/squat_shallow_rep.webm (Contains 1 authentic Shallow Rep)
"""

import os
import cv2
from pose.detector import PoseDetector
from exercises.squat import SquatCounter

def evaluate_clip(clip_path: str, expected_reps: int, expected_form: str):
    clip_name = os.path.basename(clip_path)
    print(f"\nEvaluating Clip: {clip_name} (Expected Reps: {expected_reps}, Expected Form: {expected_form})")
    print("-" * 65)

    if not os.path.exists(clip_path):
        print(f"[ERROR] Clip not found: {clip_path}")
        return False

    cap = cv2.VideoCapture(clip_path)
    detector = PoseDetector()
    counter = SquatCounter()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        res = detector.find_pose(frame)
        kp = detector.extract_landmarks(res, frame.shape)
        counter.update(kp)

    cap.release()
    detector.close()

    print(f"Frames Processed: {frame_idx}")
    print(f"Total Reps Counted:    {counter.rep_count} (Expected: {expected_reps})")
    print(f"  - Good Reps:         {counter.good_reps}")
    print(f"  - Shallow Reps:      {counter.shallow_reps}")
    print(f"  - Forward Lean Reps: {counter.forward_lean_reps}")
    print(f"Last Rep Status:        '{counter.last_rep_status}' (Expected: '{expected_form}')")

    success = (counter.rep_count == expected_reps and counter.last_rep_status == expected_form)
    print(f"Verification Result:    {'[PASSED]' if success else '[FAILED]'}")
    return success


def run_phase5_video_eval():
    print("=" * 65)
    print("PHASE 5: SQUAT COUNTER & FORM EVALUATION ON REAL CLIPS")
    print("=" * 65)

    test1 = evaluate_clip("data/clips/squat_sample.webm", expected_reps=2, expected_form="good")
    test2 = evaluate_clip("data/clips/squat_shallow_rep.webm", expected_reps=1, expected_form="too_shallow")

    print("\n" + "=" * 65)
    if test1 and test2:
        print("[ALL TESTS PASSED] Both real clips evaluated with 100% rep and form accuracy!")
    else:
        print("[TEST WARNING] One or more clip evaluations did not match expected form.")
    print("=" * 65)
    return test1 and test2


if __name__ == "__main__":
    run_phase5_video_eval()

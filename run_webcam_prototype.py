"""
Phase 6: Real-time OpenCV Webcam Prototype
Live camera capture, skeleton tracking, rep counter HUD, and performance profiling.
Optimized for 640x480 resolution on low-power Intel Core i3 CPU hardware.

Controls:
  - Press 'q' or ESC: End workout and display summary
  - Press 'r': Reset rep counters
"""

import os
import sys
import time
import argparse
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pose.detector import PoseDetector
from exercises.squat import SquatCounter
from utils.video import draw_hud_panel, draw_feedback_banner
from utils.metrics import calculate_workout_summary


def run_webcam(camera_index: int = 0, max_frames: int | None = None, show_window: bool = True):
    print("=" * 65)
    print("PHASE 6: AI GYM COACH — LIVE WEBCAM PROTOTYPE")
    print("=" * 65)
    print(f"Connecting to camera index {camera_index}...")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not access camera at index {camera_index}.")
        print("Please ensure your webcam is connected and not in use by another app.")
        return False

    # Force 640x480 resolution for low latency on CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SUCCESS] Webcam opened at resolution: {actual_w}x{actual_h}")
    print("Controls: Press 'q' to stop workout, 'r' to reset reps.\n")

    detector = PoseDetector()
    counter = SquatCounter()

    window_name = "AI Gym Coach - Live Workout (Press 'Q' to Exit)"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()
    prev_time = start_time
    fps = 0.0

    last_snapshot_saved = False
    snapshot_path = os.path.join(ROOT_DIR, "data", "processed", "webcam_live_hud.jpg")
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

    try:
        while cap.isOpened():
            curr_time = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam.")
                break

            frame_count += 1

            # Calculate instantaneous FPS
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            prev_time = curr_time

            # 1. Pose detection & Landmark extraction
            result = detector.find_pose(frame)
            keypoint_data = detector.extract_landmarks(result, frame.shape)

            # 2. Draw Skeleton lines and joint markers
            annotated = detector.draw_skeleton(frame, keypoint_data)

            # 3. Update Squat Counter state machine
            state_data = counter.update(keypoint_data)

            # 4. Render sleek HUD Telemetry Panel
            annotated = draw_hud_panel(
                frame=annotated,
                rep_count=state_data["rep_count"],
                state=state_data["state"],
                knee_angle=state_data["knee_angle"],
                back_lean=state_data["back_lean"],
                fps=fps,
                good_reps=state_data["good_reps"],
            )

            # 5. Render Live Coaching Feedback Banner
            annotated = draw_feedback_banner(
                frame=annotated,
                feedback_text=state_data["feedback"],
                status=state_data["last_rep_status"],
            )

            # Save snapshot of live HUD during workout
            if not last_snapshot_saved and frame_count >= 15:
                cv2.imwrite(snapshot_path, annotated)
                last_snapshot_saved = True

            # 6. Display to user window
            if show_window:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    print("\n[USER INPUT] Exit command received.")
                    break
                elif key == ord('r'):
                    counter.reset()
                    print("\n[USER INPUT] Rep counter reset.")

            if max_frames and frame_count >= max_frames:
                print(f"\n[INFO] Reached diagnostic test target ({max_frames} frames).")
                break

    finally:
        total_duration = time.time() - start_time
        avg_fps = frame_count / total_duration if total_duration > 0 else 0.0

        cap.release()
        detector.close()
        if show_window:
            cv2.destroyAllWindows()

        # Workout session report
        summary = calculate_workout_summary(
            total_reps=counter.rep_count,
            good_reps=counter.good_reps,
            shallow_reps=counter.shallow_reps,
            forward_lean_reps=counter.forward_lean_reps,
            duration_sec=total_duration,
            avg_fps=avg_fps,
        )

        print("-" * 65)
        print("WORKOUT SESSION SUMMARY:")
        print(f"  - Total Duration:       {summary['duration_sec']} seconds")
        print(f"  - Frames Captured:      {frame_count}")
        print(f"  - Average Performance:  {summary['avg_fps']} FPS")
        print(f"  - Total Reps:           {summary['total_reps']}")
        print(f"  - Clean Form Reps:      {summary['good_reps']}")
        print(f"  - Shallow Reps:         {summary['shallow_reps']}")
        print(f"  - Forward Lean Reps:    {summary['forward_lean_reps']}")
        print(f"  - Form Quality Score:   {summary['form_score']}%")
        if last_snapshot_saved:
            print(f"  - Live HUD snapshot:    {snapshot_path}")
        print("=" * 65)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gym Coach Live Webcam Prototype")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (diagnostic)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    run_webcam(
        camera_index=args.camera,
        max_frames=args.max_frames,
        show_window=not args.no_display,
    )

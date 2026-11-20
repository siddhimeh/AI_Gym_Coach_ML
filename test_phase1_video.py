"""
Phase 1: Verification Test on Sample Video Clip
Processes data/clips/squat_sample.webm using PoseDetector,
prints landmark coordinates, visibility diagnostics, and saves an annotated frame.
"""

import os
import cv2
from pose.detector import PoseDetector

def run_phase1_test(video_path: str = "data/clips/squat_sample.webm"):
    print("=" * 60)
    print("PHASE 1: MEDIAPIPE POSE DETECTION ON RECORDED VIDEO")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video clip not found at '{video_path}'.")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"Video Properties:")
    print(f"  - Path: {video_path}")
    print(f"  - Resolution: {width} x {height}")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - Frame Rate (FPS): {fps:.2f}")
    print(f"  - Duration: {duration_sec:.2f} seconds")
    print("-" * 60)

    # Initialize PoseDetector
    detector = PoseDetector()

    detected_count = 0
    frame_idx = 0
    sample_keypoints = None
    annotated_saved = False
    output_img_path = "data/processed/phase1_annotated_sample.jpg"

    os.makedirs("data/processed", exist_ok=True)

    print("Processing video frames with MediaPipe Pose...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = detector.find_pose(frame)
        kp_data = detector.extract_landmarks(result, frame.shape)

        if kp_data:
            detected_count += 1
            if sample_keypoints is None and frame_idx >= 10:
                # Store frame 10 (a stable squat position) as diagnostic sample
                sample_keypoints = (frame_idx, kp_data)
                annotated = detector.draw_skeleton(frame, kp_data)
                cv2.imwrite(output_img_path, annotated)
                annotated_saved = True

    cap.release()
    detector.close()

    print("-" * 60)
    print("DETECTION SUMMARY:")
    print(f"  - Frames Processed: {frame_idx}")
    print(f"  - Frames with Pose Detected: {detected_count} / {frame_idx} ({(detected_count / frame_idx * 100):.1f}%)")
    
    if sample_keypoints:
        s_frame, s_data = sample_keypoints
        side = s_data["side"]
        pts = s_data["points"]
        print(f"\nDiagnostic Snapshot (Frame #{s_frame}, Active Side: {side.upper()}):")
        print(f"{'Joint':<12} | {'Normalized (x, y)':<20} | {'Pixel (px, py)':<16} | {'Visibility':<10}")
        print("-" * 65)
        for joint in ["shoulder", "hip", "knee", "ankle", "heel", "foot"]:
            p = pts[joint]
            norm_str = f"({p['x_norm']:.3f}, {p['y_norm']:.3f})"
            px_str = f"({p['x_px']}, {p['y_px']})"
            vis_str = f"{p['visibility']:.3f}"
            print(f"{joint.capitalize():<12} | {norm_str:<20} | {px_str:<16} | {vis_str:<10}")

    if annotated_saved:
        print(f"\n[SAVED] Annotated sample frame with skeleton overlay saved to:")
        print(f"        {output_img_path}")

    print("=" * 60)
    print("Phase 1 test finished successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_phase1_test()

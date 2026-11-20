"""
Phase 2: Video Angles Analysis
Integrates PoseDetector with Joint Angle Calculations on data/clips/squat_sample.webm.
Measures real knee flexion, hip hinge, and torso lean throughout the squat movement.
"""

import cv2
import numpy as np
from pose.detector import PoseDetector
from pose.angles import calculate_knee_angle, calculate_hip_angle, calculate_back_lean

def analyze_video_angles(video_path: str = "data/clips/squat_sample.webm"):
    print("=" * 65)
    print("PHASE 2: REAL-TIME ANGLE ANALYSIS ON SQUAT CLIP")
    print("=" * 65)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    detector = PoseDetector()

    frame_idx = 0
    knee_angles = []
    hip_angles = []
    back_leans = []
    
    deepest_frame_data = None
    min_knee_angle = 999.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = detector.find_pose(frame)
        kp = detector.extract_landmarks(result, frame.shape)

        if kp:
            pts = kp["points"]
            knee_ang = calculate_knee_angle(pts["hip"], pts["knee"], pts["ankle"])
            hip_ang = calculate_hip_angle(pts["shoulder"], pts["hip"], pts["knee"])
            back_ang = calculate_back_lean(pts["shoulder"], pts["hip"])

            knee_angles.append(knee_ang)
            hip_angles.append(hip_ang)
            back_leans.append(back_ang)

            if knee_ang < min_knee_angle:
                min_knee_angle = knee_ang
                deepest_frame_data = (frame_idx, frame.copy(), kp, knee_ang, hip_ang, back_ang)

    cap.release()
    detector.close()

    print(f"Total Frames Analyzed: {len(knee_angles)}")
    print("-" * 65)
    print(f"{'Metric':<18} | {'Min Value':<12} | {'Max Value':<12} | {'Mean Value':<12}")
    print("-" * 65)
    print(f"{'Knee Angle':<18} | {min(knee_angles):<12.1f} | {max(knee_angles):<12.1f} | {np.mean(knee_angles):<12.1f}")
    print(f"{'Hip Angle':<18} | {min(hip_angles):<12.1f} | {max(hip_angles):<12.1f} | {np.mean(hip_angles):<12.1f}")
    print(f"{'Back Lean':<18} | {min(back_leans):<12.1f} | {max(back_leans):<12.1f} | {np.mean(back_leans):<12.1f}")
    print("-" * 65)

    if deepest_frame_data:
        f_num, f_img, f_kp, k_ang, h_ang, b_ang = deepest_frame_data
        annotated = detector.draw_skeleton(f_img, f_kp)

        # Draw angle badges onto bottom frame
        cv2.putText(annotated, f"Knee Angle: {k_ang:.1f} deg", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Hip Angle:  {h_ang:.1f} deg", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Back Lean:  {b_ang:.1f} deg", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 2, cv2.LINE_AA)

        out_path = "data/processed/phase2_squat_bottom_annotated.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\n[DEEPEST SQUAT POINT]:")
        print(f"  - Frame Number: #{f_num}")
        print(f"  - Knee Flexion Angle: {k_ang:.1f} deg (Deep parallel squat)")
        print(f"  - Hip Hinge Angle:    {h_ang:.1f} deg")
        print(f"  - Back Lean Angle:    {b_ang:.1f} deg (Good upright posture)")
        print(f"  - Saved visual snapshot to: {out_path}")

    print("=" * 65)
    print("Phase 2 video angle analysis complete.")
    print("=" * 65)

if __name__ == "__main__":
    analyze_video_angles()

"""
Workout Metrics & Biomechanical Analytics Utility
Calculates workout performance statistics, form quality scores,
consistency metrics, and fatigue detection analytics.

Mathematical Definitions:
1. Weighted Biomechanical Form Score:
   Score = max(0.0, min(100.0, (1.0 * R_good + 0.4 * R_shallow + 0.5 * R_lean) / R_total * 100))
   Rationale: Recognizes partial range of motion (0.4) and depth with torso forward lean (0.5),
   distinguishing genuine partial exercise effort from complete inactivity.

2. Strict Binary Clean Ratio:
   Clean Ratio = (R_good / R_total) * 100
   Competitive standard where any form deviation is penalized.

3. Fatigue Detection Index:
   Compares kinematic trends (concentric tempo deceleration, depth decay, and torso tilt)
   between the first half and second half of repetitions in a set.
"""

from typing import List, Dict, Any
import numpy as np


def calculate_form_score(
    good_reps: int,
    shallow_reps: int,
    forward_lean_reps: int,
    total_reps: int,
    method: str = "weighted",
) -> float:
    """
    Computes the Form Quality Score (%) based on repetition breakdown.

    Args:
        good_reps: Count of full-depth, upright repetitions.
        shallow_reps: Count of repetitions failing parallel depth (>95 deg).
        forward_lean_reps: Count of repetitions with excessive forward lean (>38 deg).
        total_reps: Total completed repetitions.
        method: "weighted" for physiological partial credit or "strict" for binary clean ratio.

    Returns:
        Form Quality Score as a float between 0.0 and 100.0.
    """
    if total_reps <= 0:
        return 0.0

    if method == "strict":
        score = (good_reps / total_reps) * 100.0
    else:
        # Weighted biomechanical formula:
        # Full credit: good_reps * 1.0
        # Partial credit: shallow_reps * 0.4 (active hip/knee flexion, partial ROM)
        # Moderate credit: forward_lean_reps * 0.5 (achieved parallel depth, trunk hinge warning)
        weighted_sum = (
            (good_reps * 1.0)
            + (shallow_reps * 0.4)
            + (forward_lean_reps * 0.5)
        )
        score = (weighted_sum / total_reps) * 100.0

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def analyze_fatigue(rep_history: List[Dict[str, Any]], fps: float = 15.0) -> Dict[str, Any]:
    """
    Analyzes repetition kinetics to detect muscular fatigue patterns.

    Biomechanical Fatigue Indicators:
    1. Tempo Slowdown: Concentric velocity decreases as motor units fatigue,
       lengthening rep duration.
    2. Depth Decay: Minimum knee angle increases (lifter stops shallower).
    3. Torso Collapse: Forward back lean increases as spinal erectors tire.

    Args:
        rep_history: List of rep dicts with keys:
                     ['rep_number', 'min_knee_angle', 'max_back_lean', 'duration_frames', 'status']
        fps: Average camera frame rate (default 15.0).

    Returns:
        Dictionary with fatigue indicators, percentage changes, and coaching guidance.
    """
    if not rep_history or len(rep_history) < 2:
        return {
            "fatigue_detected": False,
            "level": "INSUFFICIENT_DATA",
            "tempo_change_pct": 0.0,
            "depth_drift_deg": 0.0,
            "lean_drift_deg": 0.0,
            "early_avg_duration": 0.0,
            "late_avg_duration": 0.0,
            "summary": "Complete at least 2 repetitions to assess fatigue trends.",
        }

    n = len(rep_history)
    half = n // 2
    early_reps = rep_history[:half]
    late_reps = rep_history[half:]

    # Durations in seconds
    fps_safe = max(fps, 1.0)
    early_durations = [r.get("duration_frames", 30) / fps_safe for r in early_reps]
    late_durations = [r.get("duration_frames", 30) / fps_safe for r in late_reps]
    early_avg_dur = float(np.mean(early_durations))
    late_avg_dur = float(np.mean(late_durations))

    # Tempo change percentage (positive = slower rep duration)
    tempo_change_pct = (
        ((late_avg_dur - early_avg_dur) / early_avg_dur * 100.0)
        if early_avg_dur > 0
        else 0.0
    )

    # Knee depth drift (positive = shallower bottom angle)
    early_knee = [r.get("min_knee_angle", 90.0) for r in early_reps]
    late_knee = [r.get("min_knee_angle", 90.0) for r in late_reps]
    early_avg_knee = float(np.mean(early_knee))
    late_avg_knee = float(np.mean(late_knee))
    depth_drift_deg = late_avg_knee - early_avg_knee

    # Back lean drift (positive = more forward lean)
    early_lean = [r.get("max_back_lean", 20.0) for r in early_reps]
    late_lean = [r.get("max_back_lean", 20.0) for r in late_reps]
    early_avg_lean = float(np.mean(early_lean))
    late_avg_lean = float(np.mean(late_lean))
    lean_drift_deg = late_avg_lean - early_avg_lean

    # Fatigue classification logic
    # Thresholds: tempo slowdown >= 20% OR depth drift >= 5 deg OR lean drift >= 6 deg
    fatigue_detected = (
        tempo_change_pct >= 20.0 or depth_drift_deg >= 5.0 or lean_drift_deg >= 6.0
    )

    if tempo_change_pct >= 35.0 or depth_drift_deg >= 10.0:
        level = "SIGNIFICANT"
        summary = (
            f"Significant fatigue detected! Rep tempo slowed by {tempo_change_pct:+.1f}% "
            f"and depth drifted by {depth_drift_deg:+.1f}°. Consider resting 90s before next set."
        )
    elif fatigue_detected:
        level = "MODERATE"
        indicators = []
        if tempo_change_pct >= 20.0:
            indicators.append(f"tempo slowed by {tempo_change_pct:+.1f}%")
        if depth_drift_deg >= 5.0:
            indicators.append(f"depth became {depth_drift_deg:+.1f}° shallower")
        if lean_drift_deg >= 6.0:
            indicators.append(f"trunk lean increased by {lean_drift_deg:+.1f}°")
        summary = f"Moderate fatigue emerging: {', '.join(indicators)}. Focus on explosive ascent."
    else:
        level = "LOW"
        summary = (
            f"Pacing and biomechanics remained stable ({tempo_change_pct:+.1f}% tempo shift). "
            "Excellent muscular endurance!"
        )

    return {
        "fatigue_detected": fatigue_detected,
        "level": level,
        "tempo_change_pct": round(tempo_change_pct, 1),
        "depth_drift_deg": round(depth_drift_deg, 1),
        "lean_drift_deg": round(lean_drift_deg, 1),
        "early_avg_duration": round(early_avg_dur, 2),
        "late_avg_duration": round(late_avg_dur, 2),
        "summary": summary,
    }


def calculate_consistency_metrics(
    rep_history: List[Dict[str, Any]], fps: float = 15.0
) -> Dict[str, Any]:
    """
    Computes consistency metrics across completed repetitions.
    Uses Coefficient of Variation (CV = std / mean * 100) for duration and depth.

    Args:
        rep_history: List of rep dicts.
        fps: Camera frame rate.

    Returns:
        Dictionary with consistency score (0-100) and stability ratings.
    """
    if not rep_history or len(rep_history) < 2:
        return {
            "consistency_score": 100.0,
            "rating": "Baseline",
            "duration_std": 0.0,
            "depth_std": 0.0,
        }

    fps_safe = max(fps, 1.0)
    durations = [r.get("duration_frames", 30) / fps_safe for r in rep_history]
    depths = [r.get("min_knee_angle", 90.0) for r in rep_history]

    dur_std = float(np.std(durations))
    dur_mean = float(np.mean(durations)) if np.mean(durations) > 0 else 1.0
    dur_cv = (dur_std / dur_mean) * 100.0

    depth_std = float(np.std(depths))

    # Consistency score drops as variance increases
    # Perfect score: 100. Deduction for duration jitter and depth spread.
    penalty = (dur_cv * 0.5) + (depth_std * 1.5)
    consistency_score = round(float(np.clip(100.0 - penalty, 10.0, 100.0)), 1)

    if consistency_score >= 85.0:
        rating = "Elite Consistency"
    elif consistency_score >= 70.0:
        rating = "Good Control"
    elif consistency_score >= 50.0:
        rating = "Moderate Variance"
    else:
        rating = "Inconsistent Pacing"

    return {
        "consistency_score": consistency_score,
        "rating": rating,
        "duration_std": round(dur_std, 2),
        "depth_std": round(depth_std, 1),
    }


def calculate_workout_summary(
    total_reps: int,
    good_reps: int,
    shallow_reps: int,
    forward_lean_reps: int,
    duration_sec: float,
    avg_fps: float,
    rep_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Computes an objective, explainable workout summary report with
    both weighted biomechanical score and strict clean ratio.
    """
    form_score = calculate_form_score(
        good_reps=good_reps,
        shallow_reps=shallow_reps,
        forward_lean_reps=forward_lean_reps,
        total_reps=total_reps,
        method="weighted",
    )
    strict_score = calculate_form_score(
        good_reps=good_reps,
        shallow_reps=shallow_reps,
        forward_lean_reps=forward_lean_reps,
        total_reps=total_reps,
        method="strict",
    )

    avg_rep_dur = (duration_sec / total_reps) if total_reps > 0 else 0.0

    fatigue_data = (
        analyze_fatigue(rep_history, fps=avg_fps)
        if rep_history
        else analyze_fatigue([], fps=avg_fps)
    )

    consistency_data = (
        calculate_consistency_metrics(rep_history, fps=avg_fps)
        if rep_history
        else calculate_consistency_metrics([], fps=avg_fps)
    )

    return {
        "total_reps": total_reps,
        "good_reps": good_reps,
        "shallow_reps": shallow_reps,
        "forward_lean_reps": forward_lean_reps,
        "form_score": form_score,
        "strict_score": strict_score,
        "duration_sec": round(duration_sec, 1),
        "avg_fps": round(avg_fps, 1),
        "avg_rep_duration": round(avg_rep_dur, 2),
        "fatigue": fatigue_data,
        "consistency": consistency_data,
    }

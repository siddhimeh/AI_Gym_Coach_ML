"""
Phase 4: Angle Visualization & Distribution Analysis
Generates publication-quality kinematic curves and distribution charts:
1. data/processed/angle_trajectory_plot.png: Angles vs. frames comparing full vs. shallow squats.
2. data/processed/angle_distribution_plot.png: Boxplots and distributions justifying form thresholds.
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_CSV = os.path.join(ROOT_DIR, "data", "processed", "squat_dataset.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


def plot_angle_trajectories(df: pd.DataFrame):
    """Plots Knee Angle and Back Lean vs. Frame for full vs. shallow squats."""
    good_df = df[df["clip"] == "squat_sample_good"].sort_values("frame")
    shallow_df = df[df["clip"] == "squat_shallow_rep"].sort_values("frame")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: Knee Angle Trajectories
    ax1.plot(good_df["frame"], good_df["knee_angle"], label="Good Squat (Full Depth)", color="#00bcd4", linewidth=2.5)
    ax1.plot(shallow_df["frame"], shallow_df["knee_angle"], label="Shallow Squat (Half Rep)", color="#ff9800", linewidth=2.2, linestyle="--")

    # Threshold guidelines
    ax1.axhline(160, color="#4caf50", linestyle=":", linewidth=1.5, label="Standing Threshold (~160 deg)")
    ax1.axhline(95, color="#e91e63", linestyle=":", linewidth=1.5, label="Full Depth Target (<=95 deg)")

    ax1.set_title("Knee Flexion Angle Across Repetition Frames", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Knee Angle (degrees)", fontsize=11)
    ax1.set_xlabel("Video Frame Index", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(40, 190)

    # Annotate minimum inflection points
    min_good = good_df.loc[good_df["knee_angle"].idxmin()]
    min_shallow = shallow_df.loc[shallow_df["knee_angle"].idxmin()]
    ax1.scatter([min_good["frame"]], [min_good["knee_angle"]], color="#00bcd4", s=90, zorder=5)
    ax1.annotate(f"Min: {min_good['knee_angle']:.1f} deg\n(Full Depth)",
                 xy=(min_good["frame"], min_good["knee_angle"]),
                 xytext=(min_good["frame"] - 35, min_good["knee_angle"] + 20),
                 arrowprops=dict(arrowstyle="->", color="#00bcd4", lw=1.5),
                 fontweight="bold")

    ax1.scatter([min_shallow["frame"]], [min_shallow["knee_angle"]], color="#ff9800", s=90, zorder=5)
    ax1.annotate(f"Min: {min_shallow['knee_angle']:.1f} deg\n(Too Shallow)",
                 xy=(min_shallow["frame"], min_shallow["knee_angle"]),
                 xytext=(min_shallow["frame"] - 30, min_shallow["knee_angle"] - 25),
                 arrowprops=dict(arrowstyle="->", color="#ff9800", lw=1.5),
                 fontweight="bold")

    # Subplot 2: Back Lean
    ax2.plot(good_df["frame"], good_df["back_lean"], label="Good Squat (Upright Torso)", color="#4caf50", linewidth=2.2)
    ax2.plot(shallow_df["frame"], shallow_df["back_lean"], label="Shallow Rep (Back Lean)", color="#9c27b0", linewidth=2.0, linestyle="--")

    ax2.axhline(35, color="#f44336", linestyle=":", linewidth=1.5, label="Excessive Forward Lean Warning (>35-40 deg)")

    ax2.set_title("Torso Lean from Vertical Across Frames", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Back Lean (degrees from vertical)", fontsize=11)
    ax2.set_xlabel("Video Frame Index", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_ylim(0, 60)

    out_file = os.path.join(PROCESSED_DIR, "angle_trajectory_plot.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Trajectory chart saved to: {out_file}")


def plot_distributions(df: pd.DataFrame):
    """Plots distributions and boxplots of knee angle and back lean across classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)

    classes = ["standing", "good", "too_shallow", "back_leaning_forward"]
    colors = ["#9e9e9e", "#00bcd4", "#ff9800", "#f44336"]

    # 1. Knee Angle Boxplot
    knee_data = [df[df["label"] == c]["knee_angle"].dropna().values for c in classes]
    bp1 = ax1.boxplot(knee_data, tick_labels=["Standing", "Good", "Shallow", "Forward Lean"],
                      patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(95, color="#e91e63", linestyle="--", label="Parallel Depth Threshold (95 deg)")
    ax1.set_title("Knee Angle Distribution by Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Knee Angle (degrees)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Back Lean Boxplot
    lean_data = [df[df["label"] == c]["back_lean"].dropna().values for c in classes]
    bp2 = ax2.boxplot(lean_data, tick_labels=["Standing", "Good", "Shallow", "Forward Lean"],
                      patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(40, color="#f44336", linestyle="--", label="Excessive Lean Cutoff (40 deg)")
    ax2.set_title("Back Lean Distribution by Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Back Lean (degrees from vertical)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9)

    out_file = os.path.join(PROCESSED_DIR, "angle_distribution_plot.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Distribution chart saved to: {out_file}")


def run_visualizations():
    print("=" * 65)
    print("PHASE 4: KINEMATIC TRAJECTORY & DISTRIBUTION ANALYSIS")
    print("=" * 65)

    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Dataset file not found: {DATA_CSV}")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded dataset: {len(df)} rows.")

    plot_angle_trajectories(df)
    plot_distributions(df)

    print("-" * 65)
    print("THRESHOLD JUSTIFICATION MATRIX (FOR COLLEGE VIVA):")
    print("  1. STANDING / UPRIGHT:")
    print("     - Knee Angle >= 160 deg | Back Lean <= 15 deg")
    print("     - Physical rationale: Normal human standing biomechanics.")
    print("  2. SQUAT INITIATION (DESCENDING):")
    print("     - Knee Angle drops below 150 deg.")
    print("     - Hysteresis: Separates subtle shifting from actual rep start.")
    print("  3. GOOD SQUAT DEPTH (INFLECTION):")
    print("     - Knee Angle <= 95 deg | Back Lean <= 35 deg")
    print("     - Physical rationale: Thigh breaks parallel with floor.")
    print("  4. TOO SHALLOW SQUAT:")
    print("     - Knee Angle turns back UP without dipping below 95-100 deg.")
    print("  5. EXCESSIVE BACK LEAN (FORM WARNING):")
    print("     - Back Lean > 40 deg during descent or bottom inflection.")
    print("=" * 65)
    print("Phase 4 analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_visualizations()

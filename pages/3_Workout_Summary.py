"""
Page 3: Workout Summary & Post-Session Analytics
Displays comprehensive metrics, dual-scoring (weighted vs strict),
repetition kinematics, fatigue detection diagnostics, coaching takeaways,
and SQLite persistence status.
"""

import importlib
import streamlit as st
import matplotlib.pyplot as plt
from utils.metrics import analyze_fatigue, calculate_consistency_metrics
import db.database
importlib.reload(db.database)
from db.database import save_workout, get_user_workouts, get_workout_by_id

st.set_page_config(page_title="Workout Summary - AI Gym Coach", page_icon="📊", layout="wide")

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Please log in on the home page first.")
    st.page_link("app.py", label="Go to Login Screen", icon="🏠")
    st.stop()

user = st.session_state.get("username", "Athlete")

# Retrieve actual finalized workout from session state, with fallback to saved_session_id or SQLite
workout = st.session_state.get("current_workout")

saved_id = st.session_state.get("saved_session_id")
if saved_id:
    db_workout = get_workout_by_id(saved_id)
    if db_workout:
        if workout and workout.get("rep_history") and not db_workout.get("rep_history"):
            db_workout["rep_history"] = workout["rep_history"]
        workout = db_workout
        st.session_state.current_workout = workout

if workout is None:
    workouts = get_user_workouts(user)
    if workouts:
        latest = workouts[0]
        workout = {
            "session_id": latest["session_id"],
            "exercise_name": latest.get("exercise_name", "Squats"),
            "planned_sets": latest.get("planned_sets", 1),
            "planned_reps": latest.get("planned_reps", 10),
            "completed_reps": latest["reps"],
            "good_reps": latest["good"],
            "shallow_reps": latest["shallow"],
            "forward_lean_reps": latest["forward_lean"],
            "form_score": latest["score"],
            "strict_score": (latest["good"] / latest["reps"] * 100.0) if latest["reps"] > 0 else None,
            "duration_sec": latest["duration"],
            "avg_fps": latest.get("avg_fps", 15.0),
            "rep_history": latest.get("rep_history") or [],
            "display_time": latest.get("display_time", "Latest Saved Session"),
        }
        st.session_state.current_workout = workout
    else:
        st.info("ℹ️ No saved workout records found in SQLite database yet. Complete a live workout to view your post-session biomechanical report.")
        workout = {
            "exercise_name": "Squats",
            "planned_sets": 1,
            "planned_reps": 10,
            "completed_reps": 0,
            "good_reps": 0,
            "shallow_reps": 0,
            "forward_lean_reps": 0,
            "form_score": None,
            "strict_score": None,
            "duration_sec": 0.0,
            "avg_fps": 0.0,
            "rep_history": [],
            "display_time": "No Session Recorded",
        }

completed_reps = int(workout.get("completed_reps", 0))
good_reps = int(workout.get("good_reps", 0))
shallow_reps = int(workout.get("shallow_reps", 0))
forward_lean_reps = int(workout.get("forward_lean_reps", 0))
duration = float(workout.get("duration_sec", 0.0))
rep_history = workout.get("rep_history") or []
avg_fps = float(workout.get("avg_fps", 15.0))
exercise_name = workout.get("exercise_name", "Squats")
display_time = workout.get("display_time", "Latest Session")
session_id = workout.get("session_id")

# Fatigue & Consistency evaluation
if completed_reps >= 2:
    fatigue = workout.get("fatigue") or analyze_fatigue(rep_history, fps=avg_fps)
    consistency = workout.get("consistency") or calculate_consistency_metrics(rep_history, fps=avg_fps)
else:
    fatigue = {
        "fatigue_detected": False,
        "level": "INSUFFICIENT_DATA",
        "summary": "At least 2 repetitions required for fatigue trend analysis.",
    }
    consistency = {
        "consistency_score": 100.0,
        "rating": "N/A (Under 2 Reps)",
        "duration_std": 0.0,
        "depth_std": 0.0,
    }

st.title("📊 Workout Session Summary")
st.caption(f"Biomechanical Report for **{user}** • Exercise: **{exercise_name}** • Completed: **{display_time}**")

# Save status banner & retry mechanism
if session_id:
    st.success(f"💾 **Session Saved**: Workout session **#{session_id}** recorded in SQLite database.")
elif workout.get("save_error"):
    err_col1, err_col2 = st.columns([3, 1])
    with err_col1:
        st.error(f"⚠️ **Database Save Failed**: {workout.get('save_error')}")
    with err_col2:
        if st.button("🔄 Retry Save", type="primary", use_container_width=True):
            try:
                sid = save_workout(workout)
                workout["session_id"] = sid
                st.session_state.current_workout = workout
                st.session_state.session_saved = True
                st.cache_data.clear()
                st.success(f"Session #{sid} saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Retry failed: {e}")

# 1. Top KPI Metric Cards Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Completed", f"{completed_reps} Reps", delta="Completed")
with c2:
    st.metric("Clean Form Reps", f"{good_reps} Reps", delta=f"{good_reps}/{max(completed_reps, 1)}")
with c3:
    if completed_reps > 0 and workout.get("form_score") is not None:
        st.metric("Form Score (Weighted)", f"{workout['form_score']:.1f}%", delta="Physiological")
    else:
        st.metric("Form Score (Weighted)", "N/A", delta="No Reps")
with c4:
    if completed_reps > 0 and workout.get("strict_score") is not None:
        st.metric("Clean Ratio (Strict)", f"{workout['strict_score']:.1f}%", delta="Binary Standard")
    else:
        st.metric("Clean Ratio (Strict)", "N/A", delta="No Reps")
with c5:
    if completed_reps > 0:
        pace = duration / completed_reps
        st.metric("Pacing", f"{pace:.1f}s / rep", delta=f"{duration:.0f}s total")
    else:
        st.metric("Pacing", "N/A", delta=f"{duration:.1f}s total")

st.markdown("---")

# 2. Fatigue & Biomechanical Diagnostics Banner
if completed_reps >= 2:
    if fatigue.get("fatigue_detected", False):
        st.warning(f"⚡ **Fatigue Detected ({fatigue.get('level', 'MODERATE')})**: {fatigue.get('summary')}")
    else:
        st.success(f"🔋 **Muscular Endurance Stable**: {fatigue.get('summary')}")
else:
    st.info(r"ℹ️ **Fatigue & Consistency Analysis:** Insufficient repetition volume for kinetic drift assessment (requires $\ge 2$ completed repetitions).")

# 3. Two-Column Analytics Layout
col_charts, col_notes = st.columns([1.3, 1])

with col_charts:
    st.markdown("### 📈 Biomechanical Kinematics & Tempo")

    tab_trajectory, tab_distribution = st.tabs(["Rep-by-Rep Trajectory", "Form Breakdown Donut"])

    with tab_trajectory:
        if completed_reps > 0 and rep_history:
            rep_nums = [r["rep_number"] for r in rep_history]
            knee_depths = [r["min_knee_angle"] for r in rep_history]
            durations = [r["duration_frames"] / max(avg_fps, 1.0) for r in rep_history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            fig.patch.set_facecolor('#0f172a')
            ax1.set_facecolor('#0f172a')
            ax2.set_facecolor('#0f172a')

            # Subplot 1: Knee Depth Inflection
            ax1.plot(rep_nums, knee_depths, marker="o", color="#00e5ff", linewidth=2.5, label="Min Knee Angle (°)")
            ax1.axhline(95.0, color="#00e676", linestyle="--", linewidth=1.5, label="Parallel Depth (≤95°)")
            ax1.set_ylabel("Knee Angle (°)", color="#94a3b8")
            ax1.tick_params(colors="#94a3b8")
            ax1.grid(True, linestyle="--", alpha=0.2)
            ax1.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white', loc="upper left")
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#334155')
            ax1.spines['bottom'].set_color('#334155')

            # Subplot 2: Rep Duration / Tempo
            ax2.plot(rep_nums, durations, marker="s", color="#ff9800", linewidth=2.5, label="Rep Duration (sec)")
            ax2.set_xlabel("Repetition Number", color="#94a3b8")
            ax2.set_ylabel("Duration (s)", color="#94a3b8")
            ax2.tick_params(colors="#94a3b8")
            ax2.grid(True, linestyle="--", alpha=0.2)
            ax2.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white', loc="upper left")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#334155')
            ax2.spines['bottom'].set_color('#334155')

            st.pyplot(fig)
            plt.close()
        elif completed_reps > 0 and not rep_history:
            st.info("ℹ️ Repetition aggregate counts recorded, but detailed per-frame trajectory telemetry is unavailable for this session.")
        else:
            st.info("No repetition telemetry available for trajectory plotting.")

    with tab_distribution:
        if completed_reps > 0:
            labels = ["Clean Form", "Too Shallow"]
            counts = [good_reps, shallow_reps]
            colors = ["#00e676", "#ff9800"]

            plot_labels = [l for l, c in zip(labels, counts) if c > 0]
            plot_counts = [c for c in counts if c > 0]
            plot_colors = [col for col, c in zip(colors, counts) if c > 0]

            if plot_counts:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                wedges, texts, autotexts = ax.pie(
                    plot_counts,
                    labels=plot_labels,
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=plot_colors,
                    textprops=dict(color="white", fontsize=10),
                    wedgeprops=dict(width=0.45, edgecolor='#1e293b')
                )
                for at in autotexts:
                    at.set_color('black')
                    at.set_weight('bold')
                ax.axis('equal')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No repetition distribution available.")

            if forward_lean_reps > 0:
                st.caption(f"⚠️ **Posture Flag:** {forward_lean_reps} of {completed_reps} reps exhibited excessive torso forward lean (>38°).")
            else:
                st.caption("✅ **Torso Posture:** Zero excessive forward lean occurrences detected.")
        else:
            st.info("No completed repetitions recorded in this session.")

with col_notes:
    st.markdown("### 🤖 AI Coach Technique Takeaways")
    
    if completed_reps == 0:
        st.info(r"ℹ️ **No valid repetitions recorded in this session.** Step fully into frame and break parallel ($\le 95^\circ$) for squats to be counted.")
    else:
        score = workout.get("form_score") or 0.0
        if score >= 85:
            st.success("🌟 **Outstanding Biomechanics!** You consistently broke parallel and maintained an upright thoracic posture throughout.")
        elif score >= 70:
            st.info("👍 **Solid Training Session!** Good overall drive, but several repetitions suffered minor depth cutoffs as the set progressed.")
        else:
            st.warning("⚠️ **Technique Attention Required:** Several reps turned around prematurely or showed excessive torso hinge.")

    st.markdown("#### 🎯 Consistency & Stability Rating")
    if completed_reps >= 2:
        st.markdown(f"""
        - **Stability Rating:** `{consistency.get('rating', 'Good Control')}` (Score: **{consistency.get('consistency_score', 80.0)}/100**)
        - **Tempo Variability (std):** `±{consistency.get('duration_std', 0.2)}s`
        - **Depth Dispersion (std):** `±{consistency.get('depth_std', 2.0)}°`
        """)
    else:
        st.caption("Consistency rating available when 2 or more repetitions are completed.")

    st.markdown("""
    **Prescribed Technique Cues for Next Session:**
    - 🧘 **Mobility Warm-up:** 2 minutes of ankle dorsiflexion and deep goblet squats.
    - 🎯 **Depth Cue:** 'Spread the floor with your feet' to open hip capsule for full depth.
    - ⏱️ **Fatigue Management:** Rest 90 seconds between sets when tempo slowdown exceeds 20%.
    """)

    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔄 Start New Workout", type="primary", use_container_width=True):
            # Reset active session state for fresh workout
            st.session_state.session_saved = False
            st.session_state.active_workout_data = None
            st.session_state.saved_session_id = None
            st.switch_page("pages/1_Setup_Workout.py")
    with b2:
        if st.button("📈 View Full History", use_container_width=True):
            st.switch_page("pages/4_History.py")

"""
Page 4: Workout History & Progress Analytics
Displays historical workout records, longitudinal trends, error breakdowns,
fatigue evolution, and CSV data export from the SQLite database.
"""

import importlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import db.database
importlib.reload(db.database)
from db.database import get_user_workouts, get_user_stats, save_workout, clear_user_workouts

st.set_page_config(page_title="Workout History - AI Gym Coach", page_icon="📈", layout="wide")

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Please log in on the home page first.")
    st.page_link("app.py", label="Go to Login Screen", icon="🏠")
    st.stop()

user = st.session_state.get("username", "athlete")

st.title("📈 Workout History & Progress Analytics")
st.caption(f"Longitudinal SQLite Training Telemetry for **{user}**")

# Query real workouts from SQLite
workouts = get_user_workouts(user)

# If no workouts yet, offer one-click demo data initialization
if not workouts:
    st.info("No saved workout records found in SQLite for your username yet.")
    st.markdown("You can either complete a live workout or insert progressive demonstration sessions below:")

    col_demo, _ = st.columns([1, 2])
    with col_demo:
        if st.button("📥 Insert Realistic Sample Progression", type="primary", use_container_width=True):
            sample_sessions = [
                {
                    "username": user, "exercise_name": "Squats", "planned_sets": 3, "planned_reps": 8,
                    "completed_reps": 8, "good_reps": 4, "shallow_reps": 3, "forward_lean_reps": 1,
                    "form_score": 67.5, "duration_sec": 42.0, "avg_fps": 15.1,
                },
                {
                    "username": user, "exercise_name": "Squats", "planned_sets": 3, "planned_reps": 10,
                    "completed_reps": 10, "good_reps": 7, "shallow_reps": 2, "forward_lean_reps": 1,
                    "form_score": 83.0, "duration_sec": 51.0, "avg_fps": 14.9,
                },
                {
                    "username": user, "exercise_name": "Squats", "planned_sets": 3, "planned_reps": 10,
                    "completed_reps": 10, "good_reps": 9, "shallow_reps": 1, "forward_lean_reps": 0,
                    "form_score": 94.0, "duration_sec": 48.0, "avg_fps": 15.2,
                },
                {
                    "username": user, "exercise_name": "Squats", "planned_sets": 3, "planned_reps": 12,
                    "completed_reps": 12, "good_reps": 11, "shallow_reps": 1, "forward_lean_reps": 0,
                    "form_score": 95.0, "duration_sec": 56.0, "avg_fps": 15.0,
                },
            ]
            for s in sample_sessions:
                save_workout(s)
            st.success("Sample progression written to SQLite database!")
            st.rerun()

    st.stop()

# Load into DataFrame
df_history = pd.DataFrame(workouts)

# Lifetime Performance Metrics Row
stats = get_user_stats(user)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Sessions", stats["total_workouts"])
with m2:
    st.metric("Total Volume", f"{stats['total_reps']} Reps")
with m3:
    if stats["total_reps"] > 0 or stats["avg_form_score"] > 0:
        st.metric("Avg Form Score", f"{stats['avg_form_score']}%", delta="Target: ≥85%")
    else:
        st.metric("Avg Form Score", "N/A", delta="No Rated Reps")
with m4:
    if stats["total_reps"] > 0:
        clean_pct = stats.get("clean_ratio", 0.0)
        st.metric("Clean Rep Ratio", f"{clean_pct}%", delta="Strict Quality")
    else:
        st.metric("Clean Rep Ratio", "N/A", delta="Zero Volume")
with m5:
    total_mins = stats["total_duration_sec"] / 60.0
    st.metric("Total Training Time", f"{total_mins:.1f} mins")

st.markdown("---")

# Main Content Layout: Table & Controls (Left) vs Analytical Charts (Right)
col_left, col_right = st.columns([1.1, 1.4])

with col_left:
    col_h_title, col_h_ref = st.columns([2.5, 1])
    with col_h_title:
        st.markdown("### 📋 SQLite Session History")
    with col_h_ref:
        if st.button("🔄 Refresh", use_container_width=True, help="Force refresh SQLite session data"):
            st.cache_data.clear()
            st.rerun()
    
    table_cols = ["session_id", "date", "time", "exercise_name", "reps", "good", "shallow", "forward_lean", "score", "duration"]
    available_cols = [c for c in table_cols if c in df_history.columns]
    display_df = df_history[available_cols].copy()

    if "score" in display_df.columns:
        display_df["score"] = display_df["score"].apply(
            lambda s: f"{float(s):.1f}%" if (s is not None and pd.notna(s)) else "N/A"
        )
    if "duration" in display_df.columns:
        display_df["duration"] = display_df["duration"].apply(
            lambda d: f"{float(d):.1f}" if (d is not None and pd.notna(d)) else "0.0"
        )

    col_rename = {
        "session_id": "ID",
        "date": "Date",
        "time": "Time",
        "exercise_name": "Exercise",
        "reps": "Total",
        "good": "Clean",
        "shallow": "Shallow",
        "forward_lean": "Lean",
        "score": "Score (%)",
        "duration": "Duration (s)",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Action buttons: CSV Export & Database Maintenance
    st.markdown("#### 💾 Data Export & Options")
    csv_bytes = df_history.to_csv(index=False).encode("utf-8")
    
    exp_col, reset_col = st.columns(2)
    with exp_col:
        st.download_button(
            label="📥 Download CSV Dataset",
            data=csv_bytes,
            file_name=f"gym_coach_{user}_history.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export all training sessions for Excel, Pandas, or viva evaluation."
        )
    with reset_col:
        if st.button("🗑️ Reset User History", type="secondary", use_container_width=True):
            clear_user_workouts(user)
            st.warning("Workout records cleared for this user.")
            st.rerun()

with col_right:
    st.markdown("### 📊 Longitudinal Biomechanical Trends")
    
    # Chronological ordering (oldest to newest) for time-series charts
    sort_cols = ["timestamp", "session_id"] if "timestamp" in df_history.columns else ["session_id"]
    trend_df = df_history.sort_values(sort_cols, ascending=[True] * len(sort_cols)).copy()
    trend_df["session_label"] = [f"#{s}" for s in trend_df["session_id"]]
    trend_df["score_numeric"] = pd.to_numeric(trend_df["score"], errors="coerce")

    # Pace: only calculate for reps > 0 to avoid division by zero or fabricated pace
    trend_df["pace_sec_rep"] = [
        (d / r) if r > 0 else np.nan
        for d, r in zip(trend_df["duration"], trend_df["reps"])
    ]

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Form Progression", "Error Composition", "Volume & Pacing"
    ])

    with chart_tab1:
        # 1. Form Score Progression Line Chart (preserves gaps/markers for N/A zero-rep sessions)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        # Plot line connecting valid scores
        ax.plot(
            trend_df["session_label"], trend_df["score_numeric"],
            marker="o", color="#00e5ff", linewidth=2.5, label="Form Quality Score (%)"
        )

        # Clearly mark zero-rep / unrated sessions with distinct marker rather than fake 0 score
        na_mask = trend_df["score_numeric"].isna()
        if na_mask.any():
            ax.scatter(
                trend_df.loc[na_mask, "session_label"],
                [50.0] * na_mask.sum(),
                marker="x", color="#94a3b8", s=80, label="N/A (0 Reps)", zorder=5
            )
            for _, r in trend_df[na_mask].iterrows():
                ax.annotate("N/A", (r["session_label"], 53.0), color="#94a3b8", fontsize=8, ha="center")

        ax.axhline(85, color="#00e676", linestyle="--", linewidth=1.5, label="Proficiency Benchmark (85%)")
        ax.fill_between(
            trend_df["session_label"], 85, 105,
            color="#00e676", alpha=0.08, label="Optimal Form Zone"
        )

        ax.set_ylim(40, 105)
        ax.set_ylabel("Score (%)", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white', loc="lower right")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color('#334155')

        st.pyplot(fig)
        plt.close()

    with chart_tab2:
        # 2. Grouped Bar Chart of Error Distribution (avoids false stacking of overlapping lean faults)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        x_indices = np.arange(len(trend_df))
        bar_width = 0.25

        good = trend_df["good"].values
        shallow = trend_df["shallow"].values
        lean = trend_df["forward_lean"].values

        ax.bar(x_indices - bar_width, good, width=bar_width, label="Clean Form", color="#00e676")
        ax.bar(x_indices, shallow, width=bar_width, label="Too Shallow", color="#ff9800")
        ax.bar(x_indices + bar_width, lean, width=bar_width, label="Forward Lean (Fault)", color="#f44336")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(trend_df["session_label"], color="#94a3b8")
        ax.set_ylabel("Repetitions", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.2, axis="y")
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color('#334155')

        st.pyplot(fig)
        plt.close()

    with chart_tab3:
        # 3. Volume and Pacing (Seconds / Rep) Evolution (no fabricated pace for 0-rep sessions)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')

        x_indices = np.arange(len(trend_df))
        ax1.bar(x_indices, trend_df["reps"], width=0.45, color="#3b82f6", alpha=0.8, label="Completed Reps (Volume)")
        ax1.set_ylabel("Total Reps", color="#3b82f6")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(trend_df["session_label"], color="#94a3b8")
        ax1.tick_params(axis="y", labelcolor="#3b82f6")
        ax1.tick_params(axis="x", labelcolor="#94a3b8")

        ax2 = ax1.twinx()
        ax2.plot(x_indices, trend_df["pace_sec_rep"], color="#eab308", marker="D", linewidth=2.0, label="Pace (s/rep)")
        ax2.set_ylabel("Pace (s / rep)", color="#eab308")
        ax2.tick_params(axis="y", labelcolor="#eab308")

        for spine in ["top"]:
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)

        st.pyplot(fig)
        plt.close()

# ----------------- SESSION DRILL-DOWN INSPECTOR -----------------
st.markdown("---")
st.markdown("### 🔍 Historical Session Deep Dive")

session_options = {}
for _, row in df_history.iterrows():
    s_score = f"{row['score']:.1f}%" if (row['score'] is not None and pd.notna(row['score'])) else "N/A"
    lbl = f"Session #{row['session_id']} ({row['date']} • {row['reps']} reps • {s_score} score)"
    session_options[lbl] = row['session_id']

selected_label = st.selectbox("Select a past session to inspect details:", list(session_options.keys()))
selected_id = session_options[selected_label]
selected_session = df_history[df_history["session_id"] == selected_id].iloc[0]

d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    st.metric("Completed Reps", f"{selected_session['reps']} / {selected_session.get('planned_reps', 10)}")
with d2:
    if selected_session['reps'] > 0:
        clean_str = f"({selected_session['good'] / selected_session['reps'] * 100:.1f}%)"
    else:
        clean_str = "(N/A)"
    st.metric("Clean Reps", f"{selected_session['good']} {clean_str}")
with d3:
    st.metric("Shallow Reps", selected_session['shallow'])
with d4:
    st.metric("Forward Lean Reps", selected_session['forward_lean'])
with d5:
    if selected_session['reps'] > 0:
        pace_val = selected_session['duration'] / selected_session['reps']
        st.metric("Rep Pacing", f"{pace_val:.2f}s / rep")
    else:
        st.metric("Rep Pacing", "N/A (0 reps)")

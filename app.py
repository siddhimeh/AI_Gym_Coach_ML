"""
AI Gym Coach — Main Application Entry Point
Browser-based computer vision fitness coaching application.
Phase 7: Streamlit Application Shell & Session Management.
"""

import os
import sys
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import APP_DISCLAIMER, EXERCISES

# Page Configuration
st.set_page_config(
    page_title="AI Gym Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Modern Aesthetics
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #141824 0%, #0c0f17 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hero Title Styling */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00e5ff 0%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #00e5ff;
        margin: 0.3rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Disclaimer Callout */
    .disclaimer-box {
        background: rgba(234, 179, 8, 0.08);
        border-left: 4px solid #eab308;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 0.85rem;
        color: #fde047;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = "Akarsh"
if "selected_exercise" not in st.session_state:
    st.session_state.selected_exercise = "squats"
if "target_sets" not in st.session_state:
    st.session_state.target_sets = 3
if "target_reps" not in st.session_state:
    st.session_state.target_reps = 10
if "current_workout" not in st.session_state:
    st.session_state.current_workout = {
        "completed_reps": 8,
        "good_reps": 7,
        "shallow_reps": 1,
        "forward_lean_reps": 0,
        "form_score": 87.5,
        "duration_sec": 42.0,
    }
if "history_data" not in st.session_state:
    # Sample baseline history for Phase 7 display
    st.session_state.history_data = [
        {"session_id": 1, "date": "2026-10-10", "reps": 10, "good": 9, "shallow": 1, "score": 90.0, "duration": 48},
        {"session_id": 2, "date": "2026-10-11", "reps": 12, "good": 10, "shallow": 2, "score": 83.3, "duration": 55},
        {"session_id": 3, "date": "2026-10-12", "reps": 10, "good": 10, "shallow": 0, "score": 100.0, "duration": 45},
    ]


# Sidebar Navigation & User Info
with st.sidebar:
    st.image("assets/squat_reference.jpg", caption="Real-time Pose Tracking", use_container_width=True)
    st.markdown("### 🏋️ AI Gym Coach")
    if st.session_state.authenticated:
        st.markdown(f"**Logged in as:** `{st.session_state.username}`")
        if st.button("Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_workout = None
            st.session_state.active_workout_data = None
            st.session_state.session_saved = False
            st.session_state.saved_session_id = None
            st.cache_data.clear()
            st.rerun()
    else:
        st.markdown("*Please enter a username to begin.*")

    st.markdown("---")
    st.markdown("#### System Telemetry")
    st.caption("• **Vision Model:** MediaPipe Tasks Pose")
    st.caption("• **Inference Engine:** TFLite CPU (XNNPACK)")
    st.caption("• **Target Resolution:** 640 x 480 @ 30 FPS")
    st.caption("• **Course Project:** SRM IST B.Tech CSE")


# Main Dashboard
st.markdown('<div class="hero-title">AI GYM COACH</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Real-Time Computer Vision & Machine Learning Exercise Form Analysis</div>', unsafe_allow_html=True)

import importlib
import db.database
importlib.reload(db.database)
from db.database import get_or_create_user, get_user_stats

if not st.session_state.authenticated:
    # Login Screen
    st.markdown("### 🔐 User Login")
    st.info("Demonstration Login: Enter any student or user handle to begin your workout session.")

    col1, col2 = st.columns([2, 1])
    with col1:
        username_input = st.text_input(
            "Username / Handle",
            value=st.session_state.username,
            placeholder="e.g. Akarsh, Siddharth, Alex",
        )
        if st.button("🚀 Enter Workout Dashboard", type="primary", use_container_width=True):
            if username_input.strip():
                clean_name = username_input.strip()
                user_rec = get_or_create_user(clean_name)
                st.session_state.username = user_rec["username"]
                st.session_state.user_id = user_rec["id"]
                st.session_state.authenticated = True
                st.success(f"Welcome, {st.session_state.username} (ID: #{user_rec['id']})!")
                st.rerun()
            else:
                st.error("Please enter a valid username.")

    with col2:
        st.markdown("""
        **College Viva Note on Authentication:**
        > In this college prototype, we use a single username parameter without password hashing to keep the local demonstration rapid and transparent. Production fitness platforms implement secure OAuth2 / bcrypt password hashing.
        """)

else:
    # Logged In: Welcome Dashboard
    st.success(f"Welcome back, **{st.session_state.username}**! Ready for your workout?")

    # Metrics Overview Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Exercise</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #00e5ff;">Squats</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Side-View Camera</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Goal</div>
            <div class="metric-value">{st.session_state.target_sets * st.session_state.target_reps}</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">{st.session_state.target_sets} sets × {st.session_state.target_reps} reps</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pose Detector</div>
            <div class="metric-value" style="color: #00e676; font-size: 1.4rem;">ONLINE</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">33 Landmarks (CPU)</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Database Status</div>
            <div class="metric-value" style="color: #00e676; font-size: 1.4rem;">CONNECTED</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">SQLite Local Store</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Navigation Cards
    st.markdown("### 📌 Select an Action")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        #### 1. Setup Workout
        Configure target sets, repetitions, and review biomechanical angle guidelines.
        """)
        st.page_link("pages/1_Setup_Workout.py", label="Go to Setup ⚙️", icon="⚙️", use_container_width=True)

    with c2:
        st.markdown("""
        #### 2. Live Workout
        Start camera session with live skeleton overlay and repetition tracking.
        """)
        st.page_link("pages/2_Live_Workout.py", label="Start Workout 🎥", icon="🎥", use_container_width=True)

    with c3:
        st.markdown("""
        #### 3. Summary
        Inspect completed repetition counts, depth analytics, and form scores.
        """)
        st.page_link("pages/3_Workout_Summary.py", label="View Summary 📊", icon="📊", use_container_width=True)

    with c4:
        st.markdown("""
        #### 4. History
        Browse previous workouts and view longitudinal progress graphs.
        """)
        st.page_link("pages/4_History.py", label="View History 📈", icon="📈", use_container_width=True)


# Educational Viva Disclaimer Callout
st.markdown(f'<div class="disclaimer-box">⚠️ <b>Disclaimer:</b> {APP_DISCLAIMER}</div>', unsafe_allow_html=True)

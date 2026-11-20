"""
Page 2: Live Workout & Real-Time Biofeedback
Phase 9: Live Streamlit Integration with MediaPipe Pose & Squat State Machine.
Supports both physical webcam hardware and recorded benchmark video clips.
"""

import os
import sys
import time
import uuid
import logging
from datetime import datetime, timezone
import cv2
import streamlit as st

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import importlib
import exercises.squat
import db.database
importlib.reload(exercises.squat)
importlib.reload(db.database)

from exercises.squat import SquatCounter
from pose.detector import PoseDetector
from utils.video import draw_hud_panel, draw_feedback_banner
from utils.metrics import calculate_workout_summary
from db.database import save_workout

try:
    from db.database import format_timestamp_for_display
except (ImportError, AttributeError):
    def format_timestamp_for_display(ts_str: str | None) -> tuple[str, str, str]:
        now_local = datetime.now()
        return (
            now_local.strftime("%Y-%m-%d"),
            now_local.strftime("%H:%M:%S"),
            now_local.strftime("%Y-%m-%d %H:%M:%S"),
        )

from coach import LLMCoach, TTSCoach

st.set_page_config(page_title="Live Workout - AI Gym Coach", page_icon="🎥", layout="wide")

if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Please log in on the home page first.")
    st.page_link("app.py", label="Go to Login Screen", icon="🏠")
    st.stop()

# User & Session Setup
user = st.session_state.get("username", "Athlete")
sets = st.session_state.get("target_sets", 3)
reps = st.session_state.get("target_reps", 10)

# Initialize workout session token if starting new workout
if "session_token" not in st.session_state:
    st.session_state.session_token = uuid.uuid4().hex[:12]
    st.session_state.session_saved = False
    st.session_state.saved_session_id = None
    st.session_state.active_workout_data = None


def finalize_and_save_workout(workout_data: dict, intentional_finish: bool = False) -> int | None:
    """
    Persists workout to SQLite database ensuring:
    1. Valid reps (>0) are saved.
    2. Intentionally finished 0-rep workouts (stand still & finish) are recorded with score=None.
    3. Abandoned / zero-frame accidental aborts are not recorded.
    4. Exact session_id is returned and cached in session_state.
    5. Cache is invalidated so History & Summary show fresh data.
    """
    if not workout_data:
        return None

    if st.session_state.get("session_saved") and st.session_state.get("saved_session_id"):
        return st.session_state.saved_session_id

    reps_count = int(workout_data.get("completed_reps", 0))
    dur = float(workout_data.get("duration_sec", 0.0))
    frames = int(workout_data.get("frame_count", 0))

    # Evaluate save conditions:
    # A. Completed reps > 0: always save.
    # B. Intentional finish: always save (even if 0 reps).
    # Otherwise treat as abandoned/cancelled without saving.
    if not (reps_count > 0 or intentional_finish):
        return None

    rep_history = workout_data.get("rep_history") or []
    summary = calculate_workout_summary(
        total_reps=reps_count,
        good_reps=int(workout_data.get("good_reps", 0)),
        shallow_reps=int(workout_data.get("shallow_reps", 0)),
        forward_lean_reps=int(workout_data.get("forward_lean_reps", 0)),
        duration_sec=dur,
        avg_fps=float(workout_data.get("avg_fps", 0.0)),
        rep_history=rep_history,
    )

    # For zero-rep sessions, score is undefined (None / NULL in database)
    form_score = summary["form_score"] if reps_count > 0 else None
    strict_score = summary["strict_score"] if reps_count > 0 else None
    utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _, _, display_time = format_timestamp_for_display(utc_timestamp)

    workout_rec = {
        "username": user,
        "exercise_name": workout_data.get("exercise_name", "Squats"),
        "planned_sets": sets,
        "planned_reps": reps,
        "completed_reps": reps_count,
        "good_reps": int(workout_data.get("good_reps", 0)),
        "shallow_reps": int(workout_data.get("shallow_reps", 0)),
        "forward_lean_reps": int(workout_data.get("forward_lean_reps", 0)),
        "form_score": form_score,
        "strict_score": strict_score,
        "duration_sec": summary["duration_sec"],
        "avg_fps": summary["avg_fps"],
        "rep_history": rep_history,
        "fatigue": summary["fatigue"],
        "consistency": summary["consistency"],
        "session_token": st.session_state.session_token,
        "timestamp": utc_timestamp,
        "display_time": display_time,
    }

    try:
        sid = save_workout(workout_rec)
        workout_rec["session_id"] = sid
        st.session_state.current_workout = workout_rec
        st.session_state.session_saved = True
        st.session_state.saved_session_id = sid
        st.cache_data.clear()
        return sid
    except Exception as e:
        logger.error(f"Failed to save workout session: {e}", exc_info=True)
        workout_rec["save_error"] = str(e)
        st.session_state.current_workout = workout_rec
        return None


st.title("🎥 Live Workout Session")
st.caption(f"Athlete: **{user}** | Exercise: **Squats (Side-View)** | Planned Goal: **{sets} sets × {reps} reps**")

col_video, col_side = st.columns([1.6, 1])

with col_side:
    st.markdown("### 📖 Correct-Form Reference")
    gif_path = os.path.join(ROOT_DIR, "assets", "squat.gif")
    ref_img_path = os.path.join(ROOT_DIR, "assets", "squat_reference.jpg")
    
    if os.path.exists(gif_path):
        st.image(gif_path, caption="Biomechanical Reference: Full Depth Squat", width=320)
    elif os.path.exists(ref_img_path):
        st.image(ref_img_path, caption="Biomechanical Reference: Full Depth Squat", width=320)

    st.markdown("### ⚙️ Video Source Options")
    video_source_type = st.radio(
        "Select Camera Input Source:",
        ["Live Webcam (Camera 0)", "Benchmark Video Clip (Sample)"],
        index=0,
        help="Use 'Live Webcam' for real-time exercise practice, or 'Benchmark Video' to evaluate recorded form without moving your laptop."
    )

    st.markdown("### 🎙️ Coaching Audio & AI Options")
    enable_audio = st.toggle("🔊 Spoken Audio Cues (gTTS)", value=False, help="Speaks real-time form corrections out loud")
    use_llm = st.toggle("🤖 AI Coach Insights (Throttled)", value=True, help="Generates adaptive coaching feedback with 6s throttling")

    st.markdown("---")
    st.markdown("### 💡 Setup & Alignment Tips")
    st.markdown(r"""
    - **Positioning:** Stand 6–8 feet away so full body is visible.
    - **Camera Height:** Waist/hip level gives the most accurate knee angle.
    - **Orientation:** Turn sideways to the camera (pure side-view).
    - **Movement:** Break parallel at the bottom ($\le 95^\circ$) and keep chest up.
    """)

with col_video:
    st.markdown("### Live Viewport & Telemetry HUD")

    # Placeholder containers for dynamic streaming
    video_placeholder = st.empty()
    status_banner_placeholder = st.empty()
    
    # Telemetry metric placeholders
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        metric_reps = st.empty()
    with col_m2:
        metric_knee = st.empty()
    with col_m3:
        metric_lean = st.empty()
    with col_m4:
        metric_state = st.empty()

    # Default idle telemetry display before clicking Start
    metric_reps.metric("Completed Reps", f"0 / {reps}", delta="Set 1 of 3")
    metric_knee.metric("Knee Angle", "--", delta="Awaiting Start")
    metric_lean.metric("Back Lean", "--", delta="Awaiting Start")
    metric_state.metric("FSM State", "IDLE", delta="Camera Off")
    status_banner_placeholder.info("Click '▶️ Start Workout' below to activate camera capture and AI coaching.")

    # Show idle poster if workout not running
    video_placeholder.image(ref_img_path if os.path.exists(ref_img_path) else gif_path, caption="Camera Standby - Click Start Workout to begin", use_container_width=True)

    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        run_workout = st.checkbox("▶️ START WORKOUT / CAMERA", value=False, help="Activates webcam capture and real-time landmark estimation")
    with ctrl_col2:
        finish_clicked = st.button("⏹️ Finish & View Summary", type="secondary", use_container_width=True)

    if finish_clicked:
        if not st.session_state.get("session_saved"):
            active_data = st.session_state.get("active_workout_data")
            if not active_data:
                active_data = {
                    "username": user,
                    "exercise_name": "Squats",
                    "planned_sets": sets,
                    "planned_reps": reps,
                    "completed_reps": 0,
                    "good_reps": 0,
                    "shallow_reps": 0,
                    "forward_lean_reps": 0,
                    "duration_sec": 0.0,
                    "avg_fps": 0.0,
                    "rep_history": [],
                    "frame_count": 0,
                    "session_token": st.session_state.get("session_token"),
                }
            finalize_and_save_workout(active_data, intentional_finish=True)
        st.switch_page("pages/3_Workout_Summary.py")

    if run_workout:
        # Determine source
        if "Benchmark" in video_source_type:
            src_path = os.path.join(ROOT_DIR, "data", "clips", "squat_sample.webm")
            cap = cv2.VideoCapture(src_path)
            is_looping_video = True
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            is_looping_video = False

        if not cap.isOpened():
            st.error("🚨 Could not access the selected video device or file. Ensure no other application is using your webcam.")
        else:
            stop_at_target = st.session_state.get("stop_at_target", True)
            detector = PoseDetector()
            try:
                counter = SquatCounter(target_reps=reps, stop_at_target=stop_at_target)
            except TypeError:
                counter = SquatCounter()
                if hasattr(counter, "set_target"):
                    counter.set_target(target_reps=reps, stop_at_target=stop_at_target)
                else:
                    counter.target_reps = reps
                    counter.stop_at_target = stop_at_target
            llm_coach = LLMCoach(cooldown=6.0)
            tts_coach = TTSCoach(cooldown=5.0)
            audio_placeholder = st.empty()

            fps = 0.0
            prev_time = time.time()
            start_time = time.time()
            frame_count = 0

            status_banner_placeholder.success("🟢 Camera Active — AI Gym Coach is tracking your form. Step into frame!")

            try:
                while run_workout:
                    curr_time = time.time()
                    ret, frame = cap.read()

                    if not ret:
                        if is_looping_video:
                            # Loop demo video
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            st.warning("Webcam feed disconnected or ended.")
                            break

                    frame_count += 1
                    dt = curr_time - prev_time
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                    prev_time = curr_time

                    # Force 640x480 resolution for low-latency CPU processing
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))

                    # 1. Pose estimation
                    result = detector.find_pose(frame)
                    kp_data = detector.extract_landmarks(result, frame.shape)

                    # 2. Skeleton drawing
                    annotated = detector.draw_skeleton(frame, kp_data)

                    # 3. Squat state machine update
                    state_data = counter.update(kp_data)

                    # Continuously preserve live telemetry in session state
                    st.session_state.active_workout_data = {
                        "username": user,
                        "exercise_name": "Squats",
                        "planned_sets": sets,
                        "planned_reps": reps,
                        "completed_reps": counter.rep_count,
                        "good_reps": counter.good_reps,
                        "shallow_reps": counter.shallow_reps,
                        "forward_lean_reps": counter.forward_lean_reps,
                        "duration_sec": curr_time - start_time,
                        "avg_fps": fps,
                        "rep_history": counter.get_rep_history(),
                        "frame_count": frame_count,
                        "session_token": st.session_state.session_token,
                    }

                    # 4. Generate AI coaching feedback (throttled 6s cooldown)
                    if use_llm and state_data["last_rep_status"] != "none":
                        coaching_msg = llm_coach.generate_feedback(
                            status=state_data["last_rep_status"],
                            knee_angle=state_data["knee_angle"],
                            back_lean=state_data["back_lean"],
                            rep_count=state_data["rep_count"]
                        )
                    else:
                        coaching_msg = state_data["feedback"]

                    # 5. Spoken Audio Feedback (gTTS, throttled 5s cooldown)
                    if enable_audio and state_data["last_rep_status"] != "none":
                        audio_path = tts_coach.speak(coaching_msg)
                        if audio_path:
                            audio_placeholder.audio(audio_path, autoplay=True)

                    # 6. HUD Telemetry overlay
                    annotated = draw_hud_panel(
                        frame=annotated,
                        rep_count=state_data["rep_count"],
                        state=state_data["state"],
                        knee_angle=state_data["knee_angle"],
                        back_lean=state_data["back_lean"],
                        fps=fps,
                        good_reps=state_data["good_reps"],
                    )

                    # 7. Feedback banner overlay
                    annotated = draw_feedback_banner(
                        frame=annotated,
                        feedback_text=coaching_msg,
                        status=state_data["last_rep_status"],
                    )

                    # Convert BGR to RGB for Streamlit rendering
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, use_container_width=True)

                    # Update Streamlit live metrics
                    k_val = state_data["knee_angle"]
                    b_val = state_data["back_lean"]
                    metric_reps.metric("Completed Reps", f"{state_data['rep_count']} / {reps}", delta=f"{state_data['good_reps']} clean")
                    metric_knee.metric("Knee Angle", f"{k_val:.0f}°" if k_val is not None else "--", delta="Target: <=95°")
                    metric_lean.metric("Back Lean", f"{b_val:.0f}°" if b_val is not None else "--", delta="Target: <=38°")
                    metric_state.metric("FSM State", state_data["state"].upper(), delta=f"{fps:.1f} FPS")

                    # Live coaching alert in Streamlit UI
                    if state_data.get("target_reached"):
                        status_banner_placeholder.success(f"🎯 **Target Achieved!** Completed {state_data['rep_count']} of {reps} planned reps.")
                    elif state_data["last_rep_status"] == "good":
                        status_banner_placeholder.success(f"🟢 **Form Feedback:** {coaching_msg}")
                    elif state_data["last_rep_status"] in ("too_shallow", "back_leaning_forward"):
                        status_banner_placeholder.warning(f"⚠️ **Form Feedback:** {coaching_msg}")
                    else:
                        status_banner_placeholder.info(f"ℹ️ **Status:** {coaching_msg}")

            finally:
                total_duration = time.time() - start_time
                avg_fps = frame_count / total_duration if total_duration > 0 else 0.0
                cap.release()
                detector.close()

                workout_data = {
                    "username": user,
                    "exercise_name": "Squats",
                    "planned_sets": sets,
                    "planned_reps": reps,
                    "completed_reps": counter.rep_count,
                    "good_reps": counter.good_reps,
                    "shallow_reps": counter.shallow_reps,
                    "forward_lean_reps": counter.forward_lean_reps,
                    "duration_sec": total_duration,
                    "avg_fps": avg_fps,
                    "rep_history": counter.get_rep_history(),
                    "frame_count": frame_count,
                    "session_token": st.session_state.session_token,
                }
                st.session_state.active_workout_data = workout_data

                # If validated reps were performed and not saved yet, save immediately
                if counter.rep_count > 0 and not st.session_state.get("session_saved"):
                    sid = finalize_and_save_workout(workout_data, intentional_finish=False)
                    if sid:
                        status_banner_placeholder.success(f"Workout session #{sid} successfully saved to SQLite!")
                elif not st.session_state.get("session_saved"):
                    status_banner_placeholder.info(
                        "Camera stopped. Click '⏹️ Finish & View Summary' to complete and record this session, or restart camera."
                    )


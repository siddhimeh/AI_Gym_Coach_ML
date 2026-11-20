"""
Page 1: Workout Setup & Configuration
Allows the user to select exercise, configure target sets/reps,
and review biomechanical guidelines before starting the camera.
"""

import uuid
import streamlit as st

st.set_page_config(page_title="Workout Setup - AI Gym Coach", page_icon="⚙️", layout="wide")

# Check authentication
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Please log in on the home page first.")
    st.page_link("app.py", label="Go to Login Screen", icon="🏠")
    st.stop()

st.title("⚙️ Configure Your Workout")
st.markdown("Select your exercise parameters and review camera positioning guidelines.")

col_setup, col_guide = st.columns([1.2, 1])

with col_setup:
    st.markdown("### 1. Workout Parameters")
    
    exercise = st.selectbox(
        "Select Exercise",
        ["Squats (Side-View)"],
        index=0,
        help="Initially supporting squats using side-view camera orientation."
    )
    st.session_state.selected_exercise = exercise

    sets = st.slider(
        "Target Sets",
        min_value=1,
        max_value=10,
        value=st.session_state.get("target_sets", 3),
        step=1,
    )
    st.session_state.target_sets = sets

    reps = st.slider(
        "Target Repetitions Per Set",
        min_value=1,
        max_value=30,
        value=st.session_state.get("target_reps", 10),
        step=1,
    )
    st.session_state.target_reps = reps

    total_target = sets * reps
    st.info(f"🎯 **Total Goal:** {sets} sets × {reps} reps = **{total_target} total repetitions**")

    stop_at_target = st.toggle(
        "🛑 Hard Stop at Target Reps",
        value=st.session_state.get("stop_at_target", True),
        help="When enabled, locks the repetition counter once your planned target is reached. When disabled, extra repetitions are permitted as bonus volume."
    )
    st.session_state.stop_at_target = stop_at_target

    st.markdown("---")
    if st.button("🚀 Confirm & Proceed to Live Workout", type="primary", use_container_width=True):
        st.session_state.session_token = uuid.uuid4().hex[:12]
        st.session_state.session_saved = False
        st.session_state.saved_session_id = None
        st.session_state.active_workout_data = None
        st.success("Configuration saved!")
        st.switch_page("pages/2_Live_Workout.py")

with col_guide:
    st.markdown("### 2. Camera Positioning Guide")
    st.image("assets/squat_reference.jpg", caption="Ideal Side-View Camera Orientation", use_container_width=True)

    st.markdown("""
    **Key Camera Guidelines:**
    - 📐 **Distance:** Place laptop approximately **6–8 feet** away.
    - 🎥 **Height:** Position camera roughly at **waist / hip height**.
    - 🧍 **Visibility:** Ensure your head, hips, knees, and feet are fully inside the frame.
    - 💡 **Lighting:** Ensure good ambient lighting so joints are distinct from the background.
    """)

    st.markdown("### 3. Biomechanical Targets")
    st.markdown(r"""
    | Joint Measurement | Target | Form Status |
    |---|---|---|
    | **Knee Depth** | $\le 95^\circ$ | ✅ Full Depth (Parallel) |
    | **Standing Lockout** | $\ge 160^\circ$ | 🟢 Complete Repetition |
    | **Torso Forward Lean** | $\le 35^\circ$ | 🛡️ Safe Spinal Alignment |
    | **Excessive Lean** | $> 38^\circ$ | ⚠️ Warning (Chest Down) |
    """)

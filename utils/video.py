"""
Video & HUD Overlay Utilities
Draws stylish HUD cards, telemetry badges, and feedback banners on OpenCV frames.
Optimized for 640x480 resolution on low-power CPU hardware.
"""

import cv2
import numpy as np


def draw_hud_panel(
    frame: np.ndarray,
    rep_count: int,
    state: str,
    knee_angle: float | None,
    back_lean: float | None,
    fps: float,
    good_reps: int,
) -> np.ndarray:
    """
    Renders a semi-transparent telemetry HUD card on the top-left of the frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel dimensions
    card_w = 260
    card_h = 160
    x1, y1 = 15, 15
    x2, y2 = x1 + card_w, y1 + card_h

    # Draw dark translucent glassmorphism background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 33), -1)
    alpha = 0.78
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Card border with subtle accent glow
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 75, 95), 1, cv2.LINE_AA)

    # 1. Title & Live FPS
    cv2.putText(frame, "AI GYM COACH", (x1 + 12, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (x2 - 68, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 195), 1, cv2.LINE_AA)

    # 2. Large Rep Counter Badge
    cv2.putText(frame, "REPS", (x1 + 14, y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{rep_count:02d}", (x1 + 12, y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # Sub-stat: Good reps count
    cv2.putText(frame, f"({good_reps} clean)", (x1 + 80, y1 + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 118), 1, cv2.LINE_AA)

    # 3. State Pill Badge
    state_colors = {
        "standing": (180, 190, 205),
        "descending": (0, 165, 255),  # Orange
        "bottom": (0, 230, 118),      # Bright Green
        "ascending": (0, 229, 255),     # Teal/Yellow
    }
    st_color = state_colors.get(state.lower(), (200, 200, 200))
    state_str = state.upper()

    cv2.rectangle(frame, (x1 + 14, y1 + 120), (x1 + 120, y1 + 145), (40, 48, 62), -1)
    cv2.putText(frame, state_str, (x1 + 22, y1 + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, st_color, 1, cv2.LINE_AA)

    # 4. Joint Telemetry (Right column inside card)
    k_str = f"{knee_angle:.0f}°" if knee_angle is not None else "--"
    b_str = f"{back_lean:.0f}°" if back_lean is not None else "--"

    col2_x = x1 + 140
    cv2.putText(frame, "Knee Depth:", (col2_x, y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, k_str, (col2_x, y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 229, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Back Lean:", (col2_x, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, b_str, (col2_x, y1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 180, 0), 2, cv2.LINE_AA)

    return frame


def draw_feedback_banner(frame: np.ndarray, feedback_text: str, status: str = "none") -> np.ndarray:
    """
    Renders a dynamic coaching feedback banner across the lower third of the screen.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    banner_h = 44
    y1 = h - banner_h - 15
    y2 = h - 15
    x1, x2 = 20, w - 20

    # Choose banner accent color based on status
    if status == "good":
        accent = (0, 200, 83)      # Emerald green
    elif status in ("too_shallow", "back_leaning_forward"):
        accent = (0, 100, 255)     # Amber warning
    else:
        accent = (55, 65, 81)      # Neutral dark gray

    # Translucent banner
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Accent left border indicator
    cv2.rectangle(frame, (x1, y1), (x1 + 6, y2), accent, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 80, 95), 1, cv2.LINE_AA)

    # Banner message
    cv2.putText(frame, feedback_text, (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

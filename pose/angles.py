"""
Joint Angle Calculation Module
Calculates biomechanical angles for lower-body fitness exercises:
- Knee Angle (Hip-Knee-Ankle): Measures squat depth
- Hip Angle (Shoulder-Hip-Knee): Measures hip hinge
- Back Lean Angle (Hip-Shoulder vs Vertical): Measures torso posture / forward lean
"""

import numpy as np


def _to_point_coords(pt) -> tuple[float, float]:
    """
    Helper to extract (x, y) coordinates from a tuple, list, or dictionary.
    Supports both pixel coordinates ('x_px', 'y_px') for isotropic Euclidean geometry
    and normalized coordinates ('x_norm', 'y_norm').
    Rejects NaN and infinity values.
    """
    if isinstance(pt, dict):
        if "x_px" in pt and "y_px" in pt:
            x, y = float(pt["x_px"]), float(pt["y_px"])
        elif "x_norm" in pt and "y_norm" in pt:
            x, y = float(pt["x_norm"]), float(pt["y_norm"])
        elif "x" in pt and "y" in pt:
            x, y = float(pt["x"]), float(pt["y"])
        else:
            raise ValueError(f"Dictionary must contain 'x_norm'/'y_norm' or 'x_px'/'y_px': {pt}")
    elif isinstance(pt, (list, tuple, np.ndarray)):
        if len(pt) < 2:
            raise ValueError(f"Point sequence must have at least 2 values (x, y): {pt}")
        x, y = float(pt[0]), float(pt[1])
    else:
        raise TypeError(f"Unsupported point format: {type(pt)}")

    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"Point coordinates must be finite (got x={x}, y={y})")

    return x, y


def calculate_3point_angle(a, b, c) -> float:
    """
    Calculates the 2D interior angle at vertex B formed by line segments BA and BC.
    Formula:
        angle = |atan2(Ay - By, Ax - Bx) - atan2(Cy - By, Cx - Bx)|
        Adjusted to interior angle in range [0, 180] degrees.

    Args:
        a: First endpoint (e.g., Hip)
        b: Vertex point (e.g., Knee)
        c: Second endpoint (e.g., Ankle)

    Returns:
        float: Angle in degrees in range [0.0, 180.0].
    """
    ax, ay = _to_point_coords(a)
    bx, by = _to_point_coords(b)
    cx, cy = _to_point_coords(c)

    # Check for degenerate vectors (identical joint positions)
    len_ba = np.hypot(ax - bx, ay - by)
    len_bc = np.hypot(cx - bx, cy - by)
    if len_ba == 0 or len_bc == 0:
        return 180.0

    # Vectors BA and BC
    radians = np.arctan2(ay - by, ax - bx) - np.arctan2(cy - by, cx - bx)
    angle = np.abs(radians * 180.0 / np.pi)

    # Convert to interior angle
    if angle > 180.0:
        angle = 360.0 - angle

    return float(np.round(angle, 2))


def calculate_knee_angle(hip, knee, ankle) -> float:
    """
    Calculates the knee flexion angle formed by Hip -> Knee -> Ankle.
    Biomechanical reference:
        - Standing straight: ~170 - 180 degrees
        - Half squat: ~110 - 140 degrees
        - Parallel/Deep squat: ~70 - 90 degrees
    """
    return calculate_3point_angle(hip, knee, ankle)


def calculate_hip_angle(shoulder, hip, knee) -> float:
    """
    Calculates the hip hinge angle formed by Shoulder -> Hip -> Knee.
    Biomechanical reference:
        - Standing straight: ~160 - 180 degrees
        - Bottom of squat: ~60 - 95 degrees (depending on torso and thigh geometry)
    """
    return calculate_3point_angle(shoulder, hip, knee)


def calculate_back_lean(shoulder, hip) -> float:
    """
    Calculates the angle between the torso vector (Hip -> Shoulder) and the vertical axis.
    Coordinate reference:
        In computer vision, y=0 is at the top, so vertical upwards from Hip is (0, -1).
        Back lean is 0 degrees when the torso is perfectly upright.
        Back lean increases as the person bends or hinges forward.
    Biomechanical reference:
        - Upright standing: 0 - 15 degrees
        - Good squat forward lean: 15 - 35 degrees
        - Excessive forward lean (risk of lower-back strain): > 40 - 45 degrees
    """
    sx, sy = _to_point_coords(shoulder)
    hx, hy = _to_point_coords(hip)

    dx = abs(sx - hx)
    dy = abs(sy - hy)

    # dy cannot be 0 if person has height, but guard against div-by-zero
    if dy == 0:
        return 90.0

    radians = np.arctan2(dx, dy)
    angle = radians * 180.0 / np.pi
    return float(np.round(angle, 2))

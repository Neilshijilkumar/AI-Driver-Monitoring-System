import numpy as np

def calculate_ear(eye):
    """
    Eye Aspect Ratio (EAR) calculation.
    eye: list of (x, y) tuples (6 points)
    """

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # safety check
    if eye is None or len(eye) != 6:
        return 1.0  # eyes open (safe fallback)

    p1, p2, p3, p4, p5, p6 = eye

    vertical1 = dist(p2, p6)
    vertical2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    # prevent division crash
    if horizontal == 0:
        return 1.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    # clamp noisy values (important for real-time video)
    return float(np.clip(ear, 0.0, 0.6))


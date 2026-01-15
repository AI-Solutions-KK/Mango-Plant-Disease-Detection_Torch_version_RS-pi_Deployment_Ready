import cv2
import os
import time


def _is_valid_camera(index: int) -> bool:
    """
    Safely test if a camera index is usable.
    - No crash
    - No device lock
    - Releases camera immediately
    """
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)

    if not cap.isOpened():
        cap.release()
        return False

    # Warm-up (important for CSI & USB cams)
    time.sleep(0.2)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False

    # Basic sanity check
    h, w = frame.shape[:2]
    return h > 100 and w > 100


def get_camera_index(max_scan: int = 6) -> int:
    """
    Auto-detect available camera.

    Priority:
    1. Explicit override via ENV (MANGO_CAMERA_INDEX)
    2. CSI camera (usually index 0 on Pi)
    3. USB / Laptop camera (next available)

    Raises:
        RuntimeError if no camera found
    """

    # 1️⃣ Explicit override (optional & safe)
    env_idx = os.environ.get("MANGO_CAMERA_INDEX")
    if env_idx is not None:
        try:
            idx = int(env_idx)
            if _is_valid_camera(idx):
                print(f"📷 Using camera from ENV: index {idx}")
                return idx
        except ValueError:
            pass  # ignore invalid env value

    # 2️⃣ Auto scan
    for idx in range(max_scan):
        if _is_valid_camera(idx):
            print(f"📷 Auto-detected camera at index {idx}")
            return idx

    # 3️⃣ Fail cleanly
    raise RuntimeError(
        "❌ No camera detected.\n"
        "• Check CSI ribbon or USB camera\n"
        "• Verify permissions (/dev/video*)\n"
        "• Try setting MANGO_CAMERA_INDEX env"
    )
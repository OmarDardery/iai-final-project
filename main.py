import os
import shutil
import subprocess
import sys
import time

# Force OpenCV to use X11 backend when Wayland plugin is missing
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# --- Configuration ---
MODEL_PATH = "pose_landmarker_lite.task"

# Cross-platform key names for pyautogui (WASD for Subway Surfers usually)
PYAUTO_KEYS = {"UP": "w", "DOWN": "s", "LEFT": "a", "RIGHT": "d"}

# Cooldowns (in seconds) to prevent game-breaking key spam
COOLDOWNS = {"UP": 0.6, "DOWN": 0.6, "LEFT": 0.3, "RIGHT": 0.3}
last_action_time = {k: 0 for k in PYAUTO_KEYS}


def _init_pyautogui():
    try:
        import pyautogui
    except Exception as exc:
        raise RuntimeError("pyautogui not available") from exc

    pyautogui.FAILSAFE = False

    def _press(action):
        pyautogui.press(PYAUTO_KEYS[action])
        return True

    return _press


def _press_none(action):
    return False

def _has_x11():
    if sys.platform in {"win32", "darwin"}:
        return True
    if not os.environ.get("DISPLAY"):
        return False
    xauth = os.environ.get("XAUTHORITY", "~/.Xauthority")
    return os.path.exists(os.path.expanduser(xauth))

try:
    if _has_x11():
        PRESS_KEY_IMPL = _init_pyautogui()
    else:
        PRESS_KEY_IMPL = _press_none
except Exception:
    PRESS_KEY_IMPL = _press_none

if PRESS_KEY_IMPL is _press_none:
    print("⚠️  Pyautogui requires X11/XWayland on Linux. Please run the script in an X11 environment.")


def press_key(action):
    """Presses key using the configured backend with a logic gate for cooldowns."""
    now = time.time()
    if now - last_action_time[action] > COOLDOWNS[action]:
        if PRESS_KEY_IMPL(action):
            last_action_time[action] = now
            return True
    return False


# --- Initialize MediaPipe Tasks ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

# Calibration variables for Omar's setup
cal_x, cal_y = 0.5, 0.5
is_calibrated = False
cal_frames = 0
MAX_CAL = 30  # Number of frames to average for baseline

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("✅ System Online. Stand still to calibrate center...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # Run Inference
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            nose = landmarks[0]
            # Midpoint between shoulders for horizontal movement
            mid_x = (landmarks[11].x + landmarks[12].x) / 2

            if not is_calibrated:
                cal_x += mid_x
                cal_y += nose.y
                cal_frames += 1
                cv2.putText(
                    frame,
                    f"Calibrating: {cal_frames}/{MAX_CAL}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                if cal_frames >= MAX_CAL:
                    cal_x /= MAX_CAL
                    cal_y /= MAX_CAL
                    is_calibrated = True
            else:
                # --- Gesture Detection Logic ---
                # Threshold math: $Target < Baseline - \Delta$

                # JUMP (Nose moves UP = Y decreases)
                if nose.y < cal_y - 0.15:
                    if press_key("UP"):
                        cv2.putText(
                            frame,
                            "JUMP!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            3,
                        )

                # CROUCH (Nose moves DOWN = Y increases)
                elif nose.y > cal_y + 0.18:
                    if press_key("DOWN"):
                        cv2.putText(
                            frame,
                            "CROUCH!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            3,
                        )

                # LEFT / RIGHT
                if mid_x < cal_x - 0.12:
                    if press_key("LEFT"):
                        cv2.putText(
                            frame,
                            "LEFT",
                            (w - 200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                elif mid_x > cal_x + 0.12:
                    if press_key("RIGHT"):
                        cv2.putText(
                            frame,
                            "RIGHT",
                            (w - 200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )

            # Visual feedback (Nose tracker)
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)

        cv2.imshow("Subway Surfers Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

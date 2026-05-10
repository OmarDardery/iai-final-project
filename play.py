import cv2
import numpy as np
import joblib
from math import hypot
from time import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from playwright.sync_api import sync_playwright

MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_PATH = "pose_model.pkl"
NEUTRAL_MEAN_PATH = "neutral_mean.npy"


class CalibrationManager:
    """
    Collects landmark frames while wrists are held together.
    Emits a neutral reference vector once enough frames accumulate.
    """

    def __init__(self, wrist_threshold=0.12, required_frames=25):
        self.wrist_threshold = wrist_threshold
        self.required_frames = required_frames
        self._buffer = []
        self._count = 0

    def update(self, landmarks):
        """
        Call each frame with the raw MediaPipe landmark list.
        Returns (calibrated: bool, neutral_mean: ndarray | None, progress: float 0-1).
        """
        wrist_dist = hypot(
            landmarks[15].x - landmarks[16].x,
            landmarks[15].y - landmarks[16].y,
        )
        flat = np.array([v for pt in landmarks for v in (pt.x, pt.y, pt.z)])

        if wrist_dist < self.wrist_threshold:
            self._count += 1
            self._buffer.append(flat)
            if self._count >= self.required_frames:
                neutral = np.mean(self._buffer, axis=0)
                self._reset()
                return True, neutral, 1.0
            return False, None, self._count / self.required_frames
        else:
            self._reset()
            return False, None, 0.0

    def _reset(self):
        self._buffer.clear()
        self._count = 0


class PoseFSM:
    """
    Finite state machine for pose-to-keypress control with mirror-position logic.

    Horizontal (lean_left / neutral / lean_right):
      Maps the user's pose to a target lane (0=left, 1=center, 2=right) and
      presses ArrowLeft / ArrowRight as many times as needed to move the
      character from its current lane to the target lane.
      If the character is already in the target lane, nothing is pressed.

    Vertical (jump / crouch):
      Single key press on state entry; no lane tracking involved.

    CONFIRM_FRAMES = 2: a single stray frame is ignored; two consecutive
    identical predictions are enough to commit a transition (~33ms at 30fps).
    """

    CONFIRM_FRAMES = 2

    LANE_MAP = {
        "lean_left":  0,
        "neutral":    1,
        "lean_right": 2,
    }

    VERTICAL_KEYS = {
        "jump":   "ArrowUp",
        "crouch": "ArrowDown",
    }

    def __init__(self, page):
        self.page = page
        self.state = "neutral"
        self.character_lane = 1   # 0=left  1=center  2=right
        self._pending = None
        self._pending_count = 0

    def update(self, prediction):
        """
        Feed the latest model prediction.
        Returns the current committed state (unchanged if no transition fired).
        """
        if prediction == self.state:
            self._pending = None
            self._pending_count = 0
            return self.state

        if prediction == self._pending:
            self._pending_count += 1
        else:
            self._pending = prediction
            self._pending_count = 1

        if self._pending_count >= self.CONFIRM_FRAMES:
            self._enter(self._pending)

        return self.state

    def _enter(self, new_state):
        """Commit transition and issue the appropriate key press(es)."""
        self.state = new_state
        self._pending = None
        self._pending_count = 0

        if new_state in self.LANE_MAP:
            self._move_to_lane(self.LANE_MAP[new_state])
        elif new_state in self.VERTICAL_KEYS:
            self.page.keyboard.press(self.VERTICAL_KEYS[new_state])

    def _move_to_lane(self, target_lane):
        """Press ArrowLeft/Right the exact number of times to reach target_lane."""
        delta = target_lane - self.character_lane
        if delta == 0:
            return
        key = "ArrowRight" if delta > 0 else "ArrowLeft"
        for _ in range(abs(delta)):
            self.page.keyboard.press(key)
        self.character_lane = target_lane

    def reset(self):
        """Return to neutral and center lane without pressing anything."""
        self.state = "neutral"
        self.character_lane = 1
        self._pending = None
        self._pending_count = 0


# ── Setup ─────────────────────────────────────────────────────────────────────

pipeline = joblib.load(POSE_MODEL_PATH)
user_neutral = np.load(NEUTRAL_MEAN_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--start-maximized"])
    page = browser.new_page()
    page.goto("https://poki.com/en/g/subway-surfers")
    print("Browser launched. Join your hands to calibrate and start the game.")

    calibrator = CalibrationManager()
    fsm = PoseFSM(page)
    game_active = False

    running = True
    while running:
        ok, frame = camera.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            if not game_active:
                # ── Phase 1: initial calibration ──────────────────────────────
                done, neutral, progress = calibrator.update(lm)
                cv2.putText(frame, "JOIN HANDS TO CALIBRATE", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if progress > 0:
                    cv2.putText(frame, f"Hold... {int(progress * 100)}%", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if done:
                    user_neutral = neutral
                    game_active = True
                    page.keyboard.press("Space")
                    print("Calibrated! Game started.")

            else:
                # ── Phase 2: live play ─────────────────────────────────────────
                flat = np.array([v for pt in lm for v in (pt.x, pt.y, pt.z)])
                pred = pipeline.predict([flat - user_neutral])[0]
                current_state = fsm.update(pred)

                # Re-calibration: hands joined mid-game resets neutral + FSM
                done, neutral, progress = calibrator.update(lm)
                if done:
                    user_neutral = neutral
                    fsm.reset()
                    print("Re-calibrated!")
                elif progress > 0:
                    cv2.putText(frame, f"Recalibrating... {int(progress * 100)}%",
                                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                cv2.putText(frame, f"Pose: {current_state}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Pose Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

    browser.close()

camera.release()
cv2.destroyAllWindows()

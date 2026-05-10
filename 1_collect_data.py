import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_FOLDER = "omar's workout video"
MODEL_PATH = "pose_landmarker_lite.task"

LABEL_MAP = {
    "omar's workout routine (standing)":     "neutral",
    "omar's workout routine (left)":         "lean_left",
    "omar's workout routine (left2)":        "lean_left",
    "omar's workout routine (right)":        "lean_right",
    "omar's workout routine (right2)":       "lean_right",
    "omar's workout routine (down)":         "crouch",
    "omar_s workout routine(Jump)":          "jump",
    "omar's workout routine (hands_joined)": "hands_joined",
}

feature_cols = [f"f{i}" for i in range(99)]
rows = []

for fname in sorted(os.listdir(VIDEO_FOLDER)):
    stem = os.path.splitext(fname)[0]
    if stem not in LABEL_MAP:
        print(f"Skipping: {fname}")
        continue

    label = LABEL_MAP[stem]
    path = os.path.join(VIDEO_FOLDER, fname)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    clip_start = len(rows)

    # Fresh landmarker per video so timestamps reset to 0 each clip
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    print(f"Processing '{fname}' → {label} ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int((frame_idx / fps) * 1000)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            flat = [v for p in lm for v in (p.x, p.y, p.z)]
            rows.append(flat + [label])
        frame_idx += 1

    cap.release()
    landmarker.close()
    print(f"  {frame_idx} frames → {len(rows) - clip_start} landmark rows extracted")

df = pd.DataFrame(rows, columns=feature_cols + ["label"])

neutral_rows = df[df["label"] == "neutral"][feature_cols].values
neutral_mean = neutral_rows.mean(axis=0)
np.save("neutral_mean.npy", neutral_mean)
print(f"\nneutral_mean.npy saved (computed from {len(neutral_rows)} neutral frames)")

df[feature_cols] = df[feature_cols].values - neutral_mean

df.to_csv("dataset.csv", index=False)
print("dataset.csv saved")
print("\nClass distribution:")
print(df["label"].value_counts().to_string())

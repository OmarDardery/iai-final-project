# Pose-Controlled Game — Project Documentation

## Overview

A computer vision pipeline that lets you control a game (Subway Surfers) using body movements captured via webcam. The original version used hand-coded threshold rules to detect poses. This project replaces that with a **trained ML classifier** that maps MediaPipe body landmarks to game controls.

---

## Background — The Original System

The starting point (`integrated_game.py`) used:

- **MediaPipe Pose** to detect 33 body landmarks per frame
- **Playwright** to control a Chromium browser running the game
- **Hand-coded rules** for control logic:
  - Shoulder midpoint x-position → lean left / right (Arrow keys)
  - Shoulder midpoint y-position vs a calibrated baseline → jump / crouch (Arrow keys)
  - Distance between wrists < 120px → hands joined → start game (Space)

The limitations of the rule-based approach:
- Thresholds are fragile and need manual tuning per person / camera setup
- No ability to learn nuance or generalise across users
- Adding new gestures means rewriting logic by hand

---

## What We're Building

A drop-in replacement for the rule-based logic using a trained **MLP (Multi-Layer Perceptron)** classifier that learns pose → key mapping from video examples.

---

## Why MediaPipe Landmarks (Not Raw Frames)

The model trains on **normalized landmark coordinates**, not pixels. This means:

- **Person-agnostic** — different body sizes, skin tones, and clothing all reduce to the same skeleton geometry
- **Position-agnostic** — standing close or far from the camera, left or right of frame, produces the same normalized values
- **Lighting-agnostic** — MediaPipe handles image variation internally; the model only sees abstract coordinates
- **Resolution-agnostic** — landmarks are 0–1 normalized floats regardless of camera resolution

The model learns geometric relationships between body points (e.g. "hips rise relative to shoulders = jump"), not pixel patterns. Train on yourself, works on anyone.

> Caveat: MediaPipe itself may struggle with extreme camera angles or highly unusual body proportions, but that is a detection issue, not a model issue.

---

## MediaPipe Landmarks

MediaPipe Pose outputs **33 landmarks**, each with `(x, y, z)` coordinates normalized to the frame dimensions, plus a visibility score.

Key landmarks used in this project:

| Index | Name | Used for |
|---|---|---|
| 11 | Left shoulder | Lean detection, jump/crouch baseline |
| 12 | Right shoulder | Lean detection, jump/crouch baseline |
| 15 | Left wrist | Hands-joined detection |
| 16 | Right wrist | Hands-joined detection |
| 23 | Left hip | Jump/crouch detection |
| 24 | Right hip | Jump/crouch detection |

Each frame produces **99 features** (33 landmarks × 3 values each).

---

## Class Labels (6 Poses)

| Label | Gesture | Key |
|---|---|---|
| `neutral` | Standing still | — (no key) |
| `lean_left` | Shoulders shift left | `A` / `ArrowLeft` |
| `lean_right` | Shoulders shift right | `D` / `ArrowRight` |
| `jump` | Hips rise upward | `W` / `ArrowUp` |
| `crouch` | Hips drop downward | `S` / `ArrowDown` |
| `hands_joined` | Wrists close together | `Space` (start game) |

---

## Pipeline

```
Record video clips (one per pose)
        ↓
Extract landmarks per frame → dataset.csv
        ↓
Train MLP classifier → pose_model.pkl
        ↓
Real-time inference: webcam → landmarks → model → key press
```

### Step 1 — Data Collection

Record **6 short video clips** (30–60 seconds each), one per pose label. You can do them one at a time, as many takes as you like. No need to switch poses on the fly.

Tips:
- Keep the same camera position you'll use when playing
- Do the gesture naturally and repeatedly throughout the clip
- For `neutral`, just stand normally — varied micro-movements are fine
- For `jump` / `crouch`, exaggerate slightly so the model sees clear signal

> You have already recorded one video clip. Continue recording the remaining poses.

### Step 2 — Landmark Extraction (`1_collect_data.py`)

Reads every video clip in a folder, runs MediaPipe on each frame, and saves a CSV:

- **Input:** folder of `.mp4` / `.mov` clips named after their label (e.g. `lean_left.mp4`)
- **Output:** `dataset.csv` with 99 feature columns + 1 label column
- Typically 1,000–5,000 rows per class after extraction

### Step 3 — Training (`2_train.py`)

Loads `dataset.csv` and trains a small MLP:

- **Architecture:** 3 dense layers (128 → 64 → 32) + softmax output
- **Framework:** scikit-learn (`MLPClassifier`) — no GPU needed
- **Training time:** under 1 minute on CPU
- **Output:** `pose_model.pkl` (serialized with joblib)
- Includes a classification report and confusion matrix

### Step 4 — Real-time Game Controller (`3_play.py`)

Replaces `integrated_game.py` with the trained model:

- Webcam → MediaPipe → 99 landmarks → `pose_model.pkl` → predicted class → Playwright key press
- Debounce logic to prevent key spamming
- Calibration retained for the hands-joined start gesture as a sanity check

---

## Files to Be Written

| File | Purpose | Status |
|---|---|---|
| `integrated_game.py` | Original rule-based controller | ✅ Done (existing) |
| `1_collect_data.py` | Extract landmarks from video clips to CSV | 🔲 To write |
| `2_train.py` | Train MLP on CSV, save model | 🔲 To write |
| `3_play.py` | Real-time controller using trained model | 🔲 To write |

---

## Dependencies

```
mediapipe
opencv-python
scikit-learn
joblib
playwright
numpy
pandas
```

Install:
```bash
pip install mediapipe opencv-python scikit-learn joblib playwright numpy pandas
playwright install chromium
```

---

## Scope & Constraints

**In scope:**
- Landmark-based pose classification for 6 gesture classes
- Video clip data collection workflow
- MLP training pipeline with evaluation metrics
- Drop-in replacement controller for Subway Surfers via Playwright
- Person-independent and lighting-independent inference

**Out of scope (future work):**
- Temporal / sequence modelling (e.g. LSTM on sequences of frames) for smoother gesture transitions
- Hand gesture recognition (MediaPipe Hands) for finer control
- Multi-player or multi-person detection
- Model retraining / active learning from live gameplay
- Packaging as a standalone app

---

## Current Progress

- [x] Original rule-based controller understood and documented
- [x] Architecture decision made: MediaPipe landmarks → MLP
- [x] 6 class labels defined
- [x] Data collection started (1 video clip recorded)
- [ ] Remaining 5 video clips to record
- [ ] `1_collect_data.py` to write
- [ ] `2_train.py` to write
- [ ] `3_play.py` to write
- [ ] End-to-end test

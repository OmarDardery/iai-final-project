# 🎮 Subway Surfers - AI Pose Controller

Play **Subway Surfers** using your webcam and body movements! This project uses Google's **MediaPipe** for real-time pose detection and a custom **scikit-learn neural network** to classify your poses, then controls the game via **Playwright** without needing system-level input permissions.

Works great on **Linux Wayland** and any platform with a webcam!

---

## 📚 Project Overview

This is a **complete machine learning pipeline** for pose-based game control:

1. **Data Collection** (`generate_data_from_video_clips.ipynb`): Extract pose landmarks from training videos
2. **Model Training** (`train_pose_model.ipynb`): Train an MLP classifier on collected pose data
3. **Live Gameplay** (`play.py`): Use the trained model to detect your poses in real-time and control the game

**Good news:** The model is already trained and saved. You can jump straight to step 3!

---

## 🎯 Game Controls

Stand back so your **upper body (shoulders and wrists)** are visible to the webcam.

| Action | How To Do It |
|--------|-------------|
| **Start Game / Deploy Hoverboard** | Bring your wrists close together (like clapping or crossing wrists) |
| **Move Left** | Lean your shoulders to your left |
| **Move Right** | Lean your shoulders to your right |
| **Jump** | Raise your body/shoulders up |
| **Crouch / Roll** | Dip your body/shoulders down |

---

## 🚀 Quick Start (Already Trained Model)

### Prerequisites

- Python 3.13+
- Webcam
- Chrome/Chromium browser

### Setup & Run

Choose one installation method:

#### Method A: Using `uv` (Recommended)

```bash
# 1. Create and activate virtual environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv add mediapipe opencv-python playwright scikit-learn joblib

# 3. Set up Playwright
uv run playwright install chromium

# 4. Run the game!
uv run play.py
```

#### Method B: Standard pip

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r req.txt

# 3. Set up Playwright
playwright install chromium

# 4. Run the game!
python play.py
```

---

## 📊 Project Structure

```
.
├── play.py                          # Main game controller (run this!)
├── pose_model.pkl                   # Pre-trained MLP classifier ✅
├── neutral_mean.npy                 # Neutral pose reference vector ✅
├── confusion_matrix.png             # Model evaluation visualization
├── dataset.csv                      # Collected pose data
│
├── 1_collect_data.ipynb             # Data collection pipeline
├── train_pose_model.ipynb           # Model training pipeline
│
├── omar's workout video/            # Training videos (folder)
│   ├── omar's workout routine (standing).mp4      → neutral
│   ├── omar's workout routine (left).mp4          → lean_left
│   ├── omar's workout routine (right).mp4         → lean_right
│   ├── omar's workout routine (down).mp4          → crouch
│   └── ... (more videos for each pose)
│
├── pose_landmarker_lite.task        # MediaPipe pose detection model
├── pose_landmarker_heavy.task       # Alternative (heavier) pose model
│
└── README.md                        # You are here!
```

---

## 🔄 Customizing the Model (Advanced)

The pose model is **already trained and working**. But if you want to retrain it (e.g., to detect different poses or improve accuracy), follow these steps:

### Step 1: Replace Videos

Add or replace video files in the `omar's workout video/` folder. The filename determines the pose label:

```
"omar's workout routine (standing)" → detected as "neutral"
"omar's workout routine (left)"     → detected as "lean_left"
"omar's workout routine (right)"    → detected as "lean_right"
"omar's workout routine (down)"     → detected as "crouch"
"omar_s workout routine(Jump)"      → detected as "jump"
"omar's workout routine (hands_joined)" → detected as "hands_joined"
```

**To add a custom pose:**
1. Create a new video: `omar's workout routine (my_pose).mp4`
2. Update the `LABEL_MAP` dictionary in `1_collect_data.ipynb` to include: `"omar's workout routine (my_pose)": "my_pose"`

### Step 2: Collect New Data

Run the data collection notebook to extract pose landmarks from your videos:

```bash
uv run jupyter notebook 1_collect_data.ipynb
# Or execute it in your IDE
```

This generates:
- `dataset.csv` — all extracted pose data (normalized by neutral pose)
- `neutral_mean.npy` — the neutral pose reference vector

### Step 3: Retrain the Model

Run the training notebook to build a new classifier:

```bash
uv run jupyter notebook train_pose_model.ipynb
```

This generates:
- `pose_model.pkl` — the trained MLP classifier
- `confusion_matrix.png` — evaluation visualization

### Step 4: Play with Your Model

Now run `play.py` with your new model:

```bash
uv run play.py
```

---

## 📋 Model Details

**Architecture:**
- Input: 99 features (33 landmarks × 3 coordinates: x, y, z)
- Preprocessing: StandardScaler (normalize to mean=0, std=1)
- Model: Multi-Layer Perceptron (MLP)
  - Hidden layers: [128, 64, 32] neurons
  - Activation: ReLU
  - Output: 6 classes (neutral, lean_left, lean_right, jump, crouch, hands_joined)
- Training: Up to 1000 epochs with adaptive learning rate

**Training Data:**
- ~3,400+ labeled pose frames
- Normalized relative to "neutral" pose (difference from neutral mean)
- 80/20 train-test split

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Camera not opening?** | Ensure your webcam is not in use by another app. Check permissions. |
| **Game not registering inputs?** | Ensure good lighting and that your shoulders/wrists are clearly visible. Stay 1-2 meters from the camera. |
| **OpenCV window fails on Linux (Wayland)** | Run with: `QT_QPA_PLATFORM=xcb uv run play.py` |
| | Or install Qt packages: `sudo dnf install qt6-qtbase-gui qt6-qtwayland` (Fedora) |
| **Browser won't load Subway Surfers?** | Check internet connection. Ensure Playwright Chromium is installed: `uv run playwright install chromium` |
| **Model accuracy is poor?** | Improve lighting, position yourself better for the camera, or retrain with more/better video data. |

---

## 📚 Technologies Used

- **MediaPipe** — Real-time pose detection
- **scikit-learn** — MLP classifier + preprocessing
- **Playwright** — Browser automation & game control
- **OpenCV** — Video capture & preview
- **pandas / NumPy** — Data processing
- **joblib** — Model serialization

---

## 🎓 Learning Resources

- [MediaPipe Pose Documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [scikit-learn MLP Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Playwright Documentation](https://playwright.dev/)
- [pandas DataFrame Guide](https://pandas.pydata.org/docs/)

---

## 📝 License

This project is for educational purposes. Subway Surfers is owned by Kiloo and SYBO Games.

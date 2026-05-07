# Subway Surfers - AI Pose Controller

Play Subway Surfers using your webcam and body movements! This project uses Google's **MediaPipe** for real-time pose detection and **Playwright** to play the browser version of the game automatically without needing system-level input permissions (works great on Linux Wayland!)

## 🎮 How to Play (Controls)

Stand back so your upper body (shoulders and wrists) is visible to the webcam.

- **Start Game / Deploy Hoverboard:** Bring your wrists close together in front of you (like a clap or crossing your wrists).
- **Move Left:** Lean your shoulders to your left.
- **Move Right:** Lean your shoulders to your right.
- **Jump:** Raise your body/shoulders up.
- **Crouch/Roll:** Dip your body/shoulders down.

---

## 🚀 Installation & Running

Choose one of the methods below depending on your python environment manager.

### Method A: Using `uv` (Recommended)

If you use the lightning-fast `uv` package manager:

```bash
# 1. Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 2. Install the required packages
uv add mediapipe opencv-python playwright

# 3. Download the isolated Chromium browser for Playwright
uv run playwright install chromium

# 4. Start the game!
uv run integrated_game.py
```

### Method B: Standard Python (No `uv`)

If you are using standard `pip` and basic virtual environments:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate      # On Linux/macOS
# .venv\Scripts\activate       # On Windows

# 3. Install the required packages
pip install mediapipe opencv-python playwright

# 4. Download the isolated Chromium browser for Playwright
playwright install chromium

# 5. Start the game!
python integrated_game.py
```

## 🛠 Troubleshooting
- **Camera not opening?** Ensure your webcam is not being used by another application.
- **Game not registering inputs?** Make sure you are well-lit and your shoulders and wrists are clearly visible to the camera.

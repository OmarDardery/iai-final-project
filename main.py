import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Point this exactly to the file you just downloaded
model_path = "pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO
)

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("✅ Model loaded successfully from local file!")
except Exception as e:
    print(f"❌ Load failed: {e}")

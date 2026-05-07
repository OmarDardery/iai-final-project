import cv2
import numpy as np
from time import time, sleep
from math import hypot

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Playwright manages its own browser inside Python!
from playwright.sync_api import sync_playwright

import os
# Fix for OpenCV on some Linux setups (Wayland fallback)
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# --- 1. MediaPipe Setup ---
MODEL_PATH = "pose_landmarker_lite.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 2. Playwright Browser Setup & Game Loop ---
print("Launching Python-managed Browser via Playwright...")

# Start Playwright Context
with sync_playwright() as p:
    # Launches an isolated Chromium browser that came with the pip package
    browser = p.chromium.launch(headless=False, args=['--start-maximized'])
    page = browser.new_page()
    
    page.goto("https://poki.com/en/g/subway-surfers")
    print("Browser launched! Please click through any prompts to start the game, then return to the camera.")
    
    # Pose Baselines
    MID_Y = None
    counter = 0
    game_active = False

    # Debounce states
    x_pos_index = 1
    y_pos_index = 1

    print("Ready! Join your hands to initialize standard height and start playing.")

    running = True
    while running:
        # --- Camera & Mediapipe ---
        ok, frame = camera.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        timestamp_ms = int(time() * 1000)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- Pose Logic ---
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            # Check hands joined to start
            if not game_active:
                left_w_x, left_w_y = landmarks[15].x * frame_w, landmarks[15].y * frame_h
                right_w_x, right_w_y = landmarks[16].x * frame_w, landmarks[16].y * frame_h
                dist = hypot(left_w_x - right_w_x, left_w_y - right_w_y)
                
                cv2.putText(frame, "JOIN HANDS TO CALIBRATE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if dist < 120:
                    counter += 1
                    cv2.putText(frame, f"Starting... {counter}/10", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if counter > 10:
                        game_active = True
                        left_s_y = landmarks[11].y * frame_h
                        right_s_y = landmarks[12].y * frame_h
                        MID_Y = (left_s_y + right_s_y) / 2
                        
                        # Playwright native key press
                        page.keyboard.press("Space")
                        print("Game Started!")
                else:
                    counter = 0
                    
            else: # Game is Active
                left_s = landmarks[11]
                right_s = landmarks[12]
                
                cv2.putText(frame, "ACTIVE: LEAN, JUMP, CROUCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Left/Right detection
                mid_x = ((left_s.x + right_s.x) / 2) * frame_w
                
                horizontal_position = 'Center'
                if mid_x < frame_w // 3:
                    horizontal_position = 'Left'
                elif mid_x > 2 * (frame_w // 3):
                    horizontal_position = 'Right'

                # Use native Playwright keyboard API
                if (horizontal_position == 'Left' and x_pos_index != 0) or (horizontal_position == 'Center' and x_pos_index == 2):
                    page.keyboard.press("ArrowLeft")
                    x_pos_index -= 1
                elif (horizontal_position == 'Right' and x_pos_index != 2) or (horizontal_position == 'Center' and x_pos_index == 0):
                    page.keyboard.press("ArrowRight")
                    x_pos_index += 1

                # Jump/Crouch detection
                if MID_Y:
                    actual_mid_y = ((left_s.y + right_s.y) / 2) * frame_h
                    
                    posture = 'Standing'
                    if actual_mid_y < MID_Y - 30: # Tweaked thresholds for strictness
                        posture = 'Jumping'
                    elif actual_mid_y > MID_Y + 50:
                        posture = 'Crouching'

                    if posture == 'Jumping' and y_pos_index == 1:
                        page.keyboard.press("ArrowUp")
                        y_pos_index += 1 
                    elif posture == 'Crouching' and y_pos_index == 1:
                        page.keyboard.press("ArrowDown")
                        y_pos_index -= 1
                    elif posture == 'Standing' and y_pos_index != 1:
                        y_pos_index = 1

        cv2.imshow("Subway Surfers Controller", frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            running = False
            
    browser.close()

camera.release()
cv2.destroyAllWindows()
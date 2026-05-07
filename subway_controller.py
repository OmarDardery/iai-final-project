import cv2
import pyautogui
from time import time
from math import hypot
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
os.environ["QT_QPA_PLATFORM"] = "xcb" # Fix for OpenCV on some Linux setups

try:
    import pyautogui
except Exception as e:
    print(f"⚠️ PyAutoGUI failed to load: {e}")
    class MockPyAutoGUI:
        def press(self, key):
            print(f"👉 [MOCK INPUT]: Pressing '{key}'")
        def click(self, x=None, y=None, button='left'):
            print(f"👉 [MOCK INPUT]: Clicking {button} at ({x}, {y})")
    pyautogui = MockPyAutoGUI()

# --- Initialize MediaPipe Tasks ---
MODEL_PATH = "pose_landmarker_lite.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.PoseLandmarker.create_from_options(options)

def detectPose(image, timestamp_ms, draw=False):
    output_image = image.copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks and draw:
        for landmark in results.pose_landmarks[0]:
            h, w, _ = output_image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(output_image, (x, y), 3, (255, 255, 255), -1)

    return output_image, results

# Rewrite checkHandsJoined, checkLeftRight, and checkJumpCrouch to use the new results object.
# The new results uses an array and indices instead of enums.
# LEFT_WRIST is 15, RIGHT_WRIST is 16
# LEFT_SHOULDER is 11, RIGHT_SHOULDER is 12

def checkHandsJoined(image, results, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    
    landmarks = results.pose_landmarks[0]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    left_wrist_landmark = (left_wrist.x * width, left_wrist.y * height)
    right_wrist_landmark = (right_wrist.x * width, right_wrist.y * height)
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
    if euclidean_distance < 200:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)
    else:
        hand_status = 'Hands Not Joined'
        color = (0, 0, 255)
        
    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    return output_image, hand_status

def checkLeftRight(image, results, draw=False):
    horizontal_position = None
    height, width, _ = image.shape
    output_image = image.copy()
    
    landmarks = results.pose_landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    left_x = int(right_shoulder.x * width)
    right_x = int(left_shoulder.x * width)
    
    if (right_x <= width//2 and left_x <= width//2):
        horizontal_position = 'Left'
    elif (right_x >= width//2 and left_x >= width//2):
        horizontal_position = 'Right'
    elif (right_x >= width//2 and left_x <= width//2):
        horizontal_position = 'Center'
        
    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
    return output_image, horizontal_position

def checkJumpCrouch(image, results, MID_Y=250, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    
    landmarks = results.pose_landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    left_y = int(right_shoulder.y * height)
    right_y = int(left_shoulder.y * height)

    actual_mid_y = abs(right_y + left_y) // 2
    lower_bound = MID_Y-15
    upper_bound = MID_Y+100
    
    if (actual_mid_y < lower_bound):
        posture = 'Jumping'
    elif (actual_mid_y > upper_bound):
        posture = 'Crouching'
    else:
        posture = 'Standing'
        
    if draw:
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
        
    return output_image, posture

camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
time1 = 0
game_started = False   
x_pos_index = 1
y_pos_index = 1
MID_Y = None
counter = 0
num_of_frames = 4

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    timestamp_ms = int(time() * 1000)
    
    frame, results = detectPose(frame, timestamp_ms, draw=game_started)
    
    if results.pose_landmarks:
        if game_started:
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)
            if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                pyautogui.press('left')
                x_pos_index -= 1               
            elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                pyautogui.press('right')
                x_pos_index += 1
        else:
            cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)
        
            if checkHandsJoined(frame, results)[1] == 'Hands Joined':
                counter += 1
                cv2.putText(frame, f'Counter: {counter}/{num_of_frames}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                if counter == num_of_frames:
                    game_started = True
                    left_y = int(results.pose_landmarks[0][12].y * frame_height)
                    right_y = int(results.pose_landmarks[0][11].y * frame_height)
                    MID_Y = abs(right_y + left_y) // 2
                    pyautogui.click(x=1300, y=800, button='left')
                    pyautogui.press('space')
            else:
                counter = 0

        if MID_Y:
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
            if posture == 'Jumping' and y_pos_index == 1:
                pyautogui.press('up')
                y_pos_index += 1 
            elif posture == 'Crouching' and y_pos_index == 1:
                pyautogui.press('down')
                y_pos_index -= 1
            elif posture == 'Standing' and y_pos_index != 1:
                y_pos_index = 1
    else:
        counter = 0
        
    time2 = time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    time1 = time2
               
    cv2.imshow('Subway Surfers with Pose Detection', frame)
    k = cv2.waitKey(1) & 0xFF    
    if(k == 27):
        break
                 
camera_video.release()
cv2.destroyAllWindows()

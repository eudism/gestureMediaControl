import os

import comtypes
import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe and Camera
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Audio Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)


# Fancy terminal output
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


# Map distance to volume level (0 to 1)
def map_distance_to_volume(distance, min_dist=0.01, max_dist=0.15):
    # Clamp distance to a valid range
    distance = max(min_dist, min(distance, max_dist))
    # Normalize to 0-1
    return (distance - min_dist) / (max_dist - min_dist)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_gesture = "None"
    volume_level = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get thumb and index tip positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb and index
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
            )
            volume_level = map_distance_to_volume(distance)

            # Gesture detection for display

            if volume_level < 0.5:
                detected_gesture = "Low Volume"
            elif volume_level < 0.8:
                detected_gesture = "Medium Volume"
            else:
                detected_gesture = "High Volume"

            try:
                # Set the system volume
                volume.SetMasterVolumeLevelScalar(volume_level, None)
            except comtypes.COMError as e:
                print(f"⚠️ Audio device error: {e}")
                print("Continuing without changing volume...")

            # Draw landmarks and gesture info
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Clear terminal and broadcast data
    clear_terminal()
    print("🎥 Gesture-Controlled Media")
    print("=" * 40)
    print(f"Detected Gesture : {detected_gesture}")
    print(f"Volume Level     : {int(volume_level * 100)}%")
    print("=" * 40)
    print("Press 'q' to quit.")

    # Display Frame with gesture and volume info
    cv2.putText(
        frame,
        f"Volume: {int(volume_level * 100)}%",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Gesture: {detected_gesture}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.imshow("Gesture Controlled Media", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

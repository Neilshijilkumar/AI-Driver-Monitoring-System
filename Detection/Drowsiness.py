import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import cv2
import mediapipe as mp
import numpy as np

from ear import calculate_ear
from Detection.object_detection import detect_objects
from Alerts.alarm import play_alarm, stop_alarm
from RAG_system.rag import retrieve_response


# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

# -----------------------------
# Landmarks
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE = 1  # kept simple for demo stability

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# Variables
# -----------------------------
counter = 0
head_counter = 0
frame_count = 0

ear = 0

rag_message = "Driver Safe"

objects = []

phone_detected = False
distraction_detected = False
bottle_detected = False
drowsy_detected = False

alarm_on = False

EAR_THRESHOLD = 0.23


# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    # Reset per frame
    phone_detected = False
    distraction_detected = False
    bottle_detected = False
    drowsy_detected = False

    # -------------------------
    # Object Detection (safe)
    # -------------------------
    if frame_count % 3 == 0:
        try:
            result = detect_objects(frame)

            if result is not None:
                frame, objects, phone_detected, distraction_detected, bottle_detected = result
        except:
            pass

    # -------------------------
    # Face Processing
    # -------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            cv2.putText(frame, "FACE DETECTED",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0), 2)

            left_eye = []
            right_eye = []

            nose = face_landmarks.landmark[NOSE]
            nose_x = int(nose.x * w)

            center_x = w // 2

            # ---------------------
            # Left Eye
            # ---------------------
            for i in LEFT_EYE:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * w), int(lm.y * h)
                left_eye.append((x, y))

            # ---------------------
            # Right Eye
            # ---------------------
            for i in RIGHT_EYE:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * w), int(lm.y * h)
                right_eye.append((x, y))

            # ---------------------
            # EAR Calculation
            # ---------------------
            if len(left_eye) == 6 and len(right_eye) == 6:

                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)

                ear = (ear_left + ear_right) / 2

                cv2.putText(frame, f"EAR: {ear:.2f}",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255),
                            2)

                # ---------------------
                # Eye closure logic
                # ---------------------
                if ear < EAR_THRESHOLD:
                    counter += 1
                    cv2.putText(frame, "EYES CLOSED",
                                (250, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255), 2)
                else:
                    counter = 0

                # ---------------------
                # Drowsiness trigger
                # ---------------------
                if counter > 15:
                    drowsy_detected = True
                    cv2.putText(frame, "DROWSY!",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255), 3)

                # ---------------------
                # Head distraction
                # ---------------------
                if abs(nose_x - center_x) > 90:
                    head_counter += 1
                else:
                    head_counter = 0

                if head_counter > 10:
                    distraction_detected = True
                    cv2.putText(frame, "HEAD DISTRACTION",
                                (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255), 3)

    else:
        cv2.putText(frame, "NO FACE DETECTED",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255), 2)

        counter = 0
        head_counter = 0

    # -------------------------
    # Alert System
    # -------------------------
    current_alert = ""

    if phone_detected:
        current_alert = "phone"
        rag_message = retrieve_response("mobile phone driving danger")

    elif bottle_detected:
        current_alert = "bottle"
        rag_message = retrieve_response("drinking while driving danger")

    elif drowsy_detected:
        current_alert = "drowsy"
        rag_message = retrieve_response("driver sleepy driving danger")

    elif distraction_detected:
        current_alert = "distraction"
        rag_message = retrieve_response("driver distraction danger")

    else:
        rag_message = "Driver Focused and Safe"

    # -------------------------
    # Alarm control
    # -------------------------
    if current_alert != "" and not alarm_on:
        play_alarm()
        alarm_on = True

    elif current_alert == "" and alarm_on:
        stop_alarm()
        alarm_on = False

    # -------------------------
    # Debug info
    # -------------------------
    cv2.putText(frame, f"Objects: {objects}",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0), 2)

    cv2.putText(frame, f"Phone: {phone_detected}",
                (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255), 2)

    cv2.putText(frame, f"Bottle: {bottle_detected}",
                (20, 410),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255), 2)

    cv2.putText(frame, f"AI: {rag_message}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0), 2)

    # -------------------------
    # Show
    # -------------------------
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

from Detection.object_detection import detect_objects

from Alerts.alarm import (
    play_alarm,
    stop_alarm
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for better performance
    frame = cv2.resize(frame, (640, 480))

    # ---------------------------------
    # OBJECT DETECTION
    # ---------------------------------

    frame, objects, phone_detected, distraction_detected = detect_objects(frame)

    # ---------------------------------
    # SHOW DETECTED OBJECTS
    # ---------------------------------

    cv2.putText(
        frame,
        f"Objects: {objects}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ---------------------------------
    # PHONE DETECTION
    # ---------------------------------

    if phone_detected:

        cv2.putText(
            frame,
            "PHONE DETECTED!",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ---------------------------------
    # DISTRACTION ALERT
    # ---------------------------------

    if distraction_detected:

        cv2.putText(
            frame,
            "FOCUS ON ROAD!",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Play alarm
        play_alarm()

    else:

        # Stop alarm
        stop_alarm()

    # ---------------------------------
    # SHOW WINDOW
    # ---------------------------------

    cv2.imshow(
        "YOLO Driver Monitoring",
        frame
    )

    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()
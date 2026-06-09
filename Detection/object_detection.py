from ultralytics import YOLO
import cv2

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8n.pt")

confidence_threshold = 0.3  # slightly higher = fewer false alarms

# -------------------------------
# FUNCTION
# -------------------------------
def detect_objects(frame):

    detected_objects = []

    phone_detected = False
    distraction_detected = False
    bottle_detected = False

    results = model.predict(
        frame,
        conf=confidence_threshold,
        imgsz=416,
        verbose=False
    )

    if not results:
        return frame, [], False, False, False

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            cls = int(box.cls[0])
            class_name = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -------------------------------
            # PHONE (STRICT CHECK)
            # -------------------------------
            if class_name == "cell phone":

                phone_detected = True
                distraction_detected = True
                detected_objects.append("phone")

                color = (0, 0, 255)

            # -------------------------------
            # BOTTLE (LESS SENSITIVE)
            # -------------------------------
            elif class_name == "bottle":

                bottle_detected = True
                detected_objects.append("bottle")

                color = (255, 0, 255)

            # -------------------------------
            # CUP (OPTIONAL DISTRACTION)
            # -------------------------------
            elif class_name == "cup":

                detected_objects.append("cup")
                color = (0, 255, 255)

            else:
                continue

            # -------------------------------
            # DRAW BOX
            # -------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return (
        frame,
        detected_objects,
        phone_detected,
        distraction_detected,
        bottle_detected
    )   

import streamlit as st
import cv2
import time
import mediapipe as mp

from Detection.ear import calculate_ear
from Detection.object_detection import detect_objects
from Alerts.alarm import play_alarm, stop_alarm
from RAG_system.rag import retrieve_response

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Driver Monitoring System",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0F172A,
        #111827,
        #1E293B
    );
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#38BDF8;
}

.sub-title{
    text-align:center;
    color:#CBD5E1;
    font-size:18px;
    margin-bottom:20px;
}

.stButton > button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-weight:bold;
}

[data-testid="metric-container"]{
    background:#1E293B;
    border-radius:15px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🚗 AI Driver Monitoring System
</div>

<div class="sub-title">
Real-Time Driver Safety Monitoring using
YOLOv8 • MediaPipe • RAG
</div>
""", unsafe_allow_html=True)
with st.sidebar:

    st.title("🚗 Driver Monitoring")

    st.markdown("""
    ### Features

    ✅ Drowsiness Detection

    ✅ Mobile Phone Detection

    ✅ Drinking Detection

    ✅ Head Side Distraction

    ✅ Head Down Detection

    ✅ RAG Safety Assistant
    """)

    st.success("System Ready")

    st.info(
        "YOLOv8 + MediaPipe + FAISS"
    )
# ---------------- SESSION STATE ----------------
defaults = {
    "run": False,
    "phone_counter": 0,
    "bottle_counter": 0,
    "head_side_counter": 0,
    "head_down_counter": 0,
    "eye_start_time": None,
    "alarm_on": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- UI ----------------
c1, c2 = st.columns(2)

with c1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True

with c2:
    if st.button("🛑 Stop Camera"):
        st.session_state.run = False

metric1, metric2, metric3, metric4 = st.columns(4)

ear_metric = metric1.empty()
status_metric = metric2.empty()
object_metric = metric3.empty()
score_metric = metric4.empty()

frame_placeholder = st.empty()
status_placeholder = st.empty()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# ---------------- MEDIAPIPE ----------------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# ---------------- LOOP ----------------
while st.session_state.run:

    ret, frame = cap.read()

    if not ret:
        status_placeholder.error("Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    phone_detected = False
    bottle_detected = False

    drowsy = False
    head_side = False
    head_down = False

    # ---------------- OBJECT DETECTION ----------------
    try:
        result = detect_objects(frame)

        if result:
            (
                frame,
                _,
                phone_detected,
                _,
                bottle_detected
            ) = result

    except Exception as e:
        print("Detection Error:", e)

    # ---------------- COUNTERS ----------------
    if phone_detected:
        st.session_state.phone_counter += 1
    else:
        st.session_state.phone_counter = max(
            0,
            st.session_state.phone_counter - 1
        )

    if bottle_detected:
        st.session_state.bottle_counter += 1
    else:
        st.session_state.bottle_counter = max(
            0,
            st.session_state.bottle_counter - 1
        )

    phone_alert = st.session_state.phone_counter >= 2
    bottle_alert = st.session_state.bottle_counter >= 2

    # ---------------- FACE ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    ear = 0.0
    if results.multi_face_landmarks:

        lm = results.multi_face_landmarks[0].landmark

        left_eye = [
            (lm[i].x * w, lm[i].y * h)
            for i in LEFT_EYE
        ]

        right_eye = [
            (lm[i].x * w, lm[i].y * h)
            for i in RIGHT_EYE
        ]

        try:
            ear = (
                calculate_ear(left_eye) +
                calculate_ear(right_eye)
            ) / 2

        except:
            ear = 1.0

        cv2.putText(
            frame,
            f"EAR:{ear:.2f}",
            (20,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        # DROWSINESS
        if ear < 0.21:

            if st.session_state.eye_start_time is None:
                st.session_state.eye_start_time = time.time()

            elif (
                time.time()
                - st.session_state.eye_start_time
            ) > 1.5:

                drowsy = True

        else:
            st.session_state.eye_start_time = None

        # SIDEWAYS
        left_cheek = lm[234].x
        right_cheek = lm[454].x
        nose_x = lm[1].x

        face_center = (
            left_cheek + right_cheek
        ) / 2

        side_value = abs(
            nose_x - face_center
        )

        if side_value > 0.06:
            st.session_state.head_side_counter += 1

        else:
            st.session_state.head_side_counter = max(
                0,
                st.session_state.head_side_counter - 1
            )

        head_side = (
            st.session_state.head_side_counter >= 6
        )

        # HEAD DOWN
        forehead_y = lm[10].y
        nose_y = lm[1].y
        chin_y = lm[152].y

        face_height = (
            chin_y - forehead_y
        )

        if face_height > 0:

            ratio = (
                nose_y - forehead_y
            ) / face_height

            if ratio > 0.60:
                st.session_state.head_down_counter += 1

            else:
                st.session_state.head_down_counter = max(
                    0,
                    st.session_state.head_down_counter - 1
                )

        head_down = (
            st.session_state.head_down_counter >= 6
        )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    # ---------------- MESSAGE ----------------
    if drowsy:
        msg = retrieve_response("drowsy driving")
        frame_msg = "DROWSINESS ALERT"

    elif phone_alert:
        msg = retrieve_response("mobile phone distraction")
        frame_msg = "PHONE DETECTED"

    elif bottle_alert:
        msg = retrieve_response("drinking while driving")
        frame_msg = "DRINKING DETECTED"

    elif head_down:
        msg = retrieve_response("head down posture")
        frame_msg = "HEAD DOWN ALERT"

    elif head_side:
        msg = retrieve_response("looking sideways")
        frame_msg = "LOOKING SIDEWAYS"

    else:
        msg = "Driver is attentive and focused."
        frame_msg = "SAFE DRIVING"

    danger = (
        drowsy or
        phone_alert or
        bottle_alert or
        head_down or
        head_side
    )

    # ---------------- ALARM ----------------
    if danger and not st.session_state.alarm_on:
        play_alarm()
        st.session_state.alarm_on = True

    elif not danger and st.session_state.alarm_on:
        stop_alarm()
        st.session_state.alarm_on = False

    # ---------------- DISPLAY ----------------
    color = (
        (0,0,255)
        if danger
        else (0,255,0)
    )

    cv2.putText(
        frame,
        frame_msg,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    # ---------------- DRIVER SCORE ----------------

    driver_score = 100

    if drowsy:
        driver_score -= 30

    if phone_alert:
        driver_score -= 25

    if bottle_alert:
        driver_score -= 15

    if head_side:
        driver_score -= 15

    if head_down:
        driver_score -= 15

    driver_score = max(0, driver_score)

    # ---------------- DASHBOARD METRICS ----------------

    ear_metric.metric(
        "EAR",
        f"{ear:.2f}"
    )

    status_metric.metric(
        "Status",
        frame_msg
    )

    object_metric.metric(
        "Objects",
        int(phone_detected) +
        int(bottle_detected)
    )

    score_metric.metric(
        "Driver Score",
        f"{driver_score}/100"
    )
    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        width=700
    )

    if danger:
        status_placeholder.error(msg)
    else:
        status_placeholder.success(msg)

    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()


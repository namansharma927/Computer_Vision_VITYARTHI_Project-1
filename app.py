import streamlit as st
import cv2
import time
import numpy as np

# Import the refactored modules
import config
from detectors import EyeDetector, PhoneDetector, EmotionDetector
from core import RiskEngine, AlertSystem
from annotator import annotate

st.set_page_config(page_title="Driver Safety Monitor", page_icon="🛡️", layout="wide")

st.title("🛡️ Driver Safety Monitoring System")
st.markdown("Real-time monitoring for **Drowsiness**, **Phone Distraction**, and **Emotional state**.")

# ── Sidebar Configuration ──
st.sidebar.header("Configuration")
camera_index = st.sidebar.number_input("Camera Index", min_value=0, value=0, step=1)

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

def start_camera():
    st.session_state.run_camera = True

def stop_camera():
    st.session_state.run_camera = False

col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Start System", on_click=start_camera)
with col2:
    st.button("Stop System", on_click=stop_camera)

# ── Dashboard Layout ──
video_col, stats_col = st.columns([2, 1])

with video_col:
    st.markdown("### Live Feed")
    frame_placeholder = st.empty()

with stats_col:
    st.markdown("### System Telemetry")
    risk_metric = st.empty()
    score_metric = st.empty()
    fps_metric = st.empty()
    
    st.markdown("### Detections")
    emotion_metric = st.empty()
    phone_metric = st.empty()
    drowsy_metric = st.empty()

    st.markdown("### Alerts")
    alert_box = st.empty()

# ── Camera Loop ──
if st.session_state.run_camera:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Cannot open camera {camera_index}. Please check connection.")
        st.session_state.run_camera = False
        st.rerun()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Initialize modules
    eye_det = EyeDetector()
    phone_det = PhoneDetector()
    emo_det = EmotionDetector()
    risk_eng = RiskEngine()
    alert_sys = AlertSystem()

    fps_counter = 0
    fps_timer = time.time()
    fps = 0.0
    
    recent_alerts = []

    while st.session_state.run_camera:
        ret, frame_bgr = cap.read()
        if not ret:
            st.error("Frame read failed.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detectors
        eye_state = eye_det.process(frame_rgb, frame_bgr.shape)
        phone_state = phone_det.process(frame_bgr)
        emo_state = emo_det.process(frame_bgr)

        if "distraction_score" not in phone_state:
            phone_state["distraction_score"] = 0.0

        # Risk Engine
        risk_state = risk_eng.update(
            drowsiness_score=eye_state.get("drowsiness_score", 0.0),
            distraction_score=phone_state.get("distraction_score", 0.0),
            emotion_score=emo_state.get("emotion_score", 0.0),
        )

        # Alerts
        new_alerts = alert_sys.evaluate(eye_state, phone_state, emo_state, risk_state)
        for al in new_alerts:
            recent_alerts.append(f"[{time.strftime('%H:%M:%S')}] {al}")
        if len(recent_alerts) > 5:
            recent_alerts = recent_alerts[-5:]

        # Annotation
        left_pts, right_pts = eye_det.get_eye_landmarks(frame_rgb, frame_bgr.shape)
        annotated_rgb = annotate(frame_rgb, eye_state, phone_state, emo_state, risk_state, fps, left_pts, right_pts)

        frame_bgr_draw = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr_draw = phone_det.draw_boxes(frame_bgr_draw)
        annotated_rgb = cv2.cvtColor(frame_bgr_draw, cv2.COLOR_BGR2RGB)

        # FPS Calculation
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        # UI Updates
        frame_placeholder.image(annotated_rgb, use_column_width=True)
        
        # Color Risk
        r_level = risk_state.get('risk_level', 'SAFE')
        r_color = "green" if r_level == "LOW" else "orange" if r_level == "MEDIUM" else "red"
        
        risk_metric.markdown(f"**Overall Risk**: <span style='color:{r_color}; font-size:20px;'>{r_level}</span>", unsafe_allow_html=True)
        score_metric.markdown(f"**Risk Score**: {risk_state.get('smooth_score', 0):.1f}")
        fps_metric.markdown(f"**FPS**: {fps:.0f}")

        emotion_metric.markdown(f"**Emotion**: {emo_state.get('emotion', 'Unknown')}")
        phone_metric.markdown(f"**Phone Detected**: {'🛑 YES' if phone_state.get('phone_detected') else '✅ NO'}")
        drowsy_metric.markdown(f"**Drowsy**: {'🛑 YES' if eye_state.get('is_drowsy') else '✅ NO'}")

        alert_text = "\n".join(recent_alerts) if recent_alerts else "No active alerts."
        alert_box.code(alert_text, language="text")

    cap.release()
else:
    st.info("Click 'Start System' in the sidebar to begin monitoring.")

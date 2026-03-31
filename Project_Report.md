# VisionGuard: AI-Powered Driver Monitoring System
### BYOP Capstone Project Report

**Date**: March 29, 2026  
**Author**: Hardik  
**Repo**: [https://github.com/hardik0903/CV_proj.git](https://github.com/hardik0903/CV_proj.git)

---

## 1. Problem Statement
Driver drowsiness and distractions—such as mobile phone usage and emotional distress—are major causes of road accidents worldwide. Thousands of lives are lost every year due to momentary lapses in attention or falling asleep at the wheel. The goal of this project is to develop a robust, real-time monitoring system that can detect signs of fatigue, phone distraction, and emotional state using standard webcam hardware.

## 2. Approach & Methodology

### Core Technologies
- **Computer Vision Framework**: OpenCV bounds the real-time processing and pipeline execution.
- **MediaPipe (Face Mesh)**: Used for high-speed facial landmark extraction to detect eye aspect ratios (EAR).
- **YOLOv8 (Ultralytics)**: Evaluates the frame to identify cell phones, addressing active physical distraction.
- **DeepFace**: Evaluates emotional states sequentially.
- **Streamlit**: Orchestrates the backend AI loops with a modern, dynamic web-based dashboard perfectly suited for telemetry.

### Detection Algorithms & Risk Engine
1. **Drowsiness (EAR)**: The Eye Aspect Ratio (EAR) measures the distance between vertical and horizontal eye landmarks. A value below the critical threshold for a sustained duration flags drowsiness.
2. **Distraction (YOLO)**: The system feeds the frame into YOLOv8 to search for the `cell phone` class. If detected with high confidence in the driver's frame, a distraction flag is raised.
3. **Emotion Engine**: DeepFace determines if the user is exhibiting stress, anger, or sleepiness contextually.
4. **Risk State**: A mathematical `RiskEngine` weights the Drowsiness, Distraction, and Emotion scores into a singular moving `smooth_score` from 0-100, which outputs an overall Safety Threat Level (LOW, MEDIUM, HIGH).

## 3. Key Decisions
- **Streamlit vs Tkinter**: We adapted an upstream modular concept (initially designed for Tkinter) into Streamlit. The single execution loop requires careful session state management (`st.empty()`) to prevent the dashboard from locking up during inference.
- **Microservice Architecture**: Utilizing isolated classes (`EyeDetector`, `PhoneDetector`, `EmotionDetector`, and `RiskEngine`) guarantees that if YOLO inference lags, the EAR and Risk logic scales gracefully.

## 4. Challenges Faced
- **Dependency Conflicts**: Combining YOLOv8 (PyTorch), DeepFace (TensorFlow/Keras), and MediaPipe onto a single thread required precise dependency resolution (`protobuf<4` limits and XNNPACK delegation).
- **Real-Time Streamlit Rendering**: Standard cv2 infinite loops break Streamlit's web rendering. This was mitigated by mapping frame updates to an `st.image` placeholder object without calling `st.rerun()`.

## 5. Learnings
- **Multi-Model Pipelines**: Actively routing one RGB frame into three entirely distinct Machine Learning frameworks taught me how deep learning orchestration works in edge-deployment scenarios.
- **Mathematical CV**: Utilizing bounding boxes, Euclidean distances, and thresholding algorithms proved highly effective even against modern neural-net constraints.

## 6. Conclusion
VisionGuard proves that standard hardware combined with modern pre-trained deep learning frameworks (YOLO, MediaPipe, DeepFace) can be leveraged to build life-saving safety systems. Future iterations would encompass night-vision IR tracking and distributed edge-processing.

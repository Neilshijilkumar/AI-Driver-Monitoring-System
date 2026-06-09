# 🚗 AI Driver Monitoring System

## 📌 Overview

AI Driver Monitoring System is a real-time driver safety application developed using Computer Vision, Deep Learning, and Retrieval-Augmented Generation (RAG). The system continuously monitors driver behavior through a webcam and detects unsafe driving conditions such as drowsiness, mobile phone usage, drinking while driving, head-down posture, and driver distraction.

The system provides real-time visual alerts, audio warnings, driver safety scoring, and AI-powered safety guidance to improve road safety and reduce accident risks.

---

## ✨ Features

* 👁️ Drowsiness Detection using Eye Aspect Ratio (EAR)
* 📱 Mobile Phone Usage Detection using YOLOv8
* 🥤 Drinking Detection using YOLOv8
* ↔️ Head Side Distraction Detection using MediaPipe Face Mesh
* ⬇️ Head Down Posture Detection
* 🔊 Real-Time Audio Alert System
* 🤖 RAG-Based Safety Assistant
* 📊 Driver Safety Score Dashboard
* 🎨 Interactive Streamlit User Interface

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Computer Vision & Deep Learning

* OpenCV
* YOLOv8
* MediaPipe Face Mesh

### Retrieval-Augmented Generation (RAG)

* FAISS
* Sentence Transformers

### Frontend Dashboard

* Streamlit

### Supporting Libraries

* NumPy
* PyTorch

---

## 🏗️ System Architecture

![System Architecture](Architecture.png)

---

## 🔄 Workflow

1. Webcam captures live video frames.
2. YOLOv8 detects mobile phone and drinking behavior.
3. MediaPipe Face Mesh extracts facial landmarks.
4. Eye Aspect Ratio (EAR) is calculated for drowsiness detection.
5. Head orientation analysis detects distraction and head-down posture.
6. Alert Management Module triggers audio warnings.
7. FAISS-based RAG system retrieves safety recommendations.
8. Streamlit dashboard displays real-time monitoring metrics and driver safety score.

---

## 📁 Project Structure

```text
AI-Driver-Monitoring-System
│
├── Alerts/
│   ├── __init__.py
│   └── alarm.py
│
├── Detection/
│   ├── Drowsiness.py
│   ├── ear.py
│   └── object_detection.py
│
├── RAG_system/
│   ├── knowledge_base.txt
│   └── rag.py
│
├── app.py
├── requirements.txt
├── Architecture.png
└── README.md
```

## 🚀 Installation

```bash
git clone https://github.com/Neilshijilkumar/AI-Driver-Monitoring-System.git

cd AI-Driver-Monitoring-System

pip install -r requirements.txt

streamlit run app.py
```

## 📈 Future Enhancements

* Seatbelt Detection
* Driver Identity Verification
* Emotion Recognition
* Cloud Deployment
* GPS-Based Driver Analytics
* Advanced Driver Behavior Analysis

---

## 🎯 Project Outcome

The system successfully monitors driver attentiveness in real time and provides proactive alerts for unsafe driving behaviors. By integrating Computer Vision, Deep Learning, and RAG, the project demonstrates how AI can enhance road safety through intelligent driver monitoring.

---

## 👨‍💻 Author

**Neil Shijil Kumar**

GitHub: https://github.com/Neilshijilkumar

# 🚗 AI Driver Monitoring System

Real-time AI-powered driver monitoring system using Computer Vision, Deep Learning, and RAG to detect unsafe driving behavior and provide safety alerts.

## Features

- Drowsiness Detection using EAR (Eye Aspect Ratio)
- Mobile Phone Detection using YOLOv8
- Drinking Detection
- Head Side Distraction Detection
- Head Down Detection
- Audio Alert System
- RAG-Based Safety Assistant
- Driver Safety Score Dashboard
- Real-Time Streamlit Interface

## Tech Stack

- Python
- OpenCV
- YOLOv8
- MediaPipe Face Mesh
- Streamlit
- FAISS
- Sentence Transformers

## Project Architecture

Webcam Feed
→ MediaPipe Face Mesh
→ Driver Behavior Analysis

Webcam Feed
→ YOLOv8 Object Detection
→ Phone & Drinking Detection

→ Alert Management
→ FAISS Knowledge Base
→ RAG Safety Assistant
→ Streamlit Dashboard

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
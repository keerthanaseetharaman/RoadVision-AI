# 🚧 RoadVision AI

## Intelligent Road Damage Detection and Mapping System

RoadVision AI is an AI-powered road damage detection system that uses YOLO-based object detection to identify common road damages from road images.

The system detects road damage, estimates severity, calculates affected area, stores detection results in a database, displays damage locations on an interactive map, and provides analytics through a Streamlit dashboard.

---

## 🎯 Project Objective

The main objective of RoadVision AI is to develop an intelligent system that can assist in identifying and analyzing road surface damages from images.

The system focuses on:

- Automated road damage detection
- Damage classification
- Confidence score analysis
- Severity estimation
- Affected area calculation
- Location-based damage mapping
- Detection history
- Data analytics
- Report generation

---

## 🚀 Features

### 🔍 Road Damage Detection

Upload a road image and the YOLO model identifies road damage using bounding boxes.

Supported damage types:

- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Pothole

### 📊 Damage Analysis

For every detected damage, the system provides:

- Damage type
- Detection confidence
- Estimated severity
- Affected area percentage

### 📍 Location Mapping

Users can manually enter latitude and longitude coordinates.

The detected road damage is displayed on an interactive map using Folium.

### 🗺️ Historical Damage Map

Previously saved road damage locations can be viewed together on an interactive map.

### 📈 Dashboard

The dashboard provides:

- Total detections
- Total potholes
- Total cracks
- High severity detections
- Average confidence
- Damage type distribution
- Severity distribution

### 💾 Database

Detection results are stored using SQLite.

Stored information includes:

- Damage type
- Confidence
- Severity
- Affected area
- Latitude
- Longitude
- Detection timestamp

### 📄 Report Generation

Detection results can be downloaded as a CSV report.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLO / Ultralytics | Road damage detection |
| OpenCV | Image processing |
| Streamlit | Web application |
| Pandas | Data processing |
| NumPy | Numerical operations |
| SQLite | Database |
| Folium | Interactive maps |
| Plotly | Data visualization |
| PyTorch | Deep learning framework |

---

## 🧠 Machine Learning Model

RoadVision AI uses a YOLO-based object detection model trained for road damage detection.

### Detection Classes

| Class | Damage Type |
|---|---|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Pothole |

---

## 📂 Project Structure

```text
RoadVision-AI/
│
├── app/
│   └── app.py
│
├── dataset/
│
├── models/
│   └── road_damage_best.pt
│
├── outputs/
│
├── data.yaml
├── damage_data.yaml
├── database.py
├── requirements.txt
├── README.md
└── .gitignore
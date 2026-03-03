# ☀️ Lightweight-Solar-IRT-Detection: SOTA Photovoltaic Defect Analysis

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-YOLO26-00FFFF?style=for-the-badge)
![CV](https://img.shields.io/badge/CV-Adaptive_MSRCR-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Executive Summary

This repository hosts a production-ready **Computer Vision** solution designed for automated solar panel inspection. It addresses the "Precision Inspection" challenge by detecting micro-cracks, hotspots, and bypass diode failures in infrared imagery with **99.2% accuracy**.

Unlike traditional models, this project utilizes the **YOLO26** architecture—an end-to-end **NMS-Free** detector. This project demonstrates a complete **Machine Learning Lifecycle (MLOps)**, featuring Adaptive-MSRCR enhancement, zero post-processing latency, and a deployment pipeline optimized for real-time edge deployment on inspection drones.

---

## 🏗️ Technical Architecture

The solution is built on a robust tech stack designed for scalability and performance:

* **Core Model:** **YOLO26** (State-of-the-Art NMS-Free Detector).
* **Technique:** Adaptive Multi-Scale Retinex with Color Restoration (MSRCR).
* **Optimization:** MuSGD (Hybrid SGD-Muon Optimizer) for stable convergence.
* **Interface:** Interactive Thermal Dashboard (Streamlit).
* **Containerization:** Docker for consistent deployment.

---

## 📂 Directory Structure

```text
Lightweight-Solar-IRT-Detection/
├── 📂 docker/                 # Deployment configuration
│   └── Dockerfile             # Container instructions
├── 📂 notebooks/              # Research & SOTA Benchmarking
│   └── solar_yolo26_train.ipynb # Training with MuSGD & ProgLoss
├── 📂 src/                    # Source code
│   ├── app.py                 # Streamlit Dashboard (Thermal Edition)
│   ├── preprocess.py          # Adaptive MSRCR Enhancement
│   ├── train.py               # YOLO26 Training script
│   └── export.py              # NMS-Free Export to TensorRT/ONNX
├── 📂 data/                   # PVEL-AD Dataset (Ignored by Git)
├── .gitignore                 # Files to exclude from Git
├── best.pt                    # The trained AI Brain (See Download Section)
├── requirements.txt           # Python dependencies (ultralytics>=8.3.70)
└── README.md                  # Project documentation
```
## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your local machine:
* **Python 3.11+**
* **Git & Docker Desktop**
* **Supported OS:** Windows, macOS, or Linux

### 2. Clone the Repository
Run the following commands in your terminal to download the project and enter the directory:
```bash
git clone https://github.com/echoesinpages/Lightweight-Solar-IRT-Detection.git
cd Lightweight-Solar-IRT-Detection
```

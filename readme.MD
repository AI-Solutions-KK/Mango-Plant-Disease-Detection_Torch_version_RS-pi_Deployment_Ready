# 🥭 Mango Plant Disease Detection (Raspberry Pi Ready)

Offline AI system to detect mango leaf diseases using camera or image upload.  
Designed for Raspberry Pi deployment with minimal memory usage.

---

## 🔧 Tech Stack
- Python 3.10+
- Flask (Web UI + API)
- ONNX Runtime (CNN feature extractor)
- Scikit-learn SVM (trained classifier)
- OpenCV (camera capture)
- Works on PC & Raspberry Pi

---

## 📁 Project Structure

```text
flask_deploy/
│
├── server.py                 # Flask server (UI + API)
├── requirements.txt          # Python dependencies
├── efficientnet_b4.onnx      # ONNX CNN feature extractor
├── readme.md                 # Project documentation
│
├── uploads/
│   └── current.jpg           # Latest uploaded/captured image (auto-replaced)
│
├── app/
│   └── inference.py          # ONNX + SVM inference logic
│
├── embeddings_cache/
│   ├── svc_model.pkl         # Trained SVM classifier
│   ├── classes.npy           # Disease labels
│   └── centroids.npy         # Class centroids (similarity check)
│
├── templates/
│   └── index.html            # Web UI (Upload / Capture / Diagnose)
│
├── static/
│   └── img.png               # Optional UI assets
│
├── pi_test/                  # Raspberry Pi camera test scripts (optional)
└── .venv/                    # Virtual environment (local only)


## 🚀 Setup (PC or Raspberry Pi)

### 1️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate        # Linux / Raspberry Pi
.venv\Scripts\activate         # Windows

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run Application
python server.py

Open browser:
http://localhost:5000

For mobile / tablet:
http://<PI-IP>:5000

---

## 📸 User Flow
1. Upload Image OR Capture using camera  
2. Click Diagnose  
3. View disease, confidence, cause, treatment, prevention  
4. Click Clean to process next leaf  

---

## 🌱 Supported Diseases
- Anthracnose
- Bacterial Canker
- Powdery Mildew
- Die Back
- Sooty Mould
- Gall Midge
- Cutting Weevil
- Healthy

---

## 🧠 Model Details
- Feature extractor: EfficientNet (ONNX)
- Classifier: Pre-trained SVM (99% accuracy)
- Inference only on Pi (no training)

---

## 🖥 Hardware Recommendation
- Raspberry Pi 4 (4GB / 8GB)
- Pi Camera v2 or USB webcam
- 32GB+ SD card
- Heat sink / cooling fan

---

## 🔌 Camera Support
- USB webcam (PC / Pi)
- Pi Camera (CSI)
- Auto-detected, no config change

---

## 🔒 Offline Usage
- Fully offline after setup
- No internet required in field

---

## ✅ Deployment Status
✔ Tested  
✔ Stable  
✔ Production Ready  

---


Author: AI-Solution - Karan-kk

# 🍓 Raspberry Pi Deployment Guidelines  
(Mango Plant Disease Detection System)

This document provides step-by-step, practical guidelines to deploy the Mango Plant Disease Detection system on a Raspberry Pi for offline field usage.

---

## 1️⃣ Recommended Raspberry Pi Setup

### Hardware
- Raspberry Pi 4 (4GB or 8GB)
- 32GB or higher micro-SD card
- Pi Camera v2 OR USB webcam
- Heat sink + cooling fan (recommended)
- Power supply (5V, 3A)

---

## 2️⃣ Operating System

### Recommended OS
- Raspberry Pi OS (64-bit)
- Lite version preferred

Enable required interfaces:
```bash
sudo raspi-config
```
Enable Camera and SSH.

---

## 3️⃣ System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv libatlas-base-dev libjpeg-dev libopenjp2-7 libtiff5 libopencv-dev
```

---

## 4️⃣ Project Setup

Copy project to Pi:
```bash
scp -r flask_deploy pi@<PI-IP>:/home/pi/
```

Navigate:
```bash
cd flask_deploy
```

---

## 5️⃣ Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 6️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 7️⃣ Camera Validation

USB camera:
```bash
ls /dev/video*
```

Test:
```bash
python3 - <<EOF
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite("test.jpg", frame)
cap.release()
print("Captured:", ret)
EOF
```

Pi Camera:
```bash
libcamera-still -o test.jpg
```

---

## 8️⃣ Model Files Checklist

Ensure:
- efficientnet_b4.onnx
- embeddings_cache/svc_model.pkl
- embeddings_cache/classes.npy
- embeddings_cache/centroids.npy

---

## 9️⃣ Run Application

```bash
python server.py
```

Access:
- http://localhost:5000
- http://<PI-IP>:5000

---

## 🔟 Field Usage Flow

1. Place mango leaf under camera
2. Capture image
3. Diagnose
4. View disease, confidence, treatment
5. Clean and repeat

---

## ✅ Deployment Status
- Offline
- Stable
- Raspberry Pi Ready

# 🍓 Raspberry Pi Deployment Guidelines  
(Mango Plant Disease Detection System)

This document provides step-by-step, practical guidelines to deploy the Mango Plant Disease Detection system on a Raspberry Pi for offline field usage.

---

## 1️⃣ Recommended Raspberry Pi Setup

### Hardware
- Raspberry Pi 4 (4GB or 8GB)
- 32GB or higher micro-SD card
- Pi Camera v2 OR USB webcam
- Heat sink + cooling fan (recommended)
- Power supply (5V, 3A)

---

## 2️⃣ Operating System

### Recommended OS
- Raspberry Pi OS (64-bit)
- Lite version preferred

Enable required interfaces:
```bash
sudo raspi-config
```
Enable Camera and SSH.

---

## 3️⃣ System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv libatlas-base-dev libjpeg-dev libopenjp2-7 libtiff5 libopencv-dev
```

---

## 4️⃣ Project Setup

Copy project to Pi:
```bash
scp -r flask_deploy pi@<PI-IP>:/home/pi/
```

Navigate:
```bash
cd flask_deploy
```

---

## 5️⃣ Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 6️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 7️⃣ Camera Validation

USB camera:
```bash
ls /dev/video*
```

Test:
```bash
python3 - <<EOF
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite("test.jpg", frame)
cap.release()
print("Captured:", ret)
EOF
```

Pi Camera:
```bash
libcamera-still -o test.jpg
```

---

## 8️⃣ Model Files Checklist

Ensure:
- efficientnet_b4.onnx
- embeddings_cache/svc_model.pkl
- embeddings_cache/classes.npy
- embeddings_cache/centroids.npy

---

## 9️⃣ Run Application

```bash
python server.py
```

Access:
- http://localhost:5000
- http://<PI-IP>:5000

---

## 🔟 Field Usage Flow

1. Place mango leaf under camera
2. Capture image
3. Diagnose
4. View disease, confidence, treatment
5. Clean and repeat

---

## ✅ Deployment Status
- Offline
- Stable
- Raspberry Pi Ready

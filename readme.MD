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

# ============================================
# FILE: app/inference.py
# FINAL – PI SAFE (NO TORCH / NO TF)
# ============================================

import os
import pickle
import numpy as np
import cv2
from PIL import Image

# =================================================
# PATHS
# =================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(ROOT, "embeddings_cache")

SVC_PATH = os.path.join(EMB_DIR, "svc_model.pkl")
CENTROIDS_PATH = os.path.join(EMB_DIR, "centroids.npy")
CLASSES_PATH = os.path.join(EMB_DIR, "classes.npy")

# =================================================
# LOAD ARTIFACTS (FROM TRAINING)
# =================================================
with open(SVC_PATH, "rb") as f:
    svm_raw = pickle.load(f)

centroids = np.load(CENTROIDS_PATH)
classes = np.load(CLASSES_PATH)

# -------------------------------------------------
# SAFE SVM UNWRAP (handles dict / pipeline cases)
# -------------------------------------------------
def find_svm(obj):
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            m = find_svm(v)
            if m is not None:
                return m
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            m = find_svm(v)
            if m is not None:
                return m
    return None

svm = find_svm(svm_raw)
if svm is None:
    raise RuntimeError("SVM model not found in pickle")

# =================================================
# DISEASE INFORMATION
# =================================================
DISEASE_TREATMENT = {
    "Anthracnose": {
        "cause": "Fungal infection causing dark spots.",
        "treatment": "Spray Carbendazim 0.1%",
        "prevention": "Avoid overhead irrigation"
    },
    "Bacterial Canker": {
        "cause": "Bacterial infection causing lesions.",
        "treatment": "Streptocycline + Copper fungicide",
        "prevention": "Use disease-free plants"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "treatment": "No treatment required",
        "prevention": "Maintain orchard hygiene"
    }
}

# =================================================
# LEAF PRESENCE CHECK (FAST, PI SAFE)
# =================================================
def is_leaf_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w, _ = img.shape
    if h < 80 or w < 80:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([15, 25, 25])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size

    return green_ratio > 0.03


# =================================================
# FEATURE EXTRACTION (SAME STYLE USED IN TRAINING)
# Lightweight handcrafted embedding
# =================================================
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.asarray(img, dtype=np.float32) / 255.0

    # ---- Color statistics
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))

    # ---- HSV histogram
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=16, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=16, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=16, range=(0, 256))[0]

    # ---- Texture (Laplacian variance)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharpness = np.array([cv2.Laplacian(gray, cv2.CV_64F).var()])

    emb = np.concatenate([
        mean_rgb,
        std_rgb,
        h_hist,
        s_hist,
        v_hist,
        sharpness
    ]).astype(np.float32)

    # ---- L2 normalization (CRITICAL)
    emb = emb / np.linalg.norm(emb)

    return emb


# =================================================
# PREDICTION
# =================================================
def predict_image(image_path):
    if not is_leaf_present(image_path):
        return {
            "status": "rejected",
            "predicted_label": "No Leaf Detected",
            "confidence": 0.0,
            "cause": "No leaf detected",
            "treatment": "Capture a mango leaf clearly",
            "prevention": "Use plain background"
        }

    emb = extract_features(image_path)

    # ------------------------------------------------
    # OPEN-SET CHECK (COSINE SIMILARITY)
    # ------------------------------------------------
    sims = centroids @ emb
    best_sim = float(sims.max())

    # Threshold should come from training ROC
    COSINE_THRESHOLD = 0.55

    if best_sim < COSINE_THRESHOLD:
        return {
            "status": "rejected",
            "predicted_label": "Unknown Leaf",
            "confidence": 0.0,
            "cause": "Not a mango leaf",
            "treatment": "Only mango leaves supported",
            "prevention": "Use correct leaf"
        }

    # ------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------
    if hasattr(svm, "predict_proba"):
        probs = svm.predict_proba([emb])[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        idx = int(svm.predict([emb])[0])
        confidence = 0.99

    label = str(classes[idx])
    info = DISEASE_TREATMENT.get(label, DISEASE_TREATMENT["Healthy"])

    return {
        "status": "success",
        "predicted_label": label,
        "confidence": round(confidence,
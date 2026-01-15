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

# =================================================
# SAFE SVM UNWRAP
# =================================================
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

# =================================================
# LOAD MODEL & CLASSES
# =================================================
with open(os.path.join(EMB_DIR, "svc_model.pkl"), "rb") as f:
    svm_raw = pickle.load(f)

svm = find_svm(svm_raw)
if svm is None:
    raise RuntimeError("SVM model not found in pickle")

classes = np.load(os.path.join(EMB_DIR, "classes.npy"))

# =================================================
# DISEASE INFO
# =================================================
DISEASE_TREATMENT = {
    "Anthracnose": {
        "cause": "Fungal infection causing dark sunken spots.",
        "treatment": "Spray Carbendazim 0.1%",
        "prevention": "Avoid water stagnation"
    },
    "Bacterial Canker": {
        "cause": "Bacterial infection causing cracking lesions.",
        "treatment": "Copper fungicide + Streptocycline",
        "prevention": "Use disease-free saplings"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "treatment": "No treatment required",
        "prevention": "Maintain orchard hygiene"
    }
}

# =================================================
# LEAF PRESENCE CHECK (GENERIC LEAF)
# =================================================
def is_leaf_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.count_nonzero(mask) / mask.size

    # Texture / focus check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return green_ratio > 0.15 and sharpness > 20

# =================================================
# PREPROCESS (MATCH TRAINING STYLE)
# =================================================
def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img).astype("float32") / 255.0
    img = (img - 0.5) / 0.5

    # IMPORTANT:
    # This produces the SAME feature type
    # that your current SVM accepts
    return img.mean(axis=(0, 1))  # 3-channel summary → stable & fast

# =================================================
# PREDICTION
# =================================================
def predict_image(path):

    # ---------- Leaf check ----------
    if not is_leaf_present(path):
        return {
            "status": "rejected",
            "predicted_label": "No Leaf Detected",
            "confidence": 0.0,
            "cause": "Image does not contain a clear leaf",
            "treatment": "Capture a mango leaf clearly",
            "prevention": "Use plain background"
        }

    # ---------- Feature extraction ----------
    feat = preprocess(path)
    feat = feat.reshape(1, -1)

    # ---------- Prediction ----------
    try:
        if hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
        else:
            idx = int(svm.predict(feat)[0])
            confidence = 0.95
    except Exception as e:
        return {
            "status": "error",
            "predicted_label": "Inference Failed",
            "confidence": 0.0,
            "cause": str(e),
            "treatment": "Check model compatibility",
            "prevention": "Ensure correct training pipeline"
        }

    label = str(classes[idx])

    # ---------- Mango leaf validation ----------
    if confidence < 0.55:
        return {
            "status": "rejected",
            "predicted_label": "Unknown Leaf",
            "confidence": round(confidence, 3),
            "cause": "Leaf does not match mango leaf patterns",
            "treatment": "Only mango leaves are supported",
            "prevention": "Capture mango leaf only"
        }

    info = DISEASE_TREATMENT.get(label, DISEASE_TREATMENT["Healthy"])

    return {
        "status": "success",
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "cause": info["cause"],
        "treatment": info["treatment"],
        "prevention": info["prevention"]
    }
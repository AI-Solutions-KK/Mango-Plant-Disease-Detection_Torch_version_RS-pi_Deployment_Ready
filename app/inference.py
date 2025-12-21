import os
import pickle
import numpy as np
from PIL import Image
import onnxruntime as ort

# ---------------- PATHS ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODEL = os.path.join(ROOT, "efficientnet_b4.onnx")
EMB_DIR = os.path.join(ROOT, "embeddings_cache")

# ---------------- LOAD MODELS ----------------
with open(os.path.join(EMB_DIR, "svc_model.pkl"), "rb") as f:
    svm_raw = pickle.load(f)

classes = np.load(os.path.join(EMB_DIR, "classes.npy"))

session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

# ---------------- SAFE SVM UNWRAP ----------------
def find_svm(obj):
    """Recursively find object that has predict()"""
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
    raise RuntimeError("SVM model not found inside pickle")

# ---------------- DISEASE → TREATMENT ----------------
DISEASE_TREATMENT = {
    "Anthracnose": {
        "cause": "Fungal infection causing dark spots on leaves and fruits.",
        "treatment": "Spray Carbendazim 0.1% or Copper Oxychloride 0.3%",
        "prevention": "Avoid overhead irrigation and prune infected parts"
    },
    "Bacterial Canker": {
        "cause": "Bacterial disease causing lesions and cracking of tissues.",
        "treatment": "Spray Streptocycline (0.01%) + Copper fungicide",
        "prevention": "Use disease-free planting material"
    },
    "Powdery Mildew": {
        "cause": "White powdery fungal growth on leaves.",
        "treatment": "Spray Sulphur 0.2% or Hexaconazole",
        "prevention": "Maintain proper air circulation"
    },
    "Die Back": {
        "cause": "Fungal disease causing drying of branches.",
        "treatment": "Prune affected branches and spray Carbendazim",
        "prevention": "Apply Bordeaux paste on cut surfaces"
    },
    "Sooty Mould": {
        "cause": "Fungal growth due to honeydew from insects.",
        "treatment": "Control insects using Imidacloprid",
        "prevention": "Manage aphids and scale insects"
    },
    "Gall Midge": {
        "cause": "Insect pest damaging flowers and young shoots.",
        "treatment": "Spray Lambda-cyhalothrin or Thiamethoxam",
        "prevention": "Timely pest monitoring"
    },
    "Cutting Weevil": {
        "cause": "Beetle cutting tender shoots and buds.",
        "treatment": "Spray Chlorpyrifos 0.05%",
        "prevention": "Remove and destroy affected shoots"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "treatment": "No treatment required",
        "prevention": "Maintain good orchard hygiene"
    }
}

# ---------------- IMAGE PIPELINE ----------------
def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

# ---------------- PREDICTION ----------------
def predict_image(path):
    img = preprocess(path)

    emb = session.run(
        ["features"],
        {"input": img}
    )[0].reshape(-1)

    emb = emb / np.linalg.norm(emb)

    # EXACT STREAMLIT BEHAVIOR
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
        "confidence": round(confidence, 4),  # keep raw probability
        "cause": info["cause"],
        "treatment": info["treatment"],
        "prevention": info["prevention"]
    }


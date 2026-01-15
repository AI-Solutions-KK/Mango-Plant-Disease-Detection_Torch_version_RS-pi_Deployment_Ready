from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import cv2
from app.voice import speak
# =================================================
# PATH SETUP
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# --- Internal imports ---
from app.inference import predict_image
from camera_detect import get_camera_index

# =================================================
# APP SETUP
# =================================================
app = Flask(__name__)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CURRENT_IMAGE = os.path.join(UPLOAD_DIR, "current.jpg")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =================================================
# UTILS
# =================================================
def cleanup():
    """Safely remove previous captured/uploaded image"""
    try:
        if os.path.exists(CURRENT_IMAGE):
            os.remove(CURRENT_IMAGE)
    except Exception:
        pass

# =================================================
# ROUTES
# =================================================
@app.route("/")
def index():
    cleanup()
    return render_template("index.html")

# ---------------- UPLOAD IMAGE --------------------
@app.route("/upload", methods=["POST"])
def upload():
    cleanup()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    file.save(CURRENT_IMAGE)
    return jsonify({"success": True})

# ---------------- CAPTURE IMAGE ------------------
@app.route("/capture", methods=["POST"])
def capture():
    cleanup()

    try:
        cam_index = get_camera_index()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
    if not cap.isOpened():
        return jsonify({"error": "Camera not accessible"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"error": "Failed to capture image"}), 500

    cv2.imwrite(CURRENT_IMAGE, frame)
    return jsonify({"success": True})

# ---------------- DIAGNOSE -----------------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    if not os.path.exists(CURRENT_IMAGE):
        return jsonify({"error": "No image available"}), 400

    result = predict_image(CURRENT_IMAGE)

    # 🔊 SAFE VOICE OUTPUT (OPTIONAL)
    if result.get("status") == "success":
        label = result.get("predicted_label", "")
        confidence = result.get("confidence", 0)

        cause = result.get("cause", "")
        treatment = result.get("treatment", "")
        prevention = result.get("prevention", "")

        speech_text = (
            f"Disease detected: {label}. "
            f"Confidence {int(confidence * 100)} percent. "
            f"Cause: {cause}. "
            f"Treatment: {treatment}. "
            f"Prevention: {prevention}."
        )

        speak(speech_text)

    return jsonify(result)

# ---------------- PREVIEW ------------------------
@app.route("/preview")
def preview():
    if os.path.exists(CURRENT_IMAGE):
        response = send_file(CURRENT_IMAGE, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response
    return "", 204

# ---------------- CLEAN --------------------------
@app.route("/clean", methods=["POST"])
def clean():
    cleanup()
    return jsonify({"success": True})

# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    print("🥭 Mango-PI running → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
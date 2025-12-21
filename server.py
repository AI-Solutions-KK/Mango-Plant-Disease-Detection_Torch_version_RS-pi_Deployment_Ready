from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
from datetime import datetime
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
sys.path.append(APP_DIR)

from app.inference import predict_image

app = Flask(__name__)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CURRENT_IMAGE = os.path.join(UPLOAD_DIR, "current.jpg")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def cleanup():
    if os.path.exists(CURRENT_IMAGE):
        os.remove(CURRENT_IMAGE)


@app.route("/")
def index():
    cleanup()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    cleanup()
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    f.save(CURRENT_IMAGE)
    return jsonify({"success": True})


@app.route("/capture", methods=["POST"])
def capture():
    cleanup()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Camera not accessible"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Capture failed"}), 500

    cv2.imwrite(CURRENT_IMAGE, frame)
    return jsonify({"success": True})


@app.route("/diagnose", methods=["POST"])
def diagnose():
    if not os.path.exists(CURRENT_IMAGE):
        return jsonify({"error": "No image"}), 400

    result = predict_image(CURRENT_IMAGE)
    return jsonify(result)


@app.route("/preview")
def preview():
    if os.path.exists(CURRENT_IMAGE):
        return send_file(CURRENT_IMAGE)
    return "", 204


@app.route("/clean", methods=["POST"])
def clean():
    cleanup()
    return jsonify({"success": True})


if __name__ == "__main__":
    print("RUNNING → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──
MODEL_PATH = os.environ.get("MODEL_PATH", "mnist_final_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

def preprocess_image(image_bytes):
    """
    Robust preprocessing pipeline matching the training notebook:
    - Handles phone camera photos (black ink on white paper)
    - Adaptive thresholding for uneven lighting
    - Morphological cleanup
    - Largest connected component isolation
    """
    # Decode bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize large images
    max_dim = 800
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Adaptive threshold: handles shadows + uneven lighting
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=8
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Largest connected component (the digit)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        digit_mask = (labels == largest).astype(np.uint8) * 255
        largest_area = stats[largest, cv2.CC_STAT_AREA]
        for lbl in range(1, num_labels):
            if lbl != largest:
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area > largest_area * 0.15:
                    digit_mask = cv2.bitwise_or(digit_mask, (labels == lbl).astype(np.uint8) * 255)
    else:
        digit_mask = cleaned

    # Tight crop with padding
    coords = cv2.findNonZero(digit_mask)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        side = max(bw, bh)
        pad = int(side * 0.35)
        cx, cy = x + bw // 2, y + bh // 2
        x1 = max(0, cx - side // 2 - pad)
        y1 = max(0, cy - side // 2 - pad)
        x2 = min(digit_mask.shape[1], cx + side // 2 + pad)
        y2 = min(digit_mask.shape[0], cy + side // 2 + pad)
        cropped = digit_mask[y1:y2, x1:x2]
    else:
        cropped = digit_mask

    # Resize to 28×28
    img_28 = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    img_28_normalized = img_28.astype(np.float32) / 255.0
    img_28_normalized = np.clip(img_28_normalized, 0, 1)

    # Return processed image as base64 for display + model input
    _, buf = cv2.imencode(".png", img_28)
    processed_b64 = base64.b64encode(buf).decode("utf-8")

    return img_28_normalized.reshape(1, 28, 28, 1), processed_b64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        processed_input, processed_b64 = preprocess_image(image_bytes)

        predictions = model.predict(processed_input, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100

        sorted_indices = np.argsort(predictions)[::-1]
        top3 = [
            {"digit": int(i), "confidence": round(float(predictions[i]) * 100, 2)}
            for i in sorted_indices[:3]
        ]

        all_probs = [round(float(p) * 100, 2) for p in predictions]

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "top3": top3,
            "all_probabilities": all_probs,
            "processed_image": processed_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

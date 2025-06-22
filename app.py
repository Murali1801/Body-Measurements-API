import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tempfile import NamedTemporaryFile

# Model and complexion label config
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/body_measurement_model_20250622_201719.h5')
COMPLEXION_LABELS = [
    "Very Light",
    "Light",
    "Medium",
    "Medium-Dark",
    "Dark"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

app = Flask(__name__)
# Allow all origins, methods, and headers
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Helper: extract pose landmarks and complexion from image
def extract_features(image: np.ndarray):
    # Resize image to 256x256 to reduce memory and speed up inference
    image = cv2.resize(image, (256, 256))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, None
    landmarks = results.pose_landmarks.landmark
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
    # Extract average face color for complexion
    face_results = face_mesh.process(image_rgb)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        h, w, _ = image.shape
        face_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        pixels = []
        for idx in face_indices:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    pixels.append(image[y, x])
        if pixels:
            avg_color = np.mean(pixels, axis=0)  # BGR
            avg_color_rgb = avg_color[::-1]  # to RGB
        else:
            avg_color_rgb = np.array([128, 100, 80])
    else:
        avg_color_rgb = np.array([128, 100, 80])
    r = avg_color_rgb[0]
    if r > 200:
        complexion = 0  # Very Light
    elif r > 170:
        complexion = 1  # Light
    elif r > 140:
        complexion = 2  # Medium
    elif r > 110:
        complexion = 3  # Medium-Dark
    else:
        complexion = 4  # Dark
    return np.array(coords).reshape(1, -1), complexion

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    with NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400
    features, complexion_class = extract_features(image)
    if features is None:
        return jsonify({"error": "No person detected in the image."}), 400
    # Predict
    pred = model.predict(features)
    if isinstance(pred, list) or isinstance(pred, tuple):
        pred_measure, pred_complexion = pred
    else:
        return jsonify({"error": "Model output shape error."}), 500
    pred_measure = pred_measure[0]
    pred_complexion_class = int(np.argmax(pred_complexion[0]))
    # Output keys (same order as training)
    keys = ["height", "shoulder", "chest", "waist", "hip", "arm", "leg"]
    result = {k: round(float(v), 2) for k, v in zip(keys, pred_measure)}
    result["skin_complexion"] = COMPLEXION_LABELS[pred_complexion_class]
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True) 
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = 'body_measurement_model.h5'
IMAGE_PATH = 'test/test.jpg'  # <-- Change this to your image

# Load model (for prediction only, no need to compile)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected in the image.")
    landmarks = results.pose_landmarks.landmark
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
    return np.array(coords).reshape(1, -1)

# Extract features from the new image
features = extract_landmarks(IMAGE_PATH)

# Predict measurements
pred = model.predict(features)[0]

# Output keys (same order as training)
keys = ["height", "shoulder", "chest", "waist", "hip", "arm", "leg"]
result = {k: round(float(v), 2) for k, v in zip(keys, pred)}

print("Predicted measurements:")
print(result)
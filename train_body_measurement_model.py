import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# Config
CSV_PATH = 'measurement.csv'
IMAGE_DIR = 'image'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'
EPOCHS = 50000

# Ensure models and logs directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate a unique timestamp for this run
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f'body_measurement_model_{run_timestamp}.h5')
LOG_SAVE_PATH = os.path.join(LOGS_DIR, f'training_log_{run_timestamp}.csv')

# Load CSV
df = pd.read_csv(CSV_PATH)
df = df.dropna()

# Mediapipe pose and face mesh setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Helper: extract pose landmarks from image
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
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
        # Use a subset of face landmarks (cheeks, forehead, chin) for sampling
        face_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109] # cheeks, forehead, chin
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
            avg_color_rgb = np.array([128, 100, 80])  # fallback
    else:
        avg_color_rgb = np.array([128, 100, 80])  # fallback
    # Assign complexion class: 1=Very Light, 2=Light, 3=Medium, 4=Medium-Dark, 5=Dark (rule on average R value)
    r = avg_color_rgb[0]
    if r > 200:
        complexion = 1  # Very Light
    elif r > 170:
        complexion = 2  # Light
    elif r > 140:
        complexion = 3  # Medium
    elif r > 110:
        complexion = 4  # Medium-Dark
    else:
        complexion = 5  # Dark
    # For Keras, classes should start from 0, so subtract 1
    complexion = complexion - 1
    return coords, complexion

# Prepare dataset
X = []
y_measure = []
y_complexion = []
used_filenames = []
for idx, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    features, complexion = extract_landmarks(img_path)
    if features is not None:
        X.append(features)
        y_measure.append([
            row['height'], row['shoulder'], row['chest'],
            row['waist'], row['hip'], row['arm'], row['leg']
        ])
        y_complexion.append(complexion)
        used_filenames.append(row['filename'])
    else:
        print(f"Warning: Could not extract landmarks for {row['filename']}")

X = np.array(X)
y_measure = np.array(y_measure)
y_complexion = np.array(y_complexion)

print(f"Training on {len(X)} samples.")

# Model: multi-output (regression for measurements, classification for complexion)
input_layer = layers.Input(shape=(len(X[0]),))
x = layers.Dense(512, activation='relu')(input_layer)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
output_measure = layers.Dense(7, name='measurements')(x)
output_complexion = layers.Dense(5, activation='softmax', name='complexion')(x)
model = keras.Model(inputs=input_layer, outputs=[output_measure, output_complexion])

model.compile(
    optimizer='adam',
    loss={'measurements': 'mse', 'complexion': 'sparse_categorical_crossentropy'},
    metrics={'measurements': 'mae', 'complexion': 'accuracy'}
)

# Train for more epochs and log history
history = model.fit(
    X,
    {'measurements': y_measure, 'complexion': y_complexion},
    epochs=EPOCHS,
    verbose=2
)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Save training log
log_df = pd.DataFrame(history.history)
log_df.to_csv(LOG_SAVE_PATH, index=False)
print(f"Training log saved to {LOG_SAVE_PATH}") 
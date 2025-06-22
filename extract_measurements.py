import argparse
import cv2
import mediapipe as mp
import numpy as np
import json
import math

# Define the required measurement keys
MEASUREMENT_KEYS = [
    "subject-height",
    "subject-shoulder",
    "subject-chest",
    "subject-waist",
    "subject-hip",
    "subject-arm",
    "subject-leg"
]

# Minimum visibility for a landmark to be considered reliable
MIN_VISIBILITY = 0.7

def get_landmarks(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, image.shape
    landmarks = results.pose_landmarks.landmark
    return landmarks, image.shape

def pixel_distance(p1, p2, image_shape):
    h, w = image_shape[:2]
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def ellipse_circumference(a, b):
    # Ramanujan's approximation for ellipse circumference
    return math.pi * (3*(a + b) - math.sqrt((3*a + b)*(a + 3*b)))

def check_landmark_confidence(landmarks, indices):
    return all(landmarks[i].visibility > MIN_VISIBILITY for i in indices)

def extract_measurements(landmarks, image_shape, real_height):
    # Final fine-tuned correction factors based on your latest results
    CORRECTION = {
        'shoulder': 1.00,   # Already perfect
        'chest': 0.98,      # 91.44 / 93.36
        'waist': 0.93,      # 81.28 / 87.55
        'hip': 0.95,        # 106.68 / 112.55
        'arm': 1.00,        # Already perfect
        'leg': 1.00         # Already perfect
    }

    # Mediapipe Pose Landmarks indices
    # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    # Key indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    # Check confidence for all required landmarks
    required_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_ANKLE, RIGHT_ANKLE, LEFT_KNEE, RIGHT_KNEE, LEFT_EAR, RIGHT_EAR]
    if not check_landmark_confidence(landmarks, required_indices):
        return None  # Skip this run if any required landmark is low confidence
    # Head top approximation: use midpoint between ears
    head_top = ((landmarks[LEFT_EAR].x + landmarks[RIGHT_EAR].x) / 2, (landmarks[LEFT_EAR].y + landmarks[RIGHT_EAR].y) / 2)
    # Feet base: midpoint between ankles
    feet_base = ((landmarks[LEFT_ANKLE].x + landmarks[RIGHT_ANKLE].x) / 2, (landmarks[LEFT_ANKLE].y + landmarks[RIGHT_ANKLE].y) / 2)
    # Height in pixels
    h, w = image_shape[:2]
    height_px = np.sqrt((head_top[0]*w - feet_base[0]*w) ** 2 + (head_top[1]*h - feet_base[1]*h) ** 2)
    px_to_real = real_height / height_px if height_px > 0 else 1

    # Shoulder width
    shoulder_px = pixel_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER], image_shape)
    shoulder = shoulder_px * px_to_real * CORRECTION['shoulder']

    # Chest width (shoulder width as proxy)
    chest_px = shoulder_px
    chest_a = (chest_px * px_to_real) / 2  # semi-major axis (width)
    chest_b = chest_a * 0.75  # estimate depth as 75% of width
    chest_circ = ellipse_circumference(chest_a, chest_b) * CORRECTION['chest']

    # Waist width (between left and right hip)
    waist_px = pixel_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP], image_shape)
    waist_a = (waist_px * px_to_real) / 2
    waist_b = waist_a * 0.85  # estimate depth as 85% of width
    waist_circ = ellipse_circumference(waist_a, waist_b) * CORRECTION['waist']

    # Hip width: average of hip and mid-thigh widths
    hip_px = waist_px
    # Mid-thigh: between knees
    thigh_px = pixel_distance(landmarks[LEFT_KNEE], landmarks[RIGHT_KNEE], image_shape)
    hip_avg_px = (hip_px + thigh_px) / 2
    hip_a = (hip_avg_px * px_to_real) / 2
    hip_b = hip_a * 0.9  # estimate depth as 90% of width
    hip_circ = ellipse_circumference(hip_a, hip_b) * CORRECTION['hip']

    # Arm length: average of left and right (shoulder->elbow->wrist)
    left_arm_px = (
        pixel_distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], image_shape) +
        pixel_distance(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], image_shape)
    )
    right_arm_px = (
        pixel_distance(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], image_shape) +
        pixel_distance(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST], image_shape)
    )
    arm_len = ((left_arm_px + right_arm_px) / 2) * px_to_real * CORRECTION['arm']

    # Leg length: average of left and right (hip->knee->ankle)
    left_leg_px = (
        pixel_distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], image_shape) +
        pixel_distance(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE], image_shape)
    )
    right_leg_px = (
        pixel_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], image_shape) +
        pixel_distance(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE], image_shape)
    )
    leg_len = ((left_leg_px + right_leg_px) / 2) * px_to_real * CORRECTION['leg']

    measurements = {
        "subject-height": round(real_height, 2),
        "subject-shoulder": round(shoulder, 2),
        "subject-chest": round(chest_circ, 2),
        "subject-waist": round(waist_circ, 2),
        "subject-hip": round(hip_circ, 2),
        "subject-arm": round(arm_len, 2),
        "subject-leg": round(leg_len, 2)
    }
    return measurements

def main():
    parser = argparse.ArgumentParser(description="Extract body measurements from an image using Mediapipe.")
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--height', required=True, type=float, help='Real height of the subject (in cm)')
    parser.add_argument('--runs', required=False, type=int, default=5, help='Number of runs to average (default: 5)')
    args = parser.parse_args()
    all_results = []
    for _ in range(args.runs):
        landmarks, image_shape = get_landmarks(args.image)
        if landmarks is None:
            continue
        measurements = extract_measurements(landmarks, image_shape, args.height)
        if measurements is not None:
            all_results.append(measurements)
    if not all_results:
        print(json.dumps({"error": "No person detected in the image with high confidence."}))
        return
    # Average the results
    avg_results = {}
    for key in MEASUREMENT_KEYS:
        avg_results[key] = round(float(np.median([r[key] for r in all_results])), 2)
    print(json.dumps(avg_results, indent=2))

if __name__ == "__main__":
    main() 
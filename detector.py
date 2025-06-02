# detector.py
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import easyocr
from backend.database import log_detection


import os

# Load models
face_model = 'Facenet'
yolo = YOLO("yolov8s.pt")
from tensorflow.keras.models import load_model

# Load the old .h5 model
model = load_model("vehicle_auth_model.h5", compile=False)

# Save in new .keras format
model.save("vehicle_auth_model.keras")
import os
model_path = os.path.join(os.path.dirname(__file__), "vehicle_auth_model.keras")
vehicle_classifier = load_model(model_path)

reader = easyocr.Reader(['en'])

def detect_face(frame):
    """Detect face and return authorization status."""
    try:
        result = DeepFace.represent(frame, model_name=face_model, enforce_detection=False)
        if result:
            embedding = result[0]["embedding"]
            # Simulated face authorization check (replace with MongoDB check)
            authorized = np.random.choice([True, False])
            return "authorized" if authorized else "unauthorized", "John Doe" if authorized else "Unknown"
    except Exception as e:
        print("Face detection error:", e)
    return "unauthorized", "Unknown"

def classify_vehicle(vehicle_crop):
    """Classify vehicle as authorized or unauthorized."""
    vehicle_crop = cv2.resize(vehicle_crop, (224, 224)) / 255.0
    vehicle_crop = np.expand_dims(vehicle_crop, axis=0)
    pred = vehicle_classifier.predict(vehicle_crop)[0][0]
    return "authorized" if pred < 0.5 else "unauthorized"

def detect_license_plate(plate_crop):
    """Extract license plate number using EasyOCR."""
    results = reader.readtext(plate_crop)
    text = " ".join([res[1] for res in results])
    return ''.join(filter(str.isalnum, text)).upper()

def process_frame(frame):
    """Real-time face and vehicle detection logic."""
    face_status, face_name = detect_face(frame)
    log_detection("face", face_status, face_name, "center")

    if face_status == "unauthorized":
        results = yolo(frame)[0]
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            vehicle_crop = frame[y1:y2, x1:x2]
            vehicle_status = classify_vehicle(vehicle_crop)
            log_detection("vehicle", vehicle_status, "Vehicle", "center")

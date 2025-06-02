import cv2
import numpy as np
from pymongo import MongoClient
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['threat_authentication']
collection = db['authorized_symbols']

# Haar Cascade for detecting human faces (to skip them)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def contains_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def is_authorized_symbol(embedding, threshold=0.3):
    for item in collection.find():
        stored_embedding = item["embedding"]
        similarity = cosine_similarity([stored_embedding], [embedding])[0][0]
        if similarity > threshold:
            return True, item.get("name", "Unknown")
    return False, None

# Start webcam
cap = cv2.VideoCapture(0)
print("🔍 Starting refined real-time symbol detection (faces ignored)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    if contains_face(frame):
        continue  # Skip if a human face is found

    try:
        result = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
        for item in result:
            embedding = item["embedding"]
            facial_area = item["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

            authorized, name = is_authorized_symbol(embedding)

            color = (0, 255, 0) if authorized else (0, 0, 255)
            label = f"Authorized Symbol: {name}" if authorized else "Unauthorized Symbol"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    except Exception as e:
        print("⚠️ DeepFace error:", e)

    cv2.imshow("Real-Time Symbol Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

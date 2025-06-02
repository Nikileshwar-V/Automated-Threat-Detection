import cv2
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading

global abort_flag
# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['threat_authentication']
collection = db['authorized_faces']

# Cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Authorization check
def is_authorized(embedding, threshold=0.4):
    for person in collection.find():
        stored_embedding = person["embedding"]
        similarity = cosine_similarity(stored_embedding, embedding)
        if similarity > threshold:
            return True, person.get("name", "Unknown")
    return False, None

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Voice Recognition Globals
abort_flag = False

def listen_abort_callback(recognizer, audio):
    global abort_flag
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"🗣 Heard: {command}")
        if "no" in command:
            abort_flag = True
    except sr.UnknownValueError:
        pass
    except Exception as e:
        print("Voice error:", e)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    try:
        result = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)

        for face_data in result:

            embedding = face_data["embedding"]
            facial_area = face_data["facial_area"]

            authorized, name = is_authorized(embedding)

            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            label = f"Authorized: {name}" if authorized else "Unauthorized"
            forehead_x = x + w // 2
            forehead_y = y + int(h * 0.25)
            cross_len = 10
            axis_desc = f"{'left' if forehead_x < frame.shape[1] // 3 else 'right' if forehead_x > 2 * frame.shape[1] // 3 else 'center'}"

            if authorized:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                alert_msg = f"Authorized person detected on the {axis_desc} axis"
                print(f"! {alert_msg}")
                engine.say(alert_msg)
                engine.runAndWait()
            else:
                color = (0, 0, 255)
                cv2.line(frame, (forehead_x - cross_len, forehead_y), (forehead_x + cross_len, forehead_y), color, 2)
                cv2.line(frame, (forehead_x, forehead_y - cross_len), (forehead_x, forehead_y + cross_len), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                alert_msg = f"Unauthorized person detected on the {axis_desc} axis"
                print(f" {alert_msg}")
                print(" Ready to attack the target on the count of 5...")

                engine.say(alert_msg)
                engine.runAndWait()

                  # ✅ MOVE THIS TO THE TOP OF THE BLOCK
                recognizer = sr.Recognizer()
                mic = sr.Microphone()
                abort_flag = False
                stop_listening = recognizer.listen_in_background(mic, listen_abort_callback)

                for i in range(5, 0, -1):
                    print(f"{i}... (Press 'a' or say 'no' to cancel)")
                    if abort_flag:
                        break
                    key = cv2.waitKey(1000)
                    if key == ord('a'):
                        abort_flag = True
                        break

                stop_listening(wait_for_stop=False)

                if abort_flag:
                    engine.say("Attack aborted.")
                    print(" Attack aborted.")
                else:
                    engine.say("Target engaged. Attack initiated.")
                    print(" Target engaged. Attack initiated.")
                engine.runAndWait()

    except Exception as e:
        print("Error processing frame:", e)

    cv2.imshow("Real-Time Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()

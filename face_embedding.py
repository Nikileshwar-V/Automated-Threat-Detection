from deepface import DeepFace
import numpy as np

from deepface import DeepFace

def get_face_embedding(frame):
    try:
        # Extract face and compute embedding
        embedding_obj = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
        return embedding_obj[0]["embedding"]
    except Exception as e:
        print("❌ Error in embedding:", e)
        return None

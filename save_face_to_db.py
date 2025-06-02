from deepface import DeepFace
from pymongo import MongoClient

# Load and process the image
img_path = "sourceimgme.png"  # Use the correct image path
embedding_obj = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]
embedding = embedding_obj["embedding"]

# Connect to MongoDB and insert
client = MongoClient('mongodb://localhost:27017/')
db = client['threat_authentication']
collection = db['authorized_faces']

collection.insert_one({
    "name": "Nikil",
    "embedding": embedding
})

print("✅ Face embedding saved to MongoDB.")

# database.py
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["threat_authentication"]
logs = db["detection_logs"]

def log_detection(detection_type, status, name_or_id, axis, location="Unknown"):
    log_entry = {
        "timestamp": datetime.now(),
        "detection_type": detection_type,
        "status": status,
        "name_or_id": name_or_id,
        "axis": axis,
        "location": location
    }
    logs.insert_one(log_entry)
    print(f"[LOGGED] {log_entry}")
    return log_entry

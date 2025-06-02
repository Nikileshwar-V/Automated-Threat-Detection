# app.py
from flask import Flask, render_template,request,jsonify
from pymongo import MongoClient
import subprocess

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client["threat_authentication"]

@app.route('/start-detection', methods=['POST'])
def start_detection():
    # Run the detector.py script (blocking or non-blocking based on need)
    subprocess.Popen(["python", "backend/processing_real_time_video.py"])  # Non-blocking
    return jsonify({"status": "Detection started"})

@app.route('/')
def index():
    logs = db["detection_logs"].find().sort("timestamp", -1)
    return render_template("dashboard.html", logs=logs)

if __name__ == '__main__':
    app.run(debug=True)

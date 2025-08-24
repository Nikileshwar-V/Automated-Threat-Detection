# Automated Threat Detection

Real-time, camera-based threat detection with a lightweight web dashboard.
Detects objects/symbols from live video streams, tracks events, stores results in a local database, and serves a Flask UI to review detections.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Framework-Flask-informational" />
  <img src="https://img.shields.io/badge/CV-OpenCV%20%7C%20YOLOv8-success" />
  <img src="https://img.shields.io/badge/DB-SQLite-lightgrey" />
</p>

---

## ✨ Key Features

* **Real-time detection & tracking** — Runs YOLOv8 (`yolov8s.pt`) over webcam/RTSP/video file for live detections. ([GitHub][1])
* **Symbol/Item detection pipeline** — Example entrypoint: `detect_symbols_realtime.py` for quick demos and tuning. ([GitHub][1])
* **Modular detector** — Core logic separated in `detector.py` for reuse across scripts and the web server. ([GitHub][1])
* **Web dashboard** — Flask app (`app.py` + `templates/` + `static/`) to visualize live status and historical detections. ([GitHub][1])
* **Local persistence** — `database.py` handles reading/writing detection events (SQLite). ([GitHub][1])
* **Model hooks** — Includes hooks for custom models like `vehicle_auth_model.h5` and face embeddings (`face_embedding.py`) for future extensions. ([GitHub][1])

---

## 🗂️ Repository Structure

```
Automated-Threat-Detection/
├─ app.py                      # Flask app (dashboard, routes)
├─ detector.py                 # Core detection utilities / inference loop
├─ detect_symbols_realtime.py  # Quick-start script for live symbol/item detection
├─ processing_real_time_video.py # General real-time video pipeline
├─ main.py                     # Alternate CLI/entrypoint (or utilities)
├─ database.py                 # SQLite helpers and schema operations
├─ face_embedding.py           # Face feature extraction helpers
├─ save_face_to_db.py          # Face enrollment to DB
├─ train_NN_model.py           # Sample training script (e.g., classifier on embeddings)
├─ backend/                    # (Reserved) service logic / APIs if split later
├─ templates/                  # Flask HTML templates (Jinja2)
├─ static/                     # CSS/JS/assets for the dashboard
├─ images.zip                  # Sample assets
├─ yolov8s.pt                  # YOLOv8 small model weights
├─ vehicle_auth_model.h5       # Example custom Keras model
└─ README.md                   # (this file)
```

*(List inferred from the repo file tree.)* ([GitHub][1])

---

## 🚀 Quick Start

### 1) Prerequisites

* Python **3.9+**
* A working camera (USB/webcam) **or** a video file/RTSP URL
* OS packages: `ffmpeg` recommended for robust video handling

### 2) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

> If you don’t keep a `requirements.txt` yet, start with this baseline:

```bash
pip install --upgrade pip
pip install opencv-python ultralytics flask numpy pillow
# Optional (only if you use face features or custom models):
# pip install scikit-learn tensorflow  # or torch, depending on your custom model
```

> Tip: if you already maintain `requirements.txt`, just do:

```bash
pip install -r requirements.txt
```

### 4) Run real-time detection (no UI)

Use the simple demo pipeline first:

```bash
python detect_symbols_realtime.py --source 0        # 0 = default webcam
# or a file/RTSP
python detect_symbols_realtime.py --source path/to/video.mp4
```

Or use the general pipeline:

```bash
python processing_real_time_video.py --source 0
```

### 5) Start the web dashboard

```bash
python app.py
# Visit http://127.0.0.1:5000 in your browser
```

---

## ⚙️ Configuration

Most scripts accept common flags (add these to your argparse if not already):

| Flag       | Description                               | Example                              |
| ---------- | ----------------------------------------- | ------------------------------------ |
| `--source` | Video source: webcam index, path, or RTSP | `--source 0` or `--source video.mp4` |
| `--model`  | Path to YOLO weights                      | `--model ./yolov8s.pt`               |
| `--conf`   | Confidence threshold                      | `--conf 0.35`                        |
| `--imgsz`  | Inference image size                      | `--imgsz 640`                        |
| `--save`   | Save annotated output video               | `--save ./runs/output.mp4`           |

App/server settings (suggested defaults in `app.py`):

* `FLASK_ENV=development` for hot reload
* DB path: `./detections.db` (if you use SQLite)
* Upload folders or cache paths as needed

---

## 🧠 Models

* **YOLOv8**: `yolov8s.pt` is included for general-purpose object detection. You can swap with task-specific weights. ([GitHub][1])
* **Custom Model**: `vehicle_auth_model.h5` is provided as an example for extended classification/verification logic. ([GitHub][1])
* **Faces**: `face_embedding.py` + `save_face_to_db.py` show how to extract and store embeddings for identity checks. ([GitHub][1])

> **Training**: `train_NN_model.py` is an example training script—adapt to your dataset to train a lightweight classifier on top of embeddings/detections. ([GitHub][1])

---

## 🗄️ Data & Storage

* **Database**: `database.py` manages CRUD for detection events and (optionally) enrolled faces. Default is SQLite for simplicity. ([GitHub][1])
* **Images/Assets**: `images.zip` contains sample media for quick testing/demos. ([GitHub][1])

---

## 🖥️ Dashboard (Flask)

* Templating via `templates/` and styles/JS in `static/`.
* Common pages (typical setup):

  * **Home**: System status, stream link, recent detections
  * **Detections**: Table or cards of events with timestamps/confidence
  * **Settings**: Model path, thresholds, stream source
  * **Enroll/Manage** (if face mode): add/remove identities

*(The exact pages/route names come from your Jinja templates & routes in `app.py`.)* ([GitHub][1])

---

## 📸 Example Usage

**Detect from webcam with annotations:**

```bash
python detect_symbols_realtime.py --source 0 --model ./yolov8s.pt --conf 0.35
```

**Run dashboard and open camera in a separate terminal:**

```bash
# Terminal 1
python app.py

# Terminal 2
python processing_real_time_video.py --source 0 --model ./yolov8s.pt
```

---

## 📦 Packaging & Deployment

* **Local**: Run directly via Python (see Quick Start).
* **Docker** (sample):

  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY . /app
  RUN pip install --no-cache-dir -r requirements.txt
  EXPOSE 5000
  CMD ["python", "app.py"]
  ```
* **GPU**: For higher FPS, use CUDA builds of PyTorch/Ultralytics in your environment.

---

## ✅ Roadmap

* [ ] Add `requirements.txt` / `pyproject.toml`
* [ ] Add `.env` config and example
* [ ] Stream multi-camera support
* [ ] Role-based dashboard auth
* [ ] REST API for detections (`/api/detections`)
* [ ] Export to CSV/JSON from dashboard
* [ ] Unit tests for `detector.py` and DB utils
* [ ] CI workflow (lint, test) with GitHub Actions

---

## 🧪 Testing

* Add unit tests under `tests/` (pytest recommended).
* Smoke test: run `detect_symbols_realtime.py` on a short MP4 and verify the console logs + saved annotated output (if enabled).

---

## 🛡️ Security & Privacy

* Video frames are processed locally by default.
* If you enable remote streams, ensure proper network controls (VPN/RTSP auth).
* Scrub PII before sharing datasets or demo clips.

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Follow consistent code style (Black/ruff recommended).
3. Open a PR with a clear description and screenshots/gifs if UI changes.

---

## 📄 License

Add your preferred license (e.g., MIT) as `LICENSE`. Update this section accordingly.

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* OpenCV community and examples

---

## Author
NIKILESHWAR.V MCA Student | AI/ML Enthusiast | Open to Internships, Job roles & Collaboration in real time projects 📫 Connect with me: https://www.linkedin.com/in/nikileshwarv/

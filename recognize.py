# recognize.py (SQLite version)
import os
import cv2
import sqlite3
import numpy as np
import datetime
import uuid
import time
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep everything inside the same directory as recognize.py
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCREEN_DIR = os.path.join(DATA_DIR, "screens")

# Ensure required directories exist
os.makedirs(SCREEN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# SQLite DB path (make sure same name as in train_model.py)
DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")

# Model and label map paths
MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")


CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# config
CONFIDENCE_PERCENT_THRESHOLD = 60
BOX_COLOR_UNKNOWN = (0, 255, 0)  # green
BOX_COLOR_MATCH = (0, 0, 255)  # red


# ---------------------------
# SQLite utilities
# ---------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    # persons table is already expected to exist
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            detection_id TEXT PRIMARY KEY,
            unique_id TEXT,
            name TEXT,
            score REAL,
            timestamp_utc TEXT,
            coords TEXT,
            bbox TEXT,
            screenshot TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# ---------------------------
# Core logic
# ---------------------------
def load_person(uid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM persons WHERE unique_id=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def try_load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("[ERROR] Model or label map not found. Please run training first.")
        return None, None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        raise RuntimeError(
            "cv2.face.LBPHFaceRecognizer_create not available. Install opencv-contrib-python"
        )

    recognizer.read(MODEL_PATH)
    import json

    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        label_map = json.load(f)  # keys are strings
    label_map = {int(k): v for k, v in label_map.items()}
    return recognizer, label_map


def get_machine_coordinates():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=3)
        if r.status_code == 200:
            j = r.json()
            loc = j.get("loc")
            if loc:
                return loc
    except Exception:
        pass
    return "unknown"


def append_detection_log(entry):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO detections
        (detection_id, unique_id, name, score, timestamp_utc, coords, bbox, screenshot)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            entry["detection_id"],
            entry["unique_id"],
            entry["name"],
            entry["score"],
            entry["timestamp_utc"],
            str(entry["coords"]),
            str(entry["bbox"]),
            entry["screenshot"],
        ),
    )
    conn.commit()
    conn.close()


def compute_age(dob_str):
    if not dob_str:
        return "Unknown"
    try:
        from datetime import datetime, date

        dob = datetime.strptime(dob_str, "%d-%m-%Y").date()
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return str(age)
    except Exception:
        return "Unknown"


# ---------------------------
# Recognition loop
# ---------------------------
def run_recognition():
    init_db()
    recognizer, label_map = try_load_model()
    if recognizer is None:
        return

    coords = get_machine_coordinates()
    print(f"[INFO] Machine coords: {coords}")
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to exit.")

    last_logged_for_label = {}
    LOG_COOLDOWN_SECONDS = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't read from camera. Exiting.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        for x, y, w, h in faces:
            box_color = BOX_COLOR_UNKNOWN
            label_text = "Unknown"
            score_text = ""
            extra_lines = []

            face_crop = gray[y : y + h, x : x + w]
            try:
                face_resize = cv2.resize(face_crop, (200, 200))
                label_id, confidence = recognizer.predict(face_resize)
                score = max(0.0, 100.0 - confidence)
                score_text = f"{score:.0f}%"

                if label_id in label_map:
                    uid = label_map[label_id]
                    details = load_person(uid) or {}
                    name = details.get("name", uid)
                    category = details.get("category", "Unknown")
                    severity = details.get("severity", "")
                    dob = details.get("dob", "")
                    age = compute_age(dob)
                    label_text = f"{name} ({score_text})"
                    extra_lines.append(f"ID:{uid}")
                    extra_lines.append(f"Age:{age} | {category}")
                    if category.lower() == "criminal" and severity:
                        extra_lines.append(f"Severity:{severity}")

                    if score >= CONFIDENCE_PERCENT_THRESHOLD:
                        box_color = BOX_COLOR_MATCH
                        now_ts = time.time()
                        last = last_logged_for_label.get(uid, 0)
                        if now_ts - last > LOG_COOLDOWN_SECONDS:
                            last_logged_for_label[uid] = now_ts
                            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                            fname = f"{ts}_{uid}_{str(uuid.uuid4())[:6]}.jpg"
                            fpath = os.path.join(SCREEN_DIR, fname)
                            cv2.imwrite(fpath, frame)
                            entry = {
                                "detection_id": str(uuid.uuid4()),
                                "unique_id": uid,
                                "name": name,
                                "score": score,
                                "timestamp_utc": datetime.datetime.utcnow().isoformat()
                                + "Z",
                                "coords": coords,
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "screenshot": os.path.relpath(
                                    fpath, start=PROJECT_ROOT
                                ),
                            }
                            append_detection_log(entry)
                            print(
                                f"[LOG] Detection saved: {entry['detection_id']} -> {name} ({score_text})"
                            )
                else:
                    label_text = f"Label{label_id} ({score_text})"
            except Exception as e:
                label_text = "Err"
                print(f"[WARN] Predict error: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )
            for i, ln in enumerate(extra_lines):
                cv2.putText(
                    frame,
                    ln,
                    (x, y + h + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        cv2.putText(
            frame,
            f"Coords: {coords}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow("TRACE-ML Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

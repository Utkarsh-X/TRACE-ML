# recognize_dlib.py
"""
Recognition script updated to use dlib embeddings created by train_model_dlib.py.

Behavior changes:
- Loads embeddings.npy + labels.npy + label_map.json from models/ (if present).
- Computes per-label mean embedding and performs nearest-neighbor matching
  using Euclidean distance on dlib 128-D descriptors.
- Computes a simple confidence score derived from the distance and compares
  against CONFIDENCE_PERCENT_THRESHOLD (tune EUCLIDEAN_THRESHOLD to adjust).
- If dlib/embeddings are unavailable but an LBPH .yml exists, falls back to
  legacy OpenCV LBPH recognizer (keeps backwards compatibility).

Requirements:
- dlib and the same model files used for training must be present in models/:
    * shape_predictor_68_face_landmarks.dat
    * dlib_face_recognition_resnet_model_v1.dat
- numpy, opencv-python, geocoder (optional), scikit-learn not required.

Notes & tuning:
- EUCLIDEAN_THRESHOLD: typical dlib thresholds are ~0.4-0.6 for "same person",
  but this depends on data and capture conditions. Start at 0.6 and lower it if
  you get false positives.
- CONFIDENCE_PERCENT_THRESHOLD works on the derived percentage score; tune
  it alongside EUCLIDEAN_THRESHOLD.

Run:
    python recognize_dlib.py

"""

import os
import cv2
import sqlite3
import numpy as np
import uuid
import time
import math
import json
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep everything inside the same directory as this script
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCREEN_DIR = os.path.join(DATA_DIR, "screens")

os.makedirs(SCREEN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# DB path
DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")

# Legacy LBPH model path (fallback)
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

# Dlib-trained embeddings (produced by train_model_dlib.py)
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.npy")

# OpenCV cascade (used for fast detection fallback)
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------
# Config (tune these)
# ---------------------------
CONFIDENCE_PERCENT_THRESHOLD = 60
EUCLIDEAN_THRESHOLD = 0.6  # smaller => stricter
BOX_COLOR_UNKNOWN = (0, 255, 0)  # green
BOX_COLOR_MATCH = (0, 0, 255)  # red

# HUD toggles and styling (kept from original)
ENABLE_HUD = True
GLOW_THICKNESS = 8
HUD_ALPHA = 0.42
PANEL_PADDING = 8

# Dlib model filenames expected in MODELS_DIR
SHAPE_PREDICTOR_FN = "shape_predictor_68_face_landmarks.dat"
DM_FACE_RECOG_FN = "dlib_face_recognition_resnet_model_v1.dat"

# ---------------------------
# SQLite utilities (unchanged)
# ---------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
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


def load_person(uid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM persons WHERE unique_id=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


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


# ---------------------------
# Dlib helpers and model loader
# ---------------------------

def _load_dlib_models(models_dir):
    try:
        import dlib
    except Exception:
        raise RuntimeError("dlib not installed. Please `pip install dlib`.")

    sp_path = os.path.join(models_dir, SHAPE_PREDICTOR_FN)
    fr_path = os.path.join(models_dir, DM_FACE_RECOG_FN)

    if not os.path.exists(sp_path) or not os.path.exists(fr_path):
        raise FileNotFoundError(
            f"Missing dlib model files. Place {SHAPE_PREDICTOR_FN} and {DM_FACE_RECOG_FN} into {models_dir}"
        )

    sp = dlib.shape_predictor(sp_path)
    fr = dlib.face_recognition_model_v1(fr_path)
    detector = dlib.get_frontal_face_detector()
    return dlib, detector, sp, fr


def try_load_model():
    """
    Tries to load dlib embeddings first. If not present, falls back to LBPH if available.

    Returns (model_obj, label_map, model_type) where model_type is 'dlib' or 'lbph'.
    For dlib: model_obj is a dict with keys: mean_embeddings (Kx128), label_ids (K,), sp, fr, detector
    For lbph: model_obj is the cv2 recognizer object (like original script).
    """
    # prefer dlib embeddings
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(LABELS_PATH) and os.path.exists(LABEL_MAP_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
        labels = np.load(LABELS_PATH).astype(int)
        with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}

        # compute per-label mean embeddings for fast NN search
        unique_labels = np.unique(labels)
        mean_list = []
        label_ids = []
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            if len(idx) == 0:
                continue
            mean_vec = np.mean(embeddings[idx], axis=0)
            mean_list.append(mean_vec)
            label_ids.append(int(lab))

        mean_embeddings = np.vstack(mean_list).astype(np.float32)

        # load dlib models needed for computing descriptors at runtime
        try:
            dlib, detector, sp, fr = _load_dlib_models(MODELS_DIR)
        except Exception as e:
            raise RuntimeError(f"Failed to load dlib models: {e}")

        model_obj = {
            "type": "dlib",
            "mean_embeddings": mean_embeddings,
            "label_ids": np.array(label_ids, dtype=int),
            "embeddings": embeddings,
            "labels": labels,
            "sp": sp,
            "fr": fr,
            "detector": detector,
        }
        return model_obj, label_map, "dlib"

    # fallback to LBPH as before
    if os.path.exists(LEGACY_MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(LEGACY_MODEL_PATH)
            with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
                label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}
            return recognizer, label_map, "lbph"
        except Exception:
            pass

    raise FileNotFoundError("No usable model found. Run training (dlib) first.")


# ---------------------------
# HUD drawing helpers (copied from your original code)
# ---------------------------

def draw_text_with_outline(
    frame,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    color=(255, 255, 255),
    thickness=1,
):
    cv2.putText(frame, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_glow_box(frame, x, y, w, h, color, glow_thickness=GLOW_THICKNESS, alpha=0.12):
    overlay = frame.copy()
    for t in range(glow_thickness, 0, -2):
        cv2.rectangle(overlay, (x - t, y - t), (x + w + t, y + h + t), color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_corner_marks(frame, x, y, w, h, length=20, thickness=2, color=(0, 255, 0)):
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)


def draw_top_panel(frame, info_lines):
    overlay = frame.copy()
    x0, y0 = 5, 5
    h_txt = 17
    width = 360
    height = PANEL_PADDING * 2 + h_txt * len(info_lines)
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)
    for i, ln in enumerate(info_lines):
        draw_text_with_outline(
            frame,
            ln,
            (x0 + 10, y0 + PANEL_PADDING + (i + 1) * h_txt - 4),
            font_scale=0.5,
            color=(200, 200, 200),
            thickness=1,
        )


# ---------------------------
# Utility: compute "confidence" score from distance
# ---------------------------

def distance_to_score(distance, threshold=EUCLIDEAN_THRESHOLD):
    """
    Convert a Euclidean distance into a 0..100 score.
    If distance >= threshold -> score 0. If distance == 0 -> 100.
    """
    if distance >= threshold:
        return 0
    score = int(round(100.0 * (1.0 - (distance / float(threshold)))))
    return max(0, min(100, score))


# ---------------------------
# Geolocation helper (kept)
# ---------------------------

import geocoder


def get_machine_coordinates():
    try:
        location = geocoder.ip("me")
        if location.ok:
            lat, lng = location.latlng
            return f"{lat},{lng}"
    except Exception:
        pass
    return "unknown"


# ---------------------------
# Recognition loop
# ---------------------------

def run_recognition():
    init_db()
    try:
        model, label_map, model_type = try_load_model()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    coords = get_machine_coordinates()
    print(f"[INFO] Machine coords: {coords}")
    maps_link = f"https://www.google.com/maps?q={coords}"
    print("Google Maps Link:", maps_link)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    print("[INFO] Starting recognition. Press 'q' to exit.")

    last_logged_for_label = {}
    LOG_COOLDOWN_SECONDS = 5

    prev_time = time.time()
    fps = 0.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't read from camera. Exiting.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces quickly with OpenCV cascade for speed
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for x, y, w, h in faces:
            box_color = BOX_COLOR_UNKNOWN
            label_text = "Unknown"
            score_text = ""
            extra_lines = []

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            if model_type == "lbph":
                try:
                    face_crop = gray[y1:y2, x1:x2]
                    face_resize = cv2.resize(face_crop, (200, 200))
                    label_id, confidence = model.predict(face_resize)
                    score = max(0.0, 100.0 - confidence)
                    score_text = f"{score:.0f}%"
                    if label_id in label_map:
                        uid = label_map[label_id]
                        details = load_person(uid) or {}
                        name = details.get("name", uid)
                        label_text = f"{name} ({score_text})"
                        if score >= CONFIDENCE_PERCENT_THRESHOLD:
                            box_color = BOX_COLOR_MATCH
                            now_ts = time.time()
                            last = last_logged_for_label.get(uid, 0)
                            if now_ts - last > LOG_COOLDOWN_SECONDS:
                                last_logged_for_label[uid] = now_ts
                                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                fname = f"{ts}_{uid}_{str(uuid.uuid4())[:6]}.jpg"
                                fpath = os.path.join(SCREEN_DIR, fname)
                                cv2.imwrite(fpath, frame)
                                entry = {
                                    "detection_id": str(uuid.uuid4()),
                                    "unique_id": uid,
                                    "name": name,
                                    "score": score,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "coords": coords,
                                    "bbox": [int(x), int(y), int(w), int(h)],
                                    "screenshot": os.path.relpath(fpath, start=PROJECT_ROOT),
                                }
                                append_detection_log(entry)
                                print(f"[LOG] Detection saved: {entry['detection_id']} -> {name} ({score_text})")
                except Exception as e:
                    label_text = "Err"
                    print(f"[WARN] Predict error (LBPH): {e}")

            else:  # dlib pipeline
                try:
                    import dlib

                    # convert OpenCV rect to dlib rectangle
                    rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                    shape = model["sp"](rgb, rect)
                    descriptor = np.array(model["fr"].compute_face_descriptor(rgb, shape), dtype=np.float32)

                    # compute distances to per-label mean embeddings
                    dists = np.linalg.norm(model["mean_embeddings"] - descriptor.reshape(1, -1), axis=1)
                    idx = int(np.argmin(dists))
                    min_dist = float(dists[idx])
                    label_id = int(model["label_ids"][idx])

                    score = distance_to_score(min_dist, threshold=EUCLIDEAN_THRESHOLD)
                    score_text = f"{score:.0f}%"

                    if label_id in label_map:
                        uid = label_map[label_id]
                        details = load_person(uid) or {}
                        name = details.get("name", uid)
                        dob = details.get("dob", "")
                        age = details.get("age") or "Unknown"
                        label_text = f"{name} ({score_text})"
                        extra_lines.append(f"ID:{uid}")
                        extra_lines.append(f"Dist:{min_dist:.3f}")

                        if score >= CONFIDENCE_PERCENT_THRESHOLD:
                            box_color = BOX_COLOR_MATCH
                            now_ts = time.time()
                            last = last_logged_for_label.get(uid, 0)
                            if now_ts - last > LOG_COOLDOWN_SECONDS:
                                last_logged_for_label[uid] = now_ts
                                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                fname = f"{ts}_{uid}_{str(uuid.uuid4())[:6]}.jpg"
                                fpath = os.path.join(SCREEN_DIR, fname)
                                cv2.imwrite(fpath, frame)
                                entry = {
                                    "detection_id": str(uuid.uuid4()),
                                    "unique_id": uid,
                                    "name": name,
                                    "score": score,
                                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                    "coords": coords,
                                    "bbox": [int(x), int(y), int(w), int(h)],
                                    "screenshot": os.path.relpath(fpath, start=PROJECT_ROOT),
                                }
                                append_detection_log(entry)
                                print(f"[LOG] Detection saved: {entry['detection_id']} -> {name} ({score_text})")
                    else:
                        label_text = f"Label{label_id} ({score_text})"
                except Exception as e:
                    label_text = "Err"
                    print(f"[WARN] Predict error (dlib): {e}")

            # Draw HUD around face
            draw_glow_box(frame, x, y, w, h, box_color)
            draw_corner_marks(
                frame,
                x,
                y,
                w,
                h,
                length=max(12, int(min(w, h) * 0.15)),
                color=box_color,
            )

            if box_color == BOX_COLOR_MATCH:
                pulse = (math.sin(time.time() * 4) + 1) / 2
                center = (x + w // 2, y - 10)
                radius = int(6 + pulse * 6)
                cv2.circle(frame, center, radius, box_color, 1)
                draw_text_with_outline(
                    frame,
                    "LOCKED",
                    (center[0] - 30, center[1] + 6),
                    font_scale=0.5,
                    color=box_color,
                )

            draw_text_with_outline(frame, label_text, (x, y - 14), font_scale=0.6, color=box_color)
            for i, ln in enumerate(extra_lines):
                draw_text_with_outline(
                    frame,
                    ln,
                    (x, y + h + 20 + i * 18),
                    font_scale=0.5,
                    color=(230, 230, 230),
                )

        # HUD panel
        now = datetime.now(timezone.utc)
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - prev_time))
        prev_time = time.time()
        if ENABLE_HUD:
            info_lines = [
                f"TRACE-ML | FPS: {fps:.1f}",
                f"Coords: {coords}",
                f"Time(UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Model: {model_type}",
            ]
            draw_top_panel(frame, info_lines)

        cv2.imshow("TRACE-ML Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()

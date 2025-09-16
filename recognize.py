# recognize.py (HUD-enhanced, SQLite version)
# Integrated HUD/recognition overlays: glowing boxes, corner crosshairs,
# animated scanline, top-left status panel, pulsing lock indicator for matches.
# Keeps all original functionality: SQLite logging, screenshot saving, label loading.

import os
import cv2
import sqlite3
import numpy as np
import datetime
import uuid
import time
import math
import requests
from datetime import datetime, timezone

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

# HUD toggles and styling
ENABLE_HUD = True
SCANLINE_SPEED = 120  # pixels per second
GLOW_THICKNESS = 8
HUD_ALPHA = 0.42
PANEL_PADDING = 8

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
# Small HUD drawing helpers
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
    # draw thicker black text for better contrast
    cv2.putText(
        frame, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_glow_box(frame, x, y, w, h, color, glow_thickness=GLOW_THICKNESS, alpha=0.12):
    # Create overlay and paint multiple rectangles to give a glow-like effect,
    # then blend the overlay back.
    overlay = frame.copy()
    for t in range(glow_thickness, 0, -2):
        cv2.rectangle(overlay, (x - t, y - t), (x + w + t, y + h + t), color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_corner_marks(frame, x, y, w, h, length=20, thickness=2, color=(0, 255, 0)):
    # Top-left
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)


# def draw_scanline(frame, timestamp, color=(0, 255, 255)):
#     # Moving horizontal scanline across the frame
#     h = frame.shape[0]
#     line_y = int((timestamp * SCANLINE_SPEED) % h)
#     overlay = frame.copy()
#     cv2.line(overlay, (0, line_y), (frame.shape[1], line_y), color, 1)
#     # add a faint band below the line for effect
#     band_thickness = 8
#     cv2.rectangle(
#         overlay,
#         (0, max(0, line_y - band_thickness)),
#         (frame.shape[1], min(h, line_y + band_thickness)),
#         color,
#         -1,
#     )
#     cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)


def draw_top_panel(frame, info_lines):
    # Draw a semi-transparent panel on top-left with provided info lines
    overlay = frame.copy()
    x0, y0 = 5, 5
    # measure size
    h_txt = 17
    width = 290
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
# Recognition loop
# ---------------------------


def run_recognition():
    init_db()
    recognizer, label_map = try_load_model()
    if recognizer is None:
        return

    coords = get_machine_coordinates()
    print(f"[INFO] Machine coords: {coords}")
    maps_link = f"https://www.google.com/maps?q={coords}"
    print("Google Maps Link:", maps_link)

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to exit.")

    last_logged_for_label = {}
    LOG_COOLDOWN_SECONDS = 5

    # simple FPS calculation
    prev_time = time.time()
    fps = 0.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't read from camera. Exiting.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # # HUD: scanline and top panel
        # if ENABLE_HUD:
        #     draw_scanline(frame, t0)

        for x, y, w, h in faces:
            box_color = BOX_COLOR_UNKNOWN
            label_text = "Unknown"
            score_text = ""
            extra_lines = []

            # guard crop bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            try:
                face_crop = gray[y1:y2, x1:x2]
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

            # Draw fancy HUD around face
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

            # pulsing lock indicator for high confidence matches
            if box_color == BOX_COLOR_MATCH:
                pulse = (math.sin(time.time() * 4) + 1) / 2  # 0..1
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

            # draw label and extra info with outlines to keep readable on complex backgrounds
            draw_text_with_outline(
                frame, label_text, (x, y - 14), font_scale=0.6, color=box_color
            )
            for i, ln in enumerate(extra_lines):
                draw_text_with_outline(
                    frame,
                    ln,
                    (x, y + h + 20 + i * 18),
                    font_scale=0.5,
                    color=(230, 230, 230),
                )

        # Top-left status panel
        now = datetime.now(timezone.utc)
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - prev_time))
        prev_time = time.time()
        if ENABLE_HUD:
            info_lines = [
                f"TRACE-ML | FPS: {fps:.1f}",
                f"Coords: {coords}",
                f"Time(UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            draw_top_panel(frame, info_lines)

        cv2.imshow("TRACE-ML Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()

# train_model.py (SQLite version, patched to prevent dataset corruption)
import os
import cv2
import json
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split  # optional for accuracy check

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep data and dataset in the same directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

# Models folder inside same directory as train_model.py
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# SQLite DB path
DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")

# Model and label map paths
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------
# SQLite utils
# ---------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Ensure that the persons table exists."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_id TEXT UNIQUE,
            name TEXT,
            label INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def update_person_label(uid, label):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE persons SET label=? WHERE unique_id=?", (label, uid))
    conn.commit()
    conn.close()


# ---------------------------
# Data preparation
# ---------------------------
def prepare_training_data():
    persons = sorted(
        [
            d
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
    )
    faces, labels = [], []
    label_map = {}
    label_counter = 0

    for uid in persons:
        dirp = os.path.join(DATASET_DIR, uid)
        files = sorted(
            [
                os.path.join(dirp, f)
                for f in os.listdir(dirp)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )
        if not files:
            continue
        label_map[label_counter] = uid

        for fpath in files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape[:2]
            if h < 50 or w < 50:
                # try detecting face in original color image
                orig = cv2.imread(fpath)
                if orig is not None:
                    gray_full = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    faces_box = CASCADE.detectMultiScale(
                        gray_full, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
                    )
                    if len(faces_box) > 0:
                        x, y, w2, h2 = max(faces_box, key=lambda b: b[2] * b[3])
                        crop = gray_full[y : y + h2, x : x + w2]
                        if crop.size > 0:
                            face = cv2.resize(crop, (200, 200))
                            faces.append(face.astype("uint8"))
                            labels.append(label_counter)
                continue

            # normal resize path
            face = cv2.resize(img, (200, 200))
            faces.append(face.astype("uint8"))
            labels.append(label_counter)

        label_counter += 1

    labels = np.array(labels, dtype=np.int32)
    return faces, labels, label_map


# ---------------------------
# Training
# ---------------------------
def train_and_save():
    # Clean old model files, but never touch dataset/
    for f in [MODEL_PATH, LABEL_MAP_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Removed old model file: {f}")

    if not os.path.exists(DB_PATH):
        print(f"[WARN] Database not found at {DB_PATH}, creating a fresh one...")
    init_db()

    faces, labels, label_map = prepare_training_data()
    if len(faces) == 0:
        print("[ERROR] No faces found in dataset/. Please run collect script first.")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        print(
            "[ERROR] cv2.face.LBPHFaceRecognizer_create() not available. Install opencv-contrib-python."
        )
        return

    print(f"[INFO] Training on {len(faces)} samples, {len(set(labels))} classes.")
    recognizer.train(faces, labels)

    recognizer.write(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    with open(LABEL_MAP_PATH, "w", encoding="utf8") as f:
        json.dump(
            {str(k): v for k, v in label_map.items()}, f, indent=2, ensure_ascii=False
        )
    print(f"[INFO] Label map saved to {LABEL_MAP_PATH}")

    for k, uid in label_map.items():
        update_person_label(uid, int(k))
    print("[INFO] Updated persons table with label indices.")

    correct = 0
    for i, face in enumerate(faces):
        pred_label, conf = recognizer.predict(face)
        if pred_label == int(labels[i]):
            correct += 1
    acc = 100.0 * correct / len(faces)
    print(f"[INFO] Training-set accuracy (approx): {acc:.2f}%")

    try:
        if len(faces) >= 30:
            X, y = faces, labels
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(Xtr, ytr)
            correct = sum(
                1 for i, x in enumerate(Xte) if recognizer.predict(x)[0] == yte[i]
            )
            print(f"[INFO] Holdout accuracy approx: {100.0 * correct / len(Xte):.2f}%")
    except Exception:
        pass

    print("[DONE] Training finished.")


if __name__ == "__main__":
    train_and_save()

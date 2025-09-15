# train_model.py (Fully patched SQLite + OpenCV LBPH)
import os
import cv2
import json
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep data, dataset, and models in the same folder as this script
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Paths
DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")
MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

# Haar cascade for face detection
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------
# SQLite utilities
# ---------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Ensure the persons table exists."""
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
    faces = []
    labels = []
    label_map = {}
    label_counter = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
                print(f"[WARN] Cannot read image: {fpath}")
                continue

            h, w = img.shape[:2]
            if h < 50 or w < 50:
                # Detect face in original color image
                orig = cv2.imread(fpath)
                if orig is not None:
                    gray_full = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    faces_box = CASCADE.detectMultiScale(
                        gray_full, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
                    )
                    if len(faces_box) > 0:
                        faces_box = sorted(
                            faces_box, key=lambda b: b[2] * b[3], reverse=True
                        )
                        x, y, w2, h2 = faces_box[0]
                        crop = gray_full[y : y + h2, x : x + w2]
                        if crop.size > 0:
                            crop = cv2.resize(crop, (200, 200))
                            crop = clahe.apply(crop)
                            faces.append(crop.astype("uint8"))
                            labels.append(label_counter)
                    continue

            # Normal resize path
            face_resized = cv2.resize(img, (200, 200))
            face_resized = clahe.apply(face_resized)
            faces.append(face_resized.astype("uint8"))
            labels.append(label_counter)

        label_counter += 1

    labels = np.array(labels, dtype=np.int32)
    return faces, labels, label_map


# ---------------------------
# Training
# ---------------------------
def train_and_save():
    # Remove old model files
    for f in [MODEL_PATH, LABEL_MAP_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Removed old model file: {f}")

    # Ensure DB exists
    init_db()

    faces, labels, label_map = prepare_training_data()
    if len(faces) == 0:
        print("[ERROR] No faces found in dataset/. Please run collect script first.")
        return

    # LBPH recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
    except Exception:
        print(
            "[ERROR] cv2.face.LBPHFaceRecognizer_create() not available. Install opencv-contrib-python."
        )
        return

    print(f"[INFO] Training on {len(faces)} samples, {len(set(labels))} classes.")

    # Train model
    recognizer.train(faces, labels)

    # Save model and label map
    recognizer.write(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    label_map_strkeys = {str(k): v for k, v in label_map.items()}
    with open(LABEL_MAP_PATH, "w", encoding="utf8") as f:
        json.dump(label_map_strkeys, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Label map saved to {LABEL_MAP_PATH}")

    # Update labels in DB
    for k, uid in label_map.items():
        update_person_label(uid, int(k))
    print("[INFO] Updated persons table with label indices.")

    # Quick training-set accuracy
    correct = 0
    for i, face in enumerate(faces):
        pred_label, conf = recognizer.predict(face)
        if pred_label == labels[i]:
            correct += 1
    acc = 100.0 * correct / len(faces)
    print(f"[INFO] Training-set accuracy (approx): {acc:.2f}%")

    # Optional holdout accuracy
    try:
        if len(faces) >= 30:
            Xtr, Xte, ytr, yte = train_test_split(
                faces, labels, test_size=0.2, random_state=42, stratify=labels
            )
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=16, grid_x=8, grid_y=8
            )
            recognizer.train(Xtr, ytr)
            correct = sum(
                [1 for i, x in enumerate(Xte) if recognizer.predict(x)[0] == yte[i]]
            )
            print(f"[INFO] Holdout accuracy approx: {100.0 * correct / len(Xte):.2f}%")
    except Exception:
        pass

    print("[DONE] Training finished.")


if __name__ == "__main__":
    train_and_save()

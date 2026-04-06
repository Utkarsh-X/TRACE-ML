# train_model_dlib.py
"""
Replaces the LBPH-based training pipeline with a dlib embedding pipeline.

What this script does:
- Scans dataset/ (same layout: dataset/<unique_id>/*.jpg)
- Detects faces with dlib (falls back to OpenCV cascade if needed)
- Uses dlib shape predictor to get landmarks
- Computes 128-d face embeddings with dlib's face recognition model
- Saves embeddings, labels and label_map in models/
- Updates the SQLite persons table with integer label indices (same as before)

Files produced in models/:
- embeddings.npy   (shape: [N, 128])
- labels.npy       (shape: [N])       labels are integers indexing label_map
- label_map.json   (string keys of ints -> uid)

Requirements:
- dlib (and the two pretrained model files placed into MODELS_DIR):
    * shape_predictor_68_face_landmarks.dat
    * dlib_face_recognition_resnet_model_v1.dat
  If you don't have those files, download them from the dlib model zoo and place
  them into the `models/` directory next to this script.

Optional: scikit-learn (for a quick holdout KNN accuracy check)

Keep in mind: dlib can be heavy to install. If you cannot install dlib, you can
keep using your OpenCV/LBPH pipeline.
"""

import os
import sys
import cv2
import json
import sqlite3
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.npy")

# OpenCV cascade fallback (used only when dlib detector fails)
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Dlib model filenames (should be in MODELS_DIR)
SHAPE_PREDICTOR_FN = "shape_predictor_68_face_landmarks.dat"
DM_FACE_RECOG_FN = "dlib_face_recognition_resnet_model_v1.dat"

# ---------------------------
# SQLite utils (same as your original script)
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
# DLIB helpers and training data preparation
# ---------------------------


def _load_dlib_models(models_dir):
    try:
        import dlib
    except Exception as e:
        print(
            "[ERROR] dlib is not installed or cannot be imported. Please `pip install dlib`."
        )
        raise

    sp_path = os.path.join(models_dir, SHAPE_PREDICTOR_FN)
    fr_path = os.path.join(models_dir, DM_FACE_RECOG_FN)

    if not os.path.exists(sp_path) or not os.path.exists(fr_path):
        print("[ERROR] Required dlib model files not found in models/.")
        print(f"Place {SHAPE_PREDICTOR_FN} and {DM_FACE_RECOG_FN} into: {models_dir}")
        raise FileNotFoundError("Missing dlib model files.")

    sp = dlib.shape_predictor(sp_path)
    fr = dlib.face_recognition_model_v1(fr_path)
    detector = dlib.get_frontal_face_detector()
    return detector, sp, fr


def on_new_capture(input_data, uid, train_after_save=False, assume_bgr=False):
    import uuid
    import cv2
    import numpy as np
    import dlib

    try:
        detector, sp, fr = _load_dlib_models(MODELS_DIR)
    except Exception as e:
        print(f"[ERROR] cannot load dlib models: {e}")
        return []

    # --- Decode / normalize input ---
    if isinstance(input_data, bytes):
        nparr = np.frombuffer(input_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(input_data, np.ndarray):
        img_bgr = input_data.copy()
    else:
        print("[ERROR] Unsupported input type")
        return []

    if img_bgr is None:
        print("[ERROR] invalid image/frame")
        return []

    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)
    if img_bgr.ndim == 2:  # grayscale → BGR
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # --- Convert to RGB for dlib ---
    if assume_bgr:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_bgr

    dets = detector(rgb, 1)
    if len(dets) == 0:
        print("[WARN] no face detected")
        return []

    saved_paths = []
    out_dir = os.path.join(DATASET_DIR, uid)
    os.makedirs(out_dir, exist_ok=True)

    for rect in dets:
        shape = sp(rgb, rect)
        face_chip = dlib.get_face_chip(rgb, shape, size=150)
        fname = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR))
        saved_paths.append(out_path)

    if train_after_save and saved_paths:
        train_and_save()

    return saved_paths


def prepare_training_data_dlib():
    """
    Walks dataset/ and computes dlib 128-d embeddings for each detected face.

    Returns:
        embeddings: list of np.array(128,)
        labels: list of int (same length as embeddings)
        label_map: dict label_int -> unique_id
    """
    persons = sorted(
        [
            d
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
    )

    embeddings = []
    labels = []
    label_map = {}
    label_counter = 0

    if not persons:
        return embeddings, labels, label_map

    # Try to import dlib and models
    try:
        detector, sp, fr = _load_dlib_models(MODELS_DIR)
        import dlib

    except Exception:
        raise

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
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                print(f"[WARN] couldn't read image: {fpath}")
                continue

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # dlib detector wants a numpy array in RGB
            dets = detector(rgb, 1)

            rect = None
            if len(dets) > 0:
                # choose largest detected face
                rect = max(dets, key=lambda r: r.width() * r.height())
            else:
                # fallback: use OpenCV cascade on grayscale
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces_box = CASCADE.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
                )
                if len(faces_box) > 0:
                    x, y, w, h = max(faces_box, key=lambda b: b[2] * b[3])
                    # convert to dlib rect
                    from dlib import rectangle as dlib_rectangle

                    rect = dlib_rectangle(int(x), int(y), int(x + w), int(y + h))

            if rect is None:
                print(f"[WARN] No face detected in {fpath}; skipping")
                continue

            # shape -> landmarks
            shape = sp(rgb, rect)

            # you can use get_face_chip to get aligned face if you want to save images
            # face_chip = dlib.get_face_chip(rgb, shape, size=150)

            # compute 128-d descriptor
            try:
                descriptor = np.array(
                    fr.compute_face_descriptor(rgb, shape), dtype=np.float32
                )
            except Exception as e:
                print(f"[ERROR] failed to compute descriptor for {fpath}: {e}")
                continue

            if descriptor.shape[0] != 128:
                print(
                    f"[WARN] unexpected descriptor size for {fpath}: {descriptor.shape}"
                )
                continue

            embeddings.append(descriptor)
            labels.append(label_counter)

        label_counter += 1

    return embeddings, labels, label_map


# ---------------------------
# Training orchestration and saving
# ---------------------------


def train_and_save():
    # Clean old LBPH model if exists (not used anymore)
    old_model = os.path.join(MODELS_DIR, "face_model.yml")
    if os.path.exists(old_model):
        try:
            os.remove(old_model)
            print(f"[INFO] Removed old LBPH model file: {old_model}")
        except Exception:
            pass

    init_db()

    try:
        embeddings, labels, label_map = prepare_training_data_dlib()
    except FileNotFoundError:
        print("[ERROR] Dlib model files missing. See script header for instructions.")
        return
    except Exception as e:
        print(f"[ERROR] dlib pipeline failed: {e}")
        return

    if len(embeddings) == 0:
        print(
            "[ERROR] No embeddings created. Please check your dataset/ and models/ files."
        )
        return

    embeddings = np.vstack(embeddings).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Save arrays
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)

    with open(LABEL_MAP_PATH, "w", encoding="utf8") as f:
        json.dump(
            {str(k): v for k, v in label_map.items()}, f, indent=2, ensure_ascii=False
        )

    print(f"[INFO] Saved embeddings ({embeddings.shape}) to: {EMBEDDINGS_PATH}")
    print(f"[INFO] Saved labels ({labels.shape}) to: {LABELS_PATH}")
    print(f"[INFO] Saved label map to: {LABEL_MAP_PATH}")

    # update persons table with label indices
    for k, uid in label_map.items():
        try:
            update_person_label(uid, int(k))
        except Exception:
            pass
    print("[INFO] Updated persons table with label indices.")

    # Optional: quick holdout check using KNN (if scikit-learn present)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        if embeddings.shape[0] >= 10:
            Xtr, Xte, ytr, yte = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42, stratify=labels
            )
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(Xtr, ytr)
            score = knn.score(Xte, yte) * 100.0
            print(f"[INFO] Holdout KNN accuracy (1-NN): {score:.2f}%")
    except Exception:
        # sklearn not available or something failed; ignore
        pass

    print("[DONE] Dlib embedding training finished.")


if __name__ == "__main__":
    train_and_save()

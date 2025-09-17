# collect_faces.py (dlib version)
import os
import cv2
import sqlite3
import datetime
from InquirerPy import inquirer
import tkinter as tk
from tkinter import filedialog

# import dlib helpers from train_model_dlib
from train_model_dlib import on_new_capture, train_and_save

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep everything inside the same folder as this script
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")


# ------------------- Database helpers -------------------
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")  # safer writes
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS persons (
                    unique_id TEXT PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    severity TEXT,
                    dob TEXT,
                    gender TEXT,
                    last_seen_city TEXT,
                    last_seen_country TEXT,
                    date_added TEXT,
                    notes TEXT,
                    label TEXT
                )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    path TEXT,
                    FOREIGN KEY(person_id) REFERENCES persons(unique_id)
                )"""
            )
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Database initialization failed: {e}")


def save_person_db(entry):
    """Insert/update single person and images into DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO persons (
                    unique_id, name, category, severity, dob, gender,
                    last_seen_city, last_seen_country, date_added, notes, label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["unique_id"],
                    entry["name"],
                    entry["category"],
                    entry["severity"],
                    entry["dob"],
                    entry["gender"],
                    entry["last_seen"]["city"],
                    entry["last_seen"]["country"],
                    entry["date_added"],
                    entry["notes"],
                    entry["label"],
                ),
            )
            # Clear old images, then insert
            cur.execute("DELETE FROM images WHERE person_id=?", (entry["unique_id"],))
            for img_path in entry["images"]:
                cur.execute(
                    "INSERT INTO images (person_id, path) VALUES (?, ?)",
                    (entry["unique_id"], img_path),
                )
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Failed to save entry {entry['unique_id']}: {e}")


def generate_unique_id_for_category(category):
    """Generate next ID based on existing entries in DB: Criminal -> PRC###, Missing -> PRM###"""
    prefix = "PRC" if category == "Criminal" else "PRM"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT MAX(CAST(SUBSTR(unique_id, 4) AS INTEGER)) FROM persons WHERE unique_id LIKE ?",
                (prefix + "%",),
            )
            row = cur.fetchone()
            max_num = row[0] if row and row[0] is not None else 0
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Failed to generate unique ID: {e}")
        max_num = 0
    try:
        next_num = int(max_num) + 1
    except Exception:
        next_num = 1
    return f"{prefix}{next_num:03d}"


# ------------------- Capture helpers -------------------
def capture_from_webcam(unique_id, count_target=10):
    cap = cv2.VideoCapture(0)
    saved = 0
    print(
        f"[INFO] Starting webcam capture. Aim: {count_target} images. Press 'q' to stop."
    )

    while saved < count_target:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from webcam.")
            break

        # Use dlib pipeline to save aligned face
        saved_paths = on_new_capture(
            frame, uid=unique_id, train_after_save=False, assume_bgr=True
        )
        if saved_paths:
            saved += len(saved_paths)
            print(f"[INFO] Saved {saved}/{count_target}")

        cv2.imshow("Capture Faces (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


def import_images_via_dialog(unique_id):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    paths = filedialog.askopenfilenames(
        title="Select face images (5–50 jpg/png)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
    )
    root.destroy()

    if not paths:
        print("[INFO] No files selected.")
        return 0

    saved = 0
    for src in paths:
        try:
            img_bgr = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                print(f"[ERROR] Could not read {src}")
                continue

            # --- Normalize to 3-channel BGR, uint8 ---
            if img_bgr.dtype != "uint8":
                img_bgr = img_bgr.astype("uint8")

            if len(img_bgr.shape) == 2:  # grayscale → BGR
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:  # BGRA → BGR
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

            # Debug check
            print(f"[DEBUG] Loaded {src}: dtype={img_bgr.dtype}, shape={img_bgr.shape}")

            # Pass safe BGR image
            saved_paths = on_new_capture(
                img_bgr, uid=unique_id, train_after_save=False, assume_bgr=True
            )
            if saved_paths:
                saved += len(saved_paths)
                print(f"[INFO] Imported {src} -> {saved_paths}")
        except Exception as e:
            print(f"[ERROR] while processing {src}: {e}")
    return saved


# ------------------- Main workflow -------------------
def run_collect():
    init_db()
    print("== Collect Faces & Register Person ==")
    full_name = inquirer.text(message="Full Name:").execute()

    category = inquirer.select(
        message="Category:", choices=["Criminal", "Missing"]
    ).execute()
    unique_id = generate_unique_id_for_category(category)
    print(f"[INFO] Assigned Unique ID: {unique_id}")

    severity = None
    if category == "Criminal":
        severity = inquirer.select(
            message="Severity level:", choices=["1 - Low", "2 - Medium", "3 - High"]
        ).execute()

    dob = inquirer.text(message="DOB (DD-MM-YYYY):", default="").execute()
    gender = inquirer.select(
        message="Gender:", choices=["Male", "Female", "Other", "Prefer not to say"]
    ).execute()
    last_seen_city = inquirer.text(message="Last seen - City:", default="").execute()
    last_seen_country = inquirer.text(
        message="Last seen - Country:", default=""
    ).execute()
    notes = inquirer.text(message="Notes (optional):", default="").execute()

    how = inquirer.select(
        message="How to provide photos?",
        choices=["Use Webcam", "Upload images (file dialog)"],
    ).execute()

    if how == "Use Webcam":
        num_imgs = int(
            inquirer.number(
                message="How many images to capture (5-50)?",
                default=10,
                min_allowed=5,
                max_allowed=100,
            ).execute()
        )
        saved = capture_from_webcam(unique_id, count_target=num_imgs)
    else:
        saved = import_images_via_dialog(unique_id)

    now_iso = datetime.datetime.now().isoformat()
    # collect relative paths of saved images
    person_folder = os.path.join(DATASET_DIR, unique_id)
    image_files = sorted(
        [
            os.path.relpath(os.path.join(person_folder, f), start=PROJECT_ROOT)
            for f in os.listdir(person_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    entry = {
        "name": full_name,
        "unique_id": unique_id,
        "category": category,
        "severity": severity if severity else "",
        "dob": dob,
        "gender": gender,
        "last_seen": {"city": last_seen_city, "country": last_seen_country},
        "date_added": now_iso,
        "notes": notes,
        "images": image_files,
        "label": None,
    }

    save_person_db(entry)
    print(
        f"[DONE] Saved {saved} face images for {full_name} ({unique_id}). Metadata stored in SQLite DB"
    )

    # retrain once at the end
    print("[INFO] Updating embeddings...")
    train_and_save()
    print("[DONE] You can now run recognition.")


if __name__ == "__main__":
    run_collect()

# collect_faces.py
import os
import cv2
import sqlite3
import shutil
import datetime
from InquirerPy import inquirer
import tkinter as tk
from tkinter import filedialog

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep everything inside the same folder as this script
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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


def load_person_db():
    """Return dict like before (for compatibility)."""
    db = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM persons")
            persons = cur.fetchall()

            for p in persons:
                (
                    unique_id,
                    name,
                    category,
                    severity,
                    dob,
                    gender,
                    last_seen_city,
                    last_seen_country,
                    date_added,
                    notes,
                    label,
                ) = p

                cur.execute("SELECT path FROM images WHERE person_id=?", (unique_id,))
                images = [row[0] for row in cur.fetchall()]

                db[unique_id] = {
                    "name": name,
                    "unique_id": unique_id,
                    "category": category,
                    "severity": severity,
                    "dob": dob,
                    "gender": gender,
                    "last_seen": {"city": last_seen_city, "country": last_seen_country},
                    "date_added": date_added,
                    "notes": notes,
                    "images": images,
                    "label": label,
                }
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Failed to load DB: {e}")
    return db


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


# ------------------- File helpers -------------------
def next_image_index(save_dir):
    """Return next integer index for image filenames in save_dir (avoids overwriting)."""
    if not os.path.exists(save_dir):
        return 1
    files = [
        f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    nums = []
    for f in files:
        name = os.path.splitext(f)[0]
        try:
            nums.append(int(name))
        except Exception:
            pass
    return max(nums) + 1 if nums else 1


# ------------------- Face capture helpers -------------------
def capture_from_webcam(save_dir, count_target=10, skip_seconds=0.2):
    os.makedirs(save_dir, exist_ok=True)
    start_index = next_image_index(save_dir)
    cap = cv2.VideoCapture(0)
    saved = 0
    print(
        f"[INFO] Starting webcam capture. Aim: {count_target} face images. Press 'q' to quit earlier."
    )
    while saved < count_target:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from webcam.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # Draw boxes and display
        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (200, 200))
            saved += 1
            fname = os.path.join(save_dir, f"{start_index + saved - 1}.jpg")
            cv2.imwrite(fname, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Saved {saved}/{count_target}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            print(f"[INFO] Saved {fname}")
            if saved >= count_target:
                break

        cv2.imshow("Capture Faces - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


def import_images_via_dialog(save_dir):
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

    os.makedirs(save_dir, exist_ok=True)
    start_index = next_image_index(save_dir)
    saved = 0
    for src in paths:
        try:
            img = cv2.imread(src)
            if img is None:
                print(f"[WARN] Could not read {src}, skipping.")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) == 0:
                face = cv2.resize(gray, (200, 200))
            else:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = faces[0]
                face = gray[y : y + h, x : x + w]
                face = cv2.resize(face, (200, 200))
            saved += 1
            dest = os.path.join(save_dir, f"{start_index + saved - 1}.jpg")
            cv2.imwrite(dest, face)
            print(f"[INFO] Imported and saved {dest}")
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

    # generate unique id automatically based on category and DB
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

    person_folder = os.path.join(DATASET_DIR, unique_id)
    os.makedirs(person_folder, exist_ok=True)

    if how == "Use Webcam":
        num_imgs = int(
            inquirer.number(
                message="How many images to capture (5-50)?",
                default=10,
                min_allowed=5,
                max_allowed=100,
            ).execute()
        )
        saved = capture_from_webcam(person_folder, count_target=num_imgs)
    else:
        saved = import_images_via_dialog(person_folder)

    now_iso = datetime.datetime.now().isoformat()
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
    print("You can now run the training script to create/update the recognition model.")

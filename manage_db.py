import os
import sqlite3
from InquirerPy import inquirer, prompt
from tabulate import tabulate  # <-- added for table view

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "person_db.sqlite")


# ------------------- Database helpers -------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def fetch_all_persons():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT unique_id, name FROM persons")
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_person(unique_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM persons WHERE unique_id=?", (unique_id,))
    person = cur.fetchone()
    if not person:
        conn.close()
        return None

    (
        uid,
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
    ) = person

    cur.execute("SELECT path FROM images WHERE person_id=?", (uid,))
    images = [row[0] for row in cur.fetchall()]

    conn.close()
    return {
        "unique_id": uid,
        "name": name,
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


def update_person(record):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """UPDATE persons SET
            name=?, dob=?, gender=?, last_seen_city=?, last_seen_country=?, notes=?
        WHERE unique_id=?""",
        (
            record["name"],
            record["dob"],
            record["gender"],
            record["last_seen"]["city"],
            record["last_seen"]["country"],
            record["notes"],
            record["unique_id"],
        ),
    )
    conn.commit()
    conn.close()


def delete_person(unique_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM images WHERE person_id=?", (unique_id,))
    cur.execute("DELETE FROM persons WHERE unique_id=?", (unique_id,))
    conn.commit()
    conn.close()


# ------------------- CLI workflow -------------------
def show_list_and_manage():
    persons = fetch_all_persons()
    if not persons:
        print("[INFO] No records found.")
        return

    choices = [f"{name} :: {uid}" for uid, name in persons]
    choices.append("<< Back >>")
    ans = inquirer.select(
        message="Select record to view/edit/delete:", choices=choices
    ).execute()
    if ans == "<< Back >>":
        return

    uid = ans.split("::")[-1].strip()
    record = fetch_person(uid)
    if not record:
        print("[ERROR] Record not found.")
        return

    # Display in SQL-like table format instead of JSON
    headers = ["Field", "Value"]
    table = [
        ("Unique ID", record["unique_id"]),
        ("Name", record["name"]),
        ("Category", record["category"]),
        ("Severity", record["severity"]),
        ("DOB", record["dob"]),
        ("Gender", record["gender"]),
        ("Last Seen City", record["last_seen"]["city"]),
        ("Last Seen Country", record["last_seen"]["country"]),
        ("Date Added", record["date_added"]),
        ("Notes", record["notes"]),
        ("Label", record["label"]),
        ("Images", ", ".join(record["images"]) if record["images"] else "-"),
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    action = inquirer.select(
        message="Action:", choices=["Edit", "Delete", "Back"]
    ).execute()
    if action == "Back":
        return
    if action == "Delete":
        confirm = inquirer.confirm(
            message=f"Delete record {uid}? This is irreversible.", default=False
        ).execute()
        if confirm:
            delete_person(uid)
            print("[INFO] Deleted.")
        return
    if action == "Edit":
        questions = [
            {
                "type": "input",
                "name": "name",
                "message": "Full Name:",
                "default": record.get("name", ""),
            },
            {
                "type": "input",
                "name": "dob",
                "message": "DOB (DD-MM-YYYY):",
                "default": record.get("dob", ""),
            },
            {
                "type": "input",
                "name": "gender",
                "message": "Gender:",
                "default": record.get("gender", ""),
            },
            {
                "type": "input",
                "name": "city",
                "message": "Last seen city:",
                "default": record.get("last_seen", {}).get("city", ""),
            },
            {
                "type": "input",
                "name": "country",
                "message": "Last seen country:",
                "default": record.get("last_seen", {}).get("country", ""),
            },
            {
                "type": "input",
                "name": "notes",
                "message": "Notes:",
                "default": record.get("notes", ""),
            },
        ]
        answers = prompt(questions)
        record["name"] = answers.get("name", record.get("name"))
        record["dob"] = answers.get("dob", record.get("dob"))
        record["gender"] = answers.get("gender", record.get("gender"))
        record["last_seen"] = {
            "city": answers.get("city"),
            "country": answers.get("country"),
        }
        record["notes"] = answers.get("notes", record.get("notes"))
        update_person(record)
        print("[INFO] Saved edits.")


def run_manage():
    init_db()
    while True:
        choice = inquirer.select(
            message="DB Manager - choose:", choices=["View/Edit Records", "Exit"]
        ).execute()
        if choice == "Exit":
            break
        if choice == "View/Edit Records":
            show_list_and_manage()

from InquirerPy import inquirer

from collect_faces import run_collect
from train_model import train_and_save
from recognize import run_recognition
from manage_db import run_manage


def main_menu():
    while True:
        choice = inquirer.select(
            message="TRACE-ML: Choose action",
            choices=[
                "1) Collect Faces & Register Person",
                "2) Train Model (LBPH)",
                "3) Run Recognition (webcam)",
                "4) Manage Person DB (view/edit/delete)",
                "5) Exit",
            ],
        ).execute()

        if choice.startswith("1"):
            run_collect()
        elif choice.startswith("2"):
            train_and_save()
        elif choice.startswith("3"):
            run_recognition()
        elif choice.startswith("4"):
            run_manage()
        elif choice.startswith("5"):
            print("Goodbye.")
            break


main_menu()
=======
from InquirerPy import inquirer

from collect_faces import run_collect
from train_model import train_and_save
from recognize import run_recognition
from manage_db import run_manage


def main_menu():
    while True:
        choice = inquirer.select(
            message="TRACE-ML: Choose action",
            choices=[
                "1) Collect Faces & Register Person",
                "2) Train Model (LBPH)",
                "3) Run Recognition (webcam)",
                "4) Manage Person DB (view/edit/delete)",
                "5) Exit",
            ],
        ).execute()

        if choice.startswith("1"):
            run_collect()
        elif choice.startswith("2"):
            train_and_save()
        elif choice.startswith("3"):
            run_recognition()
        elif choice.startswith("4"):
            run_manage()
        elif choice.startswith("5"):
            print("Goodbye.")
            break


main_menu()
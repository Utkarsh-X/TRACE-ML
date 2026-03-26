"""Data collection helpers for person registration."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trace_ml.core.config import Settings
from trace_ml.core.errors import CameraError


def person_image_dir(settings: Settings, person_id: str) -> Path:
    path = Path(settings.store.root) / "person_images" / person_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def capture_from_webcam(
    settings: Settings,
    person_id: str,
    count: int = 10,
    auto: bool = True,
    interval_seconds: float = 0.35,
) -> list[str]:
    cap = cv2.VideoCapture(settings.camera.device_index)
    if not cap.isOpened():
        raise CameraError(f"Unable to open webcam {settings.camera.device_index}")

    saved: list[str] = []
    out_dir = person_image_dir(settings, person_id)
    title = f"Capture for {person_id} | q=quit | space=manual-shot"
    next_auto_at = time.time() + interval_seconds

    while len(saved) < count:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()
        cv2.putText(
            frame,
            f"Captured {len(saved)}/{count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
        mode = "AUTO" if auto else "MANUAL"
        cv2.putText(
            frame,
            f"Mode: {mode} | q=quit | space=manual-shot",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        should_capture = False
        if key == ord(" "):
            should_capture = True
        elif auto and now >= next_auto_at:
            should_capture = True
            next_auto_at = now + interval_seconds

        if should_capture:
            out = out_dir / f"capture_{len(saved) + 1:03d}.jpg"
            cv2.imwrite(str(out), frame)
            saved.append(str(out))

    cap.release()
    cv2.destroyAllWindows()
    return saved


def import_from_directory(settings: Settings, person_id: str, source_dir: str) -> list[str]:
    src = Path(source_dir)
    if not src.exists():
        raise FileNotFoundError(f"Directory not found: {source_dir}")
    out_dir = person_image_dir(settings, person_id)
    saved: list[str] = []
    for file in sorted(src.iterdir()):
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(file))
        if img is None:
            continue
        out = out_dir / f"import_{len(saved) + 1:03d}.jpg"
        cv2.imwrite(str(out), img)
        saved.append(str(out))
    return saved

"""Capture worker for webcam frames.

Provides a class-level shared frame buffer so external consumers (e.g. the
MJPEG service endpoint) can read the latest frame without opening a second
camera device.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from trace_aml.core.config import Settings
from trace_aml.core.errors import CameraError


@dataclass
class FramePacket:
    frame: np.ndarray
    captured_at: float
    frame_index: int


class CameraCapture:
    """Threaded webcam capture with a shared latest-frame buffer."""

    # ── Class-level shared state (thread-safe) ──
    _shared_lock = threading.Lock()
    _shared_latest: FramePacket | None = None
    _shared_active: bool = False

    def __init__(self, settings: Settings, frame_queue: queue.Queue[FramePacket]) -> None:
        self.settings = settings
        self.frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    # ── Class methods for external consumers ──

    @classmethod
    def is_active(cls) -> bool:
        """Return True if a CameraCapture instance is currently running."""
        with cls._shared_lock:
            return cls._shared_active

    @classmethod
    def get_latest_frame(cls) -> FramePacket | None:
        """Return the most recent frame captured, or None if inactive."""
        with cls._shared_lock:
            return cls._shared_latest

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.settings.camera.device_index)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open webcam {self.settings.camera.device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.settings.camera.fps)

        with CameraCapture._shared_lock:
            CameraCapture._shared_active = True

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._cap is not None
        frame_idx = 0
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            packet = FramePacket(frame=frame, captured_at=time.time(), frame_index=frame_idx)
            frame_idx += 1

            # Update shared latest frame for external consumers (MJPEG endpoint).
            with CameraCapture._shared_lock:
                CameraCapture._shared_latest = packet

            # Push into the inference pipeline queue.
            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(packet)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        with CameraCapture._shared_lock:
            CameraCapture._shared_active = False
            CameraCapture._shared_latest = None

"""Inference worker for threaded pipeline."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from trace_aml.core.config import Settings
from trace_aml.core.models import RecognitionMatch
from trace_aml.liveness.base import LivenessResult
from trace_aml.pipeline.capture import FramePacket
from trace_aml.recognizers.arcface import ArcFaceRecognizer
from trace_aml.store.vector_store import VectorStore


@dataclass
class InferencePacket:
    frame: np.ndarray
    frame_index: int
    captured_at: float
    processed_at: float
    matches: list[RecognitionMatch]
    embeddings: list[list[float]]
    liveness: list[LivenessResult]


class InferenceWorker:
    def __init__(
        self,
        recognizer: ArcFaceRecognizer,
        store: VectorStore,
        frame_queue: queue.Queue[FramePacket],
        result_queue: queue.Queue[InferencePacket],
        settings: Settings | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.store = store
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # ── CPU offload settings ──────────────────────────────────────────
        # How many frames to skip between each inference call.
        # 0 = every frame, 1 = every other, 2 = every 3rd, etc.
        # When settings=None (unit-test context) default to 0 so every frame
        # is processed and tests don't need to think about frame counters.
        self._skip: int = int(getattr(settings.pipeline, "inference_skip_frames", 0)) if settings else 0
        # Scale factor applied to frame dimensions before ML inference.
        # The original full-res frame is kept in the output packet.
        self._scale: float = float(getattr(settings.pipeline, "inference_resolution_scale", 1.0)) if settings else 1.0
        self._frame_counter: int = 0
        # ──────────────────────────────────────────────────────────────────


    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # ── Frame skip ───────────────────────────────────────────────
            # Increment counter on every frame.  Only run inference on
            # frame 0, (skip+1), 2*(skip+1), …  All other frames are
            # discarded here before touching the GPU or gallery cache.
            self._frame_counter += 1
            if self._skip > 0 and (self._frame_counter % (self._skip + 1)) != 0:
                continue
            # ─────────────────────────────────────────────────────────────

            # ── Resolution downscale for inference ───────────────────────
            # Fewer pixels = faster ONNX decode + smaller GPU texture upload.
            # The original full-res frame travels forward in the packet so
            # the overlay renderer / screenshot saver use the full image.
            infer_frame = packet.frame
            if self._scale != 1.0:
                h, w = infer_frame.shape[:2]
                new_w = max(1, int(w * self._scale))
                new_h = max(1, int(h * self._scale))
                infer_frame = cv2.resize(
                    infer_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
            # ─────────────────────────────────────────────────────────────

            recognized = self.recognizer.match(infer_frame, self.store)
            matches = [item[0] for item in recognized]
            embeddings = [item[1] for item in recognized]
            liveness = [item[2] for item in recognized]

            # Re-scale bounding boxes back to full-res coordinates so the
            # overlay renderer draws boxes on the correct positions.
            if self._scale != 1.0:
                inv = 1.0 / self._scale
                for match in matches:
                    x, y, bw, bh = match.bbox
                    match.bbox = (
                        int(x * inv),
                        int(y * inv),
                        int(bw * inv),
                        int(bh * inv),
                    )

            out = InferencePacket(
                frame=packet.frame,          # full-res original
                frame_index=packet.frame_index,
                captured_at=packet.captured_at,
                processed_at=time.time(),
                matches=matches,
                embeddings=embeddings,
                liveness=liveness,
            )
            try:
                self.result_queue.put_nowait(out)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_queue.put_nowait(out)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

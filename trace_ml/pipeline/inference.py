"""Inference worker for threaded pipeline."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

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
    ) -> None:
        self.recognizer = recognizer
        self.store = store
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            recognized = self.recognizer.match(packet.frame, self.store)
            matches = [item[0] for item in recognized]
            embeddings = [item[1] for item in recognized]
            liveness = [item[2] for item in recognized]
            out = InferencePacket(
                frame=packet.frame,
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

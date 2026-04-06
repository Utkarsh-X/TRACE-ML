import queue
import time

import numpy as np

from trace_aml.core.models import RecognitionMatch
from trace_aml.liveness.base import LivenessResult
from trace_aml.pipeline.capture import FramePacket
from trace_aml.pipeline.inference import InferenceWorker


class _FakeRecognizer:
    def match(self, frame, store):
        match = RecognitionMatch(
            person_id="PRC001",
            name="Test",
            category="criminal",
            similarity=0.9,
            confidence=90.0,
            bbox=(0, 0, 10, 10),
            is_match=True,
        )
        return [(match, [0.0] * 512, LivenessResult(True, 1.0, "none"))]


class _FakeStore:
    pass


def test_inference_worker_processes_frame() -> None:
    frame_q: queue.Queue = queue.Queue(maxsize=2)
    result_q: queue.Queue = queue.Queue(maxsize=2)
    worker = InferenceWorker(_FakeRecognizer(), _FakeStore(), frame_q, result_q)
    worker.start()
    frame_q.put(FramePacket(frame=np.zeros((32, 32, 3), dtype=np.uint8), captured_at=time.time(), frame_index=1))
    time.sleep(0.2)
    worker.stop()
    assert not result_q.empty()
    packet = result_q.get_nowait()
    assert len(packet.matches) == 1
    assert packet.matches[0].is_match is True

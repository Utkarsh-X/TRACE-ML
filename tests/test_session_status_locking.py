import threading
from types import SimpleNamespace

from trace_aml.core.errors import DependencyError
from trace_aml.pipeline.session import RecognitionSession


def test_headless_loop_does_not_block_camera_status(monkeypatch) -> None:
    session = RecognitionSession.__new__(RecognitionSession)
    session._camera_lock = threading.Lock()
    session._camera_enabled = False
    session._recognition_enabled = False
    session._result_queue = None
    session._frame_queue = None
    session._inference = None
    session._capture = None
    session._headless_mode = False
    session.last_frame_queue_depth = 0
    session.last_result_queue_depth = 0
    session.last_latency_ms = 0.0
    session.settings = SimpleNamespace(
        camera=SimpleNamespace(device_index=0, width=640, height=480, fps=30)
    )
    session._start_writer = lambda: None
    session._stop_writer = lambda: None

    entered_sleep = threading.Event()
    release_sleep = threading.Event()

    def fake_sleep(_seconds: float) -> None:
        entered_sleep.set()
        release_sleep.wait(timeout=1.0)
        raise SystemExit

    monkeypatch.setattr("trace_aml.pipeline.session.time.sleep", fake_sleep)

    thread_error: list[BaseException] = []

    def run_headless() -> None:
        try:
            session.run_headless()
        except SystemExit:
            return
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            thread_error.append(exc)

    worker = threading.Thread(target=run_headless, daemon=True)
    worker.start()
    assert entered_sleep.wait(timeout=0.5)

    status_holder: dict[str, object] = {}

    def read_status() -> None:
        status_holder["value"] = session.get_camera_status()

    reader = threading.Thread(target=read_status, daemon=True)
    reader.start()
    reader.join(timeout=0.2)

    release_sleep.set()
    worker.join(timeout=1.0)

    assert not thread_error
    assert not reader.is_alive()
    assert status_holder["value"]["enabled"] is False


def test_enable_recognition_fails_fast_when_recognizer_dependency_is_missing() -> None:
    class _BrokenRecognizer:
        def _ensure_app(self) -> None:
            raise DependencyError("InsightFace not available. Install dependencies from requirements.txt")

    session = RecognitionSession.__new__(RecognitionSession)
    session._camera_lock = threading.Lock()
    session._camera_enabled = True
    session._recognition_enabled = False
    session._recognition_error = ""
    session._result_queue = object()
    session._frame_queue = object()
    session._inference = None
    session._capture = None
    session.recognizer = _BrokenRecognizer()
    session.store = object()
    session.settings = SimpleNamespace()
    session._handle_inference_fatal_error = lambda exc: None

    result = session.enable_recognition()

    assert result["status"] == "error"
    assert "InsightFace not available" in result["message"]
    assert session._recognition_enabled is False

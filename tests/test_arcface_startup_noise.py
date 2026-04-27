import sys
import types

from trace_aml.core.config import Settings
import trace_aml.recognizers.arcface as arcface_module
from trace_aml.recognizers.arcface import ArcFaceRecognizer


def test_ensure_app_suppresses_insightface_startup_output(monkeypatch, capsys) -> None:
    class _FakeFaceAnalysis:
        def __init__(self, name, providers):
            print(f"Applied providers: {providers}")

        def prepare(self, ctx_id, det_size):
            sys.stderr.write(f"find model: fake-model.onnx det_size={det_size}\n")

    fake_insightface = types.SimpleNamespace(
        app=types.SimpleNamespace(FaceAnalysis=_FakeFaceAnalysis)
    )
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setattr(
        arcface_module,
        "detect_providers",
        lambda preferred, allow_gpu, cuda_device_id: ["CPUExecutionProvider"],
    )

    recognizer = ArcFaceRecognizer(Settings())
    recognizer._ensure_app()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

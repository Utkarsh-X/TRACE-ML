from pathlib import Path

from trace_aml.core.config import load_settings
from trace_aml.core.logger import configure_logger


def _settings(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
camera:
  device_index: 0
logging:
  file_path: {tmp_path.as_posix()}/logs/trace_aml.log
store:
  root: {tmp_path.as_posix()}/data
  vectors_dir: {tmp_path.as_posix()}/data/vectors
  screenshots_dir: {tmp_path.as_posix()}/data/screens
  exports_dir: {tmp_path.as_posix()}/data/exports
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_configure_logger_falls_back_when_enqueue_file_sink_is_denied(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)

    calls = []

    class DummyLogger:
        def remove(self):
            return None

        def add(self, sink, **kwargs):
            calls.append((sink, kwargs))
            if len(calls) == 2 and kwargs.get("enqueue") is True:
                raise PermissionError("[WinError 5] Access is denied")
            return len(calls)

        def warning(self, *_args, **_kwargs):
            calls.append(("warning", {}))
            return None

    monkeypatch.setattr("trace_aml.core.logger.logger", DummyLogger())

    configure_logger(settings)

    assert len(calls) >= 4
    assert calls[1][1]["enqueue"] is True
    assert calls[2][0] == "warning"
    assert calls[3][1]["enqueue"] is False

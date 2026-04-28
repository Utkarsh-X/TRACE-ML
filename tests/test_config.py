import importlib
from pathlib import Path

import pytest

import trace_aml.core.config as config_module


def test_load_settings_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
camera:
  device_index: 0
store:
  root: demo_data
  vectors_dir: demo_data/vectors
  screenshots_dir: demo_data/screens
  exports_dir: demo_data/exports
""".strip(),
        encoding="utf-8",
    )
    settings = config_module.load_settings(cfg)
    assert settings.camera.device_index == 0
    assert settings.store.vectors_dir == "demo_data/vectors"
    assert settings.quality.min_valid_images >= 1
    assert settings.temporal.decision_window >= 1
    assert settings.recognition.accept_threshold > settings.recognition.review_threshold


def test_rejects_non_zero_camera(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
camera:
  device_index: 1
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(config_module.ConfigError):
        config_module.load_settings(cfg)


def test_portable_defaults_follow_trace_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACE_DATA_ROOT", "portable_data")
    importlib.reload(config_module)
    try:
        assert config_module.StoreSettings().vectors_dir == "portable_data/vectors"
        assert config_module.PdfReportSettings().output_dir == "portable_data/exports"
        assert config_module.LoggingSettings().file_path == "portable_data/logs/trace_aml.log"
    finally:
        monkeypatch.delenv("TRACE_DATA_ROOT", raising=False)
        importlib.reload(config_module)

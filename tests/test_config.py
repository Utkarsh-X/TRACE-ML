from pathlib import Path

import pytest

from trace_ml.core.config import ConfigError, load_settings


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
    settings = load_settings(cfg)
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
    with pytest.raises(ConfigError):
        load_settings(cfg)

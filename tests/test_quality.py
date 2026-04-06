from pathlib import Path

import numpy as np

from trace_aml.core.config import load_settings
from trace_aml.core.models import PersonLifecycleStatus
from trace_aml.quality.gating import decide_person_lifecycle
from trace_aml.quality.scoring import score_face_image


def _settings(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
camera:
  device_index: 0
store:
  root: {tmp_path.as_posix()}/data
  vectors_dir: {tmp_path.as_posix()}/data/vectors
  screenshots_dir: {tmp_path.as_posix()}/data/screens
  exports_dir: {tmp_path.as_posix()}/data/exports
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_quality_scoring_accepts_reasonable_face(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(42)
    frame[60:180, 100:220] = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)

    score = score_face_image(settings, frame, bbox=(100, 60, 120, 120))
    assert score.passed is True
    assert score.quality_score >= settings.quality.min_quality_score
    assert score.reasons == []


def test_quality_scoring_rejects_low_quality_face(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    score = score_face_image(settings, frame, bbox=(0, 0, 20, 20))

    assert score.passed is False
    assert "face_too_small" in score.reasons
    assert "too_dark" in score.reasons
    assert "quality_below_threshold" in score.reasons


def test_lifecycle_gating_transitions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    no_images = decide_person_lifecycle(settings, total_images=0, valid_images=0, embeddings_count=0, avg_quality=0.0)
    assert no_images.state == PersonLifecycleStatus.draft

    no_embeddings = decide_person_lifecycle(
        settings, total_images=10, valid_images=0, embeddings_count=0, avg_quality=0.7
    )
    assert no_embeddings.state == PersonLifecycleStatus.blocked

    draft_not_ready = decide_person_lifecycle(
        settings,
        total_images=10,
        valid_images=2,
        embeddings_count=settings.quality.min_embeddings_ready - 1,
        avg_quality=0.7,
    )
    assert draft_not_ready.state == PersonLifecycleStatus.draft

    blocked_quality = decide_person_lifecycle(
        settings,
        total_images=10,
        valid_images=10,
        embeddings_count=settings.quality.min_embeddings_active,
        avg_quality=settings.quality.min_quality_score - 0.01,
    )
    assert blocked_quality.state == PersonLifecycleStatus.blocked

    ready = decide_person_lifecycle(
        settings,
        total_images=10,
        valid_images=settings.quality.min_valid_images - 1,
        embeddings_count=settings.quality.min_embeddings_active,
        avg_quality=0.8,
    )
    assert ready.state == PersonLifecycleStatus.ready

    active = decide_person_lifecycle(
        settings,
        total_images=12,
        valid_images=settings.quality.min_valid_images,
        embeddings_count=settings.quality.min_embeddings_active,
        avg_quality=0.8,
    )
    assert active.state == PersonLifecycleStatus.active

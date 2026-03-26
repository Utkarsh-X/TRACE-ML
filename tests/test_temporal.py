from pathlib import Path

from trace_ml.core.config import load_settings
from trace_ml.core.models import DecisionState, RecognitionMatch
from trace_ml.pipeline.temporal import TemporalDecisionEngine


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
recognition:
  accept_threshold: 0.72
  review_threshold: 0.58
temporal:
  decision_window: 6
  smoothing_alpha: 0.6
  min_accept_votes: 2
  track_ttl_seconds: 2.0
  max_track_distance_px: 120
""".strip(),
        encoding="utf-8",
    )
    return load_settings(cfg)


def _match(
    confidence: float = 80.0,
    person_id: str = "PRC001",
    bbox: tuple[int, int, int, int] = (100, 100, 80, 80),
    flags: list[str] | None = None,
) -> RecognitionMatch:
    return RecognitionMatch(
        person_id=person_id,
        name="Alice",
        category="criminal",
        similarity=confidence / 100.0,
        confidence=confidence,
        bbox=bbox,
        is_match=True,
        quality_flags=flags or [],
        candidate_scores=[{"person_id": person_id, "name": "Alice", "similarity": confidence / 100.0}],
    )


def test_temporal_promotes_review_to_accept(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = TemporalDecisionEngine(settings)

    d1 = engine.evaluate(_match(), now_ts=100.0)
    d2 = engine.evaluate(_match(), now_ts=100.1)

    assert d1.decision_state == DecisionState.review
    assert d2.decision_state == DecisionState.accept
    assert d1.track_id == d2.track_id


def test_temporal_rejects_low_confidence_and_inactive_flags(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = TemporalDecisionEngine(settings)

    low = engine.evaluate(_match(confidence=42.0), now_ts=200.0)
    assert low.decision_state == DecisionState.reject

    inactive = engine.evaluate(_match(flags=["person_not_active"]), now_ts=200.1)
    assert inactive.decision_state == DecisionState.reject
    assert inactive.decision_reason == "person_not_active"

    live_fail = engine.evaluate(_match(flags=["liveness_fail"]), now_ts=200.2)
    assert live_fail.decision_state == DecisionState.reject
    assert live_fail.decision_reason == "liveness_reject"


def test_temporal_assigns_new_track_when_far_apart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = TemporalDecisionEngine(settings)

    near_a = engine.evaluate(_match(bbox=(100, 100, 80, 80)), now_ts=300.0)
    near_b = engine.evaluate(_match(bbox=(110, 105, 80, 80)), now_ts=300.1)
    far = engine.evaluate(_match(bbox=(500, 100, 80, 80)), now_ts=300.2)

    assert near_a.track_id == near_b.track_id
    assert far.track_id != near_a.track_id
